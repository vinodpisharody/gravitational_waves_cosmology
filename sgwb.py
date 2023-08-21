from scipy.integrate import quad
import ray
import numpy as np
from scipy import interpolate
import warnings
import functools
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore",category=RuntimeWarning)
G=6.67*10**(-11)
M=2*10**30
c=299792458
year=365*24*3600
mpc=3.08568*10**22
def waveform_factor(f,z,m1,m2,flag=None,detector=None):
  fm=[2.9740*10**(-1),4.4810*10**(-2),9.5560*10**(-2)]
  fri=[5.9411*10**(-1),8.9794*10**(-2),19.111*10**(-2)]
  fc=[8.4845*10**(-1),12.848*10**(-2),27.299*10**(-2)]
  fw=[5.0801*10**(-1),7.7515*10**(-2),2.2369*10**(-2)]
  M_m=m1+m2
  eta=m1*m2/(M_m**2)
  f_merg=((c**3)*(fm[0]*eta*eta+fm[1]*eta+fm[2]))/(np.pi*G*M_m*M)
  f_ring=((c**3)*(fri[0]*eta*eta+fri[1]*eta+fri[2]))/(np.pi*G*M_m*M)
  f_cut=((c**3)*(fc[0]*eta*eta+fc[1]*eta+fc[2]))/(np.pi*G*M_m*M)
  f_w=((c**3)*(fw[0]*eta*eta+fw[1]*eta+fw[2]))/(np.pi*G*M_m*M)
  fr=f*(1+z)
  f_isco=(c**3)/(G*(6**(3/2))*M_m*M*np.pi)
  if flag=='inspiral':
      return f_merg
  elif flag=='cutoff':
      return f_cut
  elif flag=='ringdown':
      return f_ring
  if detector=='LISA' or detector=='DECIGO':
      if fr<f_isco:
        return (fr**(-1/3))
      else:
          return 0
  else:
      if fr<f_merg:
        return (fr**(-1/3))
      elif fr>=f_merg and fr<f_ring:
        return (fr**(2/3)/f_merg)
      elif fr>=f_ring and fr<f_cut:
        return ((1/(f_merg*f_ring**(4/3)))*(fr/(1+((fr-f_ring)/(f_w/2))**2))**2)
      else:
        return 0

@ray.remote
class background():
    def __init__(self,merger,primary_mass,secondary_mass,H0=70,om_m=0.3,om_l=0.7,frequency_band=None,detector=None,verbose=True):
        self.H0=H0
        self.om_m=om_m
        self.om_l=om_l
        if isinstance(merger, tuple):
            self.merger=merger
        else:
            raise TypeError('Need to give a tuple with merger rate fucntion and z_min and z_max')
        if isinstance(primary_mass, tuple):
            self.primary_mass=primary_mass
        else:
            raise TypeError('Need to give a tuple with primary mass distribution fucntion and m1_min and m1_max')
        if isinstance(secondary_mass, tuple):
            self.secondary_mass=secondary_mass
        else:
            raise TypeError('Need to give a tuple with primary mass distribution fucntion and m1_min and m1_max')
        if isinstance(frequency_band,list) or isinstance(frequency_band,np.ndarray):
            self.frequency_band=frequency_band
        else:
            raise ValueError("Frequency not specified")
        self.detector=detector
        if detector==None and verbose==True:
            warnings.warn('You have not specified the detector.So all of the inspiral-merger-ringdown will be considered in background',category=UserWarning)

    def get_background(self):
        def memoize(maxsize=None):
            def decorator(func):
                cache = {}
                @functools.wraps(func)
                def memoized_func(*args):
                    if args in cache:
                        return cache[args]
                    result = func(*args)
                    cache[args] = result
                    if maxsize is not None and len(cache) > maxsize:
                        # Remove the oldest item (FIFO eviction).
                        cache.pop(next(iter(cache)))
                    return result
                return memoized_func        
            return decorator
        @memoize(maxsize=10000)
        def normalizer(f,m_min,m_max):
            return(quad(lambda m:f(m), m_min, m_max, epsrel=1, epsabs=1)[0])
        f=self.frequency_band
        rho=(3*(self.H0*1000*((mpc)**-1))**2)/(8*np.pi*G)
        prefactor=(M**(5/3)*((G*np.pi)**(2/3)))/(3*rho*(c**2)*(self.H0*1000*((mpc)**-1))*((mpc*1000)**3)*year)
        probability_function_primary  =  lambda m,z : self.primary_mass[0](m,z);m1_min=self.primary_mass[1];m1_max=self.primary_mass[2]
        probability_function_secondary=  lambda m,z : self.secondary_mass[0](m,z);m2_min=self.secondary_mass[1];m2_max=self.secondary_mass[2]
        z_lower,z_upper=self.merger[1],self.merger[2];rem_rate=lambda z:self.merger[0](z)
        global omega
        g_sw=np.empty(0)
        probability_function=lambda m1,m2,z:probability_function_primary(m1,z)*probability_function_secondary(m2,z)
        @ray.remote
        def omega(f):
                integrand = lambda z, m1, m2: waveform_factor(f,z,m1,m2,detector=self.detector)*((m1*m2)/(m1+m2)**(1/3)) \
                *probability_function(m1, m2,z)*(1/((((self.om_m*((1+z)**3))+self.om_l)**(0.5))*(1+z)))*(rem_rate(z)
                                /(normalizer(lambda m:probability_function_secondary(m,z),m2_min,m2_max)*normalizer(lambda m:probability_function_primary(m,z),m2,m1_max)))
                def m1_integral(m2, z):
                    return quad(lambda m1: integrand(z, m1, m2),m2, m1_max, epsrel=10, epsabs=10)[0]
                def m2_integral(z):
                    return quad(lambda m2: m1_integral(m2, z), m2_min, m2_max, epsrel=10, epsabs=10)[0]
                result, _ = quad(lambda z: m2_integral(z), z_lower, z_upper, epsrel=1, epsabs=1)
                return prefactor * f * result
        if len(f)<10:
            return ray.get(omega.remote(f))
        lower_band_cutoff=waveform_factor(0,0,m1_max, m1_max,flag='inspiral')/(1+z_upper)
        upper_band_cutoff=waveform_factor(0,0, m1_min, m1_min,flag='cutoff')
        condition_upper =( f>=upper_band_cutoff)
        mask_upper = np.where(condition_upper)
        upper_f = f[mask_upper]
        g_upper=np.zeros(np.size(upper_f))
        condition_lower = (f<=lower_band_cutoff)
        mask_lower = np.where(condition_lower)
        lower_f = f[mask_lower]
        g_lower=lower_f**(2/3)*(ray.get(omega.remote(lower_band_cutoff))/(lower_band_cutoff)**(2/3))
        condition_middle = (f>lower_band_cutoff) & (f<upper_band_cutoff)
        mask_middle= np.where(condition_middle)
        middle_f_ = f[mask_middle]
        middle_f=np.geomspace(lower_band_cutoff,upper_band_cutoff,100)
        g_sw=np.array(ray.get([omega.remote(i) for i in middle_f]))
        g_sw=interpolate.interp1d(middle_f,g_sw,kind='linear',fill_value='extrapolate')(middle_f_)
        return np.concatenate((g_lower,g_sw,g_upper))
