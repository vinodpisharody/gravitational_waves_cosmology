from scipy.integrate import quad
from scipy.integrate import tplquad
import ray
import numpy as np
from scipy import interpolate
import warnings
import functools
import matplotlib.pyplot as plt
import binary_snr as bns
import pandas as pd
warnings.filterwarnings("ignore",category=RuntimeWarning)
G=6.67*10**(-11)
M=2*10**30
c=299792458
year=365*24*3600
mpc=3.08568*10**22

data_stellar=pd.read_csv('stellar_pop3_opt.csv')

def detection_factor(m1,m2,z,detector=None):
    if m1<m2:
        raise ValueError('Primary mass cannot be less than secondary mass')
    if detector not in ['LISA', 'CE1','CE2','ET']:
        raise ValueError('Unknown Detector')
    columns = ['m1', 'm2', 'LISA', 'CE1','CE2','ET']
    df = pd.DataFrame(data_stellar, columns=columns)
    matching_rows = df[(df['m1'] == round(m1)) & (df['m2'] == round(m2))]
    if z>matching_rows.iloc[0][detector]:
         return 1
    else:
         return 0
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
    """
    A class to compute Stochastic gravitational wave background for a given population model using Phinney(2003)
    Attributes:
        merger(tuple): A size 3 tuple containing one variable merger rate as lambda function z_min and z_max as floats 
        primary_mass (tuple) :A size 3 tuple containing one variable probability function for larger mass as lambda function and m_min and m_max as floats
        secondary_mass (tuple) :A size 3 tuple containing one variable probability function for smaller mass as lambda function and m_min and m_max as floats
        H0 (float) : Present day value of Hubble's Constant in km/(Mpc*year)
        om_m (float) : Matter density
        om_m (float) : Vacuum density
        frequency_band(numpy array) : An array of frequency values to compute the background 
        detector(string):LISA or DECIGO to only consider the inpiral case .Default None to include the merger and ringdown phase
        verbose(bool):To show/hide warnings
    """
    def __init__(self,merger,primary_mass,secondary_mass,H0=70,om_m=0.3,om_l=0.7,R0=lambda m1,m2:1,frequency_band=None,detector=None,verbose=True):
        self.H0=H0
        self.om_m=om_m
        self.om_l=om_l
        if isinstance(merger, tuple):
            self.merger=merger
        else:
            raise TypeError('Need to give a tuple with merger rate function and z_min and z_max')
        if isinstance(primary_mass, tuple):
            self.primary_mass=primary_mass
        else:
            raise TypeError('Need to give a tuple with primary mass distribution function and m1_min and m1_max')
        if isinstance(secondary_mass, tuple):
            self.secondary_mass=secondary_mass
        else:
            raise TypeError('Need to give a tuple with primary mass distribution function and m2_min and m2_max')
        if isinstance(frequency_band,list) or isinstance(frequency_band,np.ndarray):
            self.frequency_band=frequency_band
        else:
            raise ValueError("Frequency not specified")
        self.detector=detector
        if detector==None:
            raise ValueError('Detector not specified')
        elif detector=='LISA' and verbose==True:
            warnings.warn('You have not specified the detector.So all of the inspiral-merger-ringdown will be considered in background',category=UserWarning)
        self.verbose=verbose
        self.R0=R0
    def get_background(self,tol_m=0.01,tol_z=0.01):
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
        @memoize(maxsize=1000)
        def normalizer(f,m_min,m_max):
            return(quad(lambda m:f(m), m_min, m_max, epsrel=1, epsabs=1)[0])
        f=self.frequency_band
        rho=(3*(self.H0*1000*((mpc)**-1))**2)/(8*np.pi*G)
        prefactor=(M**(5/3)*((G*np.pi)**(2/3)))/(3*rho*(c**2)*(self.H0*1000*((mpc)**-1))*((mpc*1000)**3)*year)
        probability_function_primary  =  lambda m,z : self.primary_mass[0](m,z);m1_min=self.primary_mass[1];m1_max=self.primary_mass[2]
        probability_function_secondary=  lambda m,z : self.secondary_mass[0](m,z);m2_min=self.secondary_mass[1];m2_max=self.secondary_mass[2]
        m=np.linspace(5,100,endpoint=False)
        z_lower,z_upper=self.merger[1],self.merger[2];rem_rate=lambda z:self.merger[0](z)
        global omega
        g_sw=np.empty(0)
        probability_function=lambda m1,m2,z:probability_function_primary(m1,z)*probability_function_secondary(m2,z)
        @ray.remote
        def omega(f):
                integrand = lambda z, m1, m2:(10**5)*prefactor*f*waveform_factor(f,z,m1,m2,detector=self.detector)*(detection_factor(m1=m1, m2=m2, z=z, detector=self.detector))*((m1*m2)/(m1+m2)**(1/3)) \
                *probability_function(m1, m2,z)*(1/((((self.om_m*((1+z)**3))+self.om_l)**(0.5))*(1+z)))*(self.R0(m1,m2)*rem_rate(z)
                                /(normalizer(lambda m:probability_function_secondary(m,z),m2_min,m2_max)*normalizer(lambda m:probability_function_primary(m,z),m2,m1_max)))
                def m1_integral(m2, z):
                    a=quad(lambda m1: integrand(z, m1, m2),m2, m1_max, epsrel=tol_m, epsabs=0,limit=100)
                    return a[0]/10**5
                def m2_integral(z):
                    return quad(lambda m2: m1_integral(m2, z), m2_min, m2_max, epsrel=tol_m, epsabs=0,limit=100)[0]
                result = quad(lambda z: m2_integral(z), z_lower, z_upper, epsrel=tol_z, epsabs=0,limit=100)
                return  result[0]
        if len(f)<10:
            return np.array(ray.get([omega.remote(i) for i in f]))
        lower_band_cutoff=waveform_factor(0,0,m1_max, m1_max,flag='inspiral')/(1+z_upper)
        upper_band_cutoff=waveform_factor(0,0, m1_min, m1_min,flag='cutoff')
        condition_upper =( f>=upper_band_cutoff)
        mask_upper = np.where(condition_upper)
        upper_f = f[mask_upper]
        g_upper=np.zeros(np.size(upper_f))
        condition_lower = (f<=lower_band_cutoff)
        mask_lower = np.where(condition_lower)
        lower_f = f[mask_lower]
        condition_middle = (f>lower_band_cutoff) & (f<upper_band_cutoff)
        mask_middle= np.where(condition_middle)
        middle_f_ = f[mask_middle]
        if len(middle_f_)==0:
            g_lower=lower_f**(2/3)*(ray.get(omega.remote(lower_band_cutoff))/(lower_band_cutoff)**(2/3))
            return np.concatenate((g_lower,g_upper))
        else:
            if self.verbose==True:
                print("Big computation",flush=True)
            else:
                pass
        if len(middle_f_)>=100:
            if self.verbose==True:
                print('Sampling 100 linearly spaced frequency bins between {}Hz and {}Hz'.format(min(middle_f_),max(middle_f_)),flush=True)
            middle_f=np.linspace(min(middle_f_),max(middle_f_),100)
            g_sw=[omega.remote(i) for i in middle_f]
            g_sw.append(omega.remote(lower_band_cutoff))
            ans=np.array(ray.get(g_sw))
            g_sw=ans[:-1]
            g_lower=lower_f**(2/3)*(ans[-1]/(lower_band_cutoff)**(2/3))
            temp=interpolate.interp1d(x=middle_f, y=g_sw,fill_value='extrapolate',kind='cubic')(middle_f_)
            g_sw=temp
        else:
            g_sw=[omega.remote(i) for i in middle_f_]
            g_sw.append(omega.remote(lower_band_cutoff))
            ans=np.array(ray.get(g_sw))
            g_sw=ans[:-1]
            g_lower=lower_f**(2/3)*(ans[-1]/(lower_band_cutoff)**(2/3))
        if self.verbose==True:
            print("Completed")
        return np.concatenate((g_lower,g_sw,g_upper))
