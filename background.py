from scipy.integrate import quad,IntegrationWarning
import ray
import numpy as np
from scipy.interpolate import interp1d as ipd
import warnings
import functools
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.optimize import minimize_scalar
import pandas as pd
import random
from scipy.integrate import simpson
from tqdm import tqdm
import emcee
from astropy import constants as const 
from astropy import units as u 
from astropy.cosmology import Planck18 as cosmo
from scipy.ndimage import median_filter
warnings.filterwarnings("ignore",category=RuntimeWarning)
warnings.filterwarnings("ignore",category=IntegrationWarning)
c=const.c.value
G=const.G.value
M=const.M_sun.value
gpc=const.pc.value*10**9
mpc=const.pc.value*10**6
year=365*24*3600
gyr=year*10**9
H=cosmo.H0.value*1000*(mpc)**-1
om_m=cosmo.Om0
om_l=cosmo.Ode0
om_k = cosmo.Ogamma0
window_size=250
freq_lisa = np.loadtxt('LISA_fit.txt',usecols=0)
sensitivity_lisa = np.loadtxt('LISA_fit.txt',usecols=1)
lisa_psd = ipd(freq_lisa,sensitivity_lisa,fill_value='extrapolate',kind='cubic')
freq_ce=np.loadtxt('CE.txt',delimiter=',',usecols=0)
sensitivity_ce1=np.loadtxt('CE.txt',delimiter=',',usecols=1)
sensitivity_ce2=np.loadtxt('CE.txt',delimiter=',',usecols=2)




freq_et=np.loadtxt('ET.txt',delimiter=',',usecols=0)
sensitivity_et=np.loadtxt('ET.txt',delimiter=',',usecols=1)



freq_lgwa = np.loadtxt('LGWA_Nb_psd.txt',usecols=0,delimiter=',')
sensitivity_lgwa_nb = np.loadtxt('LGWA_Nb_psd.txt',usecols=1,delimiter=',')
sensitivity_lgwa_nb_smooth = median_filter(sensitivity_lgwa_nb, size=window_size)
lgwa1_psd=ipd(freq_lgwa,sensitivity_lgwa_nb_smooth,fill_value='extrapolate',kind='cubic')
# 9 LGWA (Sb)
sensitivity_lgwa_si = np.loadtxt('LGWA_Si_psd.txt',usecols=1,delimiter=',')
sensitivity_lgwa_si_smooth = median_filter(sensitivity_lgwa_si, size=window_size)
lgwa2_psd=ipd(freq_lgwa,sensitivity_lgwa_si_smooth,fill_value='extrapolate',kind='cubic')
# 10 LGWA (Soundcheck)
sensitivity_lgwa_soundcheck = np.loadtxt('LGWA_Soundcheck_psd.txt',usecols=1,delimiter=',')
sensitivity_lgwa_soundcheck_smooth = median_filter(sensitivity_lgwa_soundcheck, size=window_size)
lgwa3_psd=ipd(freq_lgwa,sensitivity_lgwa_soundcheck_smooth,fill_value='extrapolate',kind='cubic')


freq_ligo= np.loadtxt('LIGO.txt',usecols=0,delimiter=',')
sensitivity_ligo = np.loadtxt('LIGO.txt',usecols=1,delimiter=',')


data_=pd.read_csv(r'z_min_IMBH_8.csv')                       ## Load appropriate file that contains z_min for a grid of source frame masses 
x = data_['m1']  
y = data_['m2']
detector_conf=['LISA','LGWA1','LGWA2','CE1+CE1+ET','CE2+CE2+ET']

interp_conf=[CloughTocher2DInterpolator(list(zip(x,y)),data_[_],maxiter=1000) for _ in detector_conf]
z_min_database = {'LISA':interp_conf[0],'LGWA1':interp_conf[1],'LGWA2':interp_conf[2],'CE1+CE1+ET':interp_conf[3],'CE2+CE2+ET':interp_conf[4]}
del x;del y
def detection_factor(m1,m2,z,f,detector=None,err=1,source_type='averaged'):
        """
        Parameters
        ----------
        m1 : float
            Source frame mass of the primary mass in the binary system
        m2 : float
            Source frame mass of the secondary mass in the binary system
        z : float
            Distance of the binary in terms of redshift 
        f : float
            Frequency at which binary is emitting Gravitational wave (In detector frame)
        detector : String
            Detector .The value depends on what detectors are there in the zmin data that is loaded. The default is None.
        err : float, optional
            The maximum value of the integrand of match-filtered SNR that will not contribute to background. The default is 1.
        source_type : string, optional
            type of source .Could be either 'averaged' or 'optimal' [face on orientation]. The default is 'averaged'.
    
        Returns
        -------
        int
            1 if the source contributes to the background and 0 if the source can be detected 
    
        """
        if m1==m2:
            pass
        else:
            temp1=m1;temp2=m2
            m1=max(temp1,temp2);m2=min(temp1,temp2)
        if m1<m2:
            raise ValueError('Primary mass cannot be less than secondary mass')
        check=z_min_database[detector](m1,m2)
        if np.isnan(check):
            raise ValueError('NaN encountered for m1={},m2={},configuration={} at z={}'.format(m1,m2,detector,z))
        else:
            if z>=check:
                return 1
            else:
                if detector in ['CE1+CE1+ET','CE2+CE2+ET','CE1+CE1','CE2+CE2','LVK','CE1','CE2','ET','LIGO']:
                    return(0)
                fm = [2.9740*10**(-1), 4.4810*10**(-2), 9.5560*10**(-2)]
                fri = [5.9411*10**(-1), 8.9794*10**(-2), 19.111*10**(-2)]
                fc = [8.4845*10**(-1), 12.848*10**(-2), 27.299*10**(-2)]
                fw = [5.0801*10**(-1), 7.7515*10**(-2), 2.2369*10**(-2)]
                m1=m1*(1+z)
                m2=m2*(1+z)
                M_total = m1+m2
                eta = m1*m2/(M_total**2)
                M_chirp=(eta**(3/5))*M_total
                M_x=(G*M_chirp*M)/c**3
                f_merg = ((c**3)*(fm[0]*eta*eta+fm[1]*eta+fm[2]))/(np.pi*G*M_total*M)
                f_ring = ((c**3)*(fri[0]*eta*eta+fri[1]*eta+fri[2]))/(np.pi*G*M_total*M)
                f_cut = ((c**3)*(fc[0]*eta*eta+fc[1]*eta+fc[2]))/(np.pi*G*M_total*M)
                f_w = ((c**3)*(fw[0]*eta*eta+fw[1]*eta+fw[2]))/(np.pi*G*M_total*M)
                w=(np.pi*f_w/2)*(f_merg/f_ring)**(2/3)
                factor=((np.sqrt(5/24))*(M_x**(5/6))*(f_merg**(-7/6)))/((np.pi**(2/3))*(cosmo.luminosity_distance(z).to_value('m')/c))
                def waveform(f):
                    _=f
                    if _ < f_merg:
                        return(factor * (_/f_merg)**(-7/6))
                    elif _ >= f_merg and _ < f_ring:
                        return(factor * (_/f_merg)**(-2/3))
                    elif _ >= f_ring and _ < f_cut:
                        return(factor * (w/(2*np.pi))*(f_w/((_-f_ring)**2+((f_w**2)/4))))
                    else:
                        return(factor * 0)
                if detector == 'LISA':
                    P = lambda f:lisa_psd(f)
                    if source_type=='averaged':
                        integrand = lambda f : (16/5)*waveform(f)**2/P(f)**2          ## factor of 4/5 when averaged over inclination
                    else:
                        integrand = lambda f : (4)*waveform(f)**2/P(f)**2            ## Factor of 1 because cos(inclination)=1 but response fucntion is present
                elif detector=='LGWA1':
                    P=lambda f:lgwa1_psd(f)
                    if source_type=='averaged':
                        integrand = lambda f : (16/5)*waveform(f)**2/P(f)         ## factor of 4/5 when averaged over inclination
                    else:
                        integrand = lambda f : (8)*waveform(f)**2/P(f)            ## Factor of 1 because cos(inclination)=1 but response fucntion is present
                elif detector=='LGWA2':
                    P=lambda f:lgwa2_psd(f)
                    if source_type=='averaged':
                        integrand = lambda f : (16/5)*waveform(f)**2/P(f)         ## factor of 4/5 when averaged over inclination
                    else:
                        integrand = lambda f : (8)*waveform(f)**2/P(f)            ## Factor of 1 because cos(inclination)=1 but response fucntion is present
                elif detector=='LGWA3':
                    P=lambda f:lgwa3_psd(f)
                    if source_type=='averaged':
                        integrand = lambda f : (16/5)*waveform(f)**2/P(f)         ## factor of 4/5 when averaged over inclination
                    else:
                        integrand = lambda f : (8)*waveform(f)**2/P(f)            ## Factor of 1 because cos(inclination)=1 but response fucntion is present
                if integrand(f)*f<err:
                    return(1)
                else:
                    return 0
def match_filtered_snr(m1,m2,z,detector=None,threshold=(8,1),quantity='snr',source_type='averaged',freq_band=None):
    """
    Parameters
    ----------
    m1 : float
        Source frame mass of the primary mass in the binary system
    m2 : float
        Source frame mass of the secondary mass in the binary system
    z : float
        Distance of the binary in terms of redshift 
    detector : String
        Detector .The value depends on what detectors are there in the zmin data that is loaded. The default is None.
    threshold : tuple, optional
        tuple containing the threshold value of SNR for which source is detectable and the maximum value of the integrand of match-filtered SNR that will not contribute to background . The default is (8,0.1).
    quantity : string, optional
        Has to be either 'snr' or 'statistics' or 'detection' which gives the snr for the given source-detector , detectibility as 0 or 1 and again detectibility but taking into account frequency and the integrand of the SNR formula respectively. The default is 'snr'.
    source_type : string, optional
        type of source .Could be either 'averaged' or 'optimal' [face on orientation]. The default is 'averaged'.
    freq_band : numpy array, optional
        Computes the detectibility of the source in this frequency band if detector is andy of 'LISA,LGWA1,LGWA2,LGWA3' . The default is None.

    Returns
    -------
    detection_factor : int
        0 or 1 depending on if source is detectable based on the cutoff criteria of the snr if quantity='detection'
    snr : float
        match filtered snr of the source if quantity='snr' 
    list of the form [m1,m2,z, 0 or 1] if quantity='statistics' 
    """
    if detector not in ['LISA','LGWA1','LGWA2','LGWA3','CE1+CE1+ET','CE2+CE2+ET','LVK','Null','CE1','CE2','ET','LIGO']:
        raise ValueError('Invalid Detector')
    if quantity not in ['snr','detection','statistics']:
        raise ValueError('quantity has to be either snr or detector or statistics')
    if source_type not in ['averaged','optimal']:
        raise ValueError('source_type has to be either averaged or optimal')
    fm = [2.9740*10**(-1), 4.4810*10**(-2), 9.5560*10**(-2)]
    fri = [5.9411*10**(-1), 8.9794*10**(-2), 19.111*10**(-2)]
    fc = [8.4845*10**(-1), 12.848*10**(-2), 27.299*10**(-2)]
    fw = [5.0801*10**(-1), 7.7515*10**(-2), 2.2369*10**(-2)]
    m1=m1*(1+z)
    m2=m2*(1+z)
    M_total = m1+m2
    eta = m1*m2/(M_total**2)
    M_chirp=(eta**(3/5))*M_total
    M_x=(G*M_chirp*M)/c**3
    f_merg = ((c**3)*(fm[0]*eta*eta+fm[1]*eta+fm[2]))/(np.pi*G*M_total*M)
    f_ring = ((c**3)*(fri[0]*eta*eta+fri[1]*eta+fri[2]))/(np.pi*G*M_total*M)
    f_cut = ((c**3)*(fc[0]*eta*eta+fc[1]*eta+fc[2]))/(np.pi*G*M_total*M)
    f_w = ((c**3)*(fw[0]*eta*eta+fw[1]*eta+fw[2]))/(np.pi*G*M_total*M)
    w=(np.pi*f_w/2)*(f_merg/f_ring)**(2/3)
    factor=((np.sqrt(5/24))*(M_x**(5/6))*(f_merg**(-7/6)))/((np.pi**(2/3))*(cosmo.luminosity_distance(z).to_value('m')/c))
    def waveform(f):
        _=f
        if _ < f_merg:
            return(factor * (_/f_merg)**(-7/6))
        elif _ >= f_merg and _ < f_ring:
            return(factor * (_/f_merg)**(-2/3))
        elif _ >= f_ring and _ < f_cut:
            return(factor * (w/(2*np.pi))*(f_w/((_-f_ring)**2+((f_w**2)/4))))
        else:
            return(factor * 0)
    if detector == 'LISA':
        P = lambda f:lisa_psd(f)
        if source_type=='averaged':
            integrand = lambda f : (16/5)*waveform(f)**2/P(f)**2          ## factor of 4/5 when averaged over inclination
        else:
            integrand = lambda f : (4)*waveform(f)**2/P(f)**2            ## Factor of 2 because cos(inclination)=1
        min_f = min(((f_cut)**(-8/3) + 1*year *256*np.pi**(8/3)/5 *M_x**(5/3))**(-3/8),(0.1**(-8/3) + 1*year *256*np.pi**(8/3)/5 *M_x**(5/3))**(-3/8))
        max_f = min(1,f_cut)
        snr = np.sqrt(quad(integrand,min_f,max_f,epsabs=0,epsrel=0.1,limit=500)[0])
    elif detector=='LGWA1':
        if source_type=='averaged':
            integrand =(16/5)*np.array([waveform(f) for f in freq_lgwa])**2/sensitivity_lgwa_nb
        else:
            integrand =(8)*np.array([waveform(f) for f in freq_lgwa])**2/sensitivity_lgwa_nb
        snr=np.sqrt(2*simpson(integrand,freq_lgwa))
        
        if source_type=='averaged':
            integrand=lambda f : 16/5 * waveform(f)**2/lgwa1_psd(f)
        else:
            integrand=lambda f : 8 * waveform(f)**2/lgwa1_psd(f)
    elif detector=='LGWA2':
        if source_type=='averaged':
            integrand =(16/5)*np.array([waveform(f) for f in freq_lgwa])**2/sensitivity_lgwa_si
        else:
            integrand =(8)*np.array([waveform(f) for f in freq_lgwa])**2/sensitivity_lgwa_si
        snr=np.sqrt(2*simpson(integrand,freq_lgwa))
        if source_type=='averaged':
            integrand=lambda f : 16/5 * waveform(f)**2/lgwa2_psd(f)
        else:
            integrand=lambda f : 8 * waveform(f)**2/lgwa2_psd(f)
    elif detector=='LGWA3':
        if source_type=='averaged':
            integrand =(16/5)*np.array([waveform(f) for f in freq_ce])**2/sensitivity_lgwa_soundcheck
        else:
            integrand =(8)*np.array([waveform(f) for f in freq_ce])**2/sensitivity_lgwa_soundcheck
        snr=np.sqrt(2*simpson(integrand,freq_lgwa))
        if source_type=='averaged':
            integrand=lambda f : 16/5 * waveform(f)**2/lgwa3_psd(f)
        else:
            integrand=lambda f : 8 * waveform(f)**2/lgwa3_psd(f)
    elif detector=='CE1':
        if source_type=='averaged':
            integrand =(16/25)*np.array([waveform(f) for f in freq_ce])**2/sensitivity_ce1**2
        else:
            integrand =(8/5)*np.array([waveform(f) for f in freq_ce])**2/sensitivity_ce1**2
        snr=np.sqrt(2*simpson(integrand,freq_ce))
        integrand=ipd(freq_ce,integrand,fill_value='extrapolate')
    elif detector=='CE2':
        if source_type=='averaged':
            integrand = (16/25)*np.array([waveform(f) for f in freq_ce])**2/sensitivity_ce2**2
        else:
            integrand = (8/5)*np.array([waveform(f) for f in freq_ce])**2/sensitivity_ce2**2
        snr=np.sqrt(2*simpson(integrand,freq_ce))
        integrand=ipd(freq_ce,integrand,fill_value='extrapolate')
    elif detector=='ET':
        if source_type=='averaged':
            integrand = (72/25)*np.array([waveform(f) for f in freq_et])**2/sensitivity_et**2
        else:
            integrand =  (36/5)*np.array([waveform(f) for f in freq_et])**2/sensitivity_et**2
        snr=np.sqrt(simpson(integrand,freq_et))
        integrand=ipd(freq_ce,integrand,fill_value='extrapolate')
    elif detector == 'CE1+CE1+ET':
        if source_type=='averaged':
            integrand =(16/25)*np.array([waveform(f) for f in freq_ce])**2/sensitivity_ce1**2
        else:
            integrand =  (8/5)*np.array([waveform(f) for f in freq_ce])**2/sensitivity_ce1**2
        snr_ce1=2*simpson(integrand,freq_ce)
        if source_type=='averaged':
            integrand =  (12/25)*np.array([waveform(f) for f in freq_et])**2/sensitivity_et**2
        else:
            integrand =  (6/5)*np.array([waveform(f) for f in freq_et])**2/sensitivity_et**2
        snr_et=simpson(integrand,freq_et)
        snr=np.sqrt(snr_ce1+snr_et)
        integrand=ipd(freq_et,integrand,fill_value='extrapolate')
    elif detector == 'CE2+CE2+ET':
        if source_type=='averaged':
            integrand =  (16/25)*np.array([waveform(f) for f in freq_ce])**2/sensitivity_ce2**2
        else:
            integrand =  (8/5)*np.array([waveform(f) for f in freq_ce])**2/sensitivity_ce2**2
        snr_ce2=2*simpson(integrand,freq_ce)
        if source_type=='averaged':
            integrand = (12/25)*np.array([waveform(f) for f in freq_et])**2/sensitivity_et**2
        else:
            integrand = (6/5)*np.array([waveform(f) for f in freq_et])**2/sensitivity_et**2
        snr_et=simpson(integrand,freq_et)
        snr=np.sqrt(snr_ce2+snr_et)
    elif detector == 'LVK':
        if source_type=='averaged':
            integrand =(16/25)*np.array([waveform(f) for f in freq_ligo])**2/sensitivity_ligo**2
        else:
            integrand =(8/5)*np.array([waveform(f) for f in freq_ligo])**2/sensitivity_ligo**2
        snr=np.sqrt(3*simpson(integrand,freq_ligo))
    elif detector == 'LIGO':
        if source_type=='averaged':
            integrand =(16/25)*np.array([waveform(f) for f in freq_ligo])**2/sensitivity_ligo**2
        else:
            integrand =(8/5)*np.array([waveform(f) for f in freq_ligo])**2/sensitivity_ligo**2
        snr=np.sqrt(simpson(integrand,freq_ligo))
    if quantity=='snr':
        return(snr)
    elif quantity=='statistics':
        if detector in ['CE1+CE1+ET','CE2+CE2+ET','CE1+CE1','CE2+CE2','LVK','CE1','CE2','ET','LIGO']:
            if snr<threshold[0]:
                detectibility =0
            else:
                detectibility=1
            return([m1/(1+z),m2/(1+z),z,np.round(snr,decimals=3),detectibility])
        else:
            if isinstance(freq_band,list) or isinstance(freq_band,np.ndarray):
                pass
            else:
                raise ValueError('Frequency band not given')
            if snr<threshold[0]:
                detectibility =0
            elif snr>=threshold[0] and np.any((integrand(freq_band)*freq_band<threshold[1]) & (integrand(freq_band)*freq_band>0)) :
                detectibility=-1
            else:
                detectibility=1
            return([m1/(1+z),m2/(1+z),z,np.round(snr,decimals=3),detectibility])
    else:
        if detector in ['CE1+CE1+ET','CE2+CE2+ET','CE1+CE1','CE2+CE2','LVK','CE1','CE2','ET','LIGO']:
            if snr>threshold[0]:
                detection_factor = np.zeros_like(freq_band)
            else:
                detection_factor = np.ones_like(freq_band)
            return detection_factor
        else:
            if isinstance(freq_band,list) or isinstance(freq_band,np.ndarray):
                pass
            else:
                raise ValueError('Frequency band not given')
            if snr>threshold[0]:
                detection_factor = np.zeros_like(freq_band)
                mask =  np.array([integrand(_) for _ in freq_band])*freq_band < threshold[1] 
                detection_factor[mask] =1
            else:
                detection_factor = np.ones_like(freq_band)
            return detection_factor
def memoize_sim(maxsize=None):
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
@memoize_sim(maxsize=1000)
def normalize(f,m_min,m_max):
    return(quad(lambda m:f(m), m_min, m_max, epsrel=1, epsabs=0)[0])
def waveform_factor(f,z,m1,m2,flag=None):
  m1=m1
  m2=m2
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
  if isinstance(f,np.ndarray) or isinstance(f,list):
      result = np.zeros_like(fr)
      
      mask1 = fr < f_merg
      result[mask1] = fr[mask1]**(-1/3)
      
      mask2 = (fr >= f_merg) & (fr < f_ring)
      result[mask2] = (fr[mask2]**(2/3) / f_merg)
      
      mask3 = (fr >= f_ring) & (fr < f_cut)
      result[mask3] = ((1 / (f_merg * f_ring**(4/3))) * (fr[mask3] / (1 + ((fr[mask3] - f_ring) / (f_w/2))**2))**2)
      return(result)
  else:
      if fr<f_merg:
          return (fr**(-1/3))
      elif fr>=f_merg and fr<f_ring:
          return (fr**(2/3)/f_merg)
      elif fr>=f_ring and fr<f_cut:
          return ((1/(f_merg*f_ring**(4/3)))*(fr/(1+((fr-f_ring)/(f_w/2))**2))**2)
      else:
          return 0

def sampler(merger_rate, pdfm1, pdfm2, Nsamples, Nburn=2500):
    def pdf_m1(m1,z):
        return pdfm1[0](m1,z)

    def pdf_m2(m2,z):
        return pdfm2[0](m2,z)

    def pdf_z(z):
        return merger_rate[0](z)

    def log_probability(params):
        m1, m2, z = params

        if not (m1_min <= m1 <= m1_max) or not (m2_min <= m2 <= m2_max) or not (z_min <= z <= z_max):
            return -np.inf

        if m1 <= 0 or m2 <= 0 or z <= 0:
            return -np.inf

        log_prob_m1 = np.log(pdf_m1(m1,z))
        log_prob_m2 = np.log(pdf_m2(m2,z))
        log_prob_z = np.log(pdf_z(z))

        total_log_prob = log_prob_m1 + log_prob_m2 + log_prob_z

        return total_log_prob

    nwalkers = 9
    ndim = 3
    burnin_steps = Nburn
    nsamples = Nsamples//nwalkers + burnin_steps

    m1_min, m1_max = pdfm1[1], pdfm1[2]
    m2_min, m2_max = pdfm2[1], pdfm2[2]
    z_min, z_max = merger_rate[1], merger_rate[2]

    initial_params = [random.uniform(m1_min, m1_max), random.uniform(
        m2_min, m2_max), random.uniform(z_min, z_max)]  
    initial_pos = [initial_params + 0.1 *
                   np.random.randn(ndim) for _ in range(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)

    sampler.run_mcmc(initial_pos, nsamples, progress=True)

    samples = sampler.get_chain(flat=True, discard=burnin_steps)
    return(samples)
@ray.remote(max_task_retries=-1,max_restarts=-1)
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
    def __init__(self,merger,primary_mass,secondary_mass,R0=lambda m1,m2:1,frequency_band=None,detector=None,verbose=True):
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
        self.verbose=verbose
        self.R0=R0
    def get_background_analytic(self,tol_m=0.01,tol_z=0.01,limit=100,err=1,source_type='averaged'):
        """
        Parameters
        ----------
        tol_m : float, optional
            Quad tolerance of the inner m integration. The default is 0.01.
        tol_z : float, optional
            Quad tolerance of the z integral. The default is 0.01.
        limit : int, optional
            limit of quad integration. The default is 100.
        err : float, optional
            The maximum value of the integrand of match-filtered SNR that will not contribute to background. The default is 1.
        source_type : string, optional
            type of source .Could be either 'averaged' or 'optimal' [face on orientation]. The default is 'averaged'.

        Returns
        -------
        numpy array
            Array containing the background at the given frequency bins.

        """
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
            return(quad(lambda m:f(m), m_min, m_max, epsrel=1, epsabs=0)[0])
        f=self.frequency_band
        rho=(3*(cosmo.H0.value*1000*((mpc)**-1))**2)/(8*np.pi*G)                                                                 #### Changed self.H0 to 100 to get h^2Omega(gw)
        prefactor=(M**(5/3)*((G*np.pi)**(2/3)))/(3*rho*(c**2)*((mpc*1000)**3)*year)
        probability_function_primary  =  lambda m,z : self.primary_mass[0](m,z);m1_min=self.primary_mass[1];m1_max=self.primary_mass[2]
        probability_function_secondary=  lambda m,z : self.secondary_mass[0](m,z);m2_min=self.secondary_mass[1];m2_max=self.secondary_mass[2]           
        z_lower,z_upper=self.merger[1],self.merger[2]        
        def rem_rate(z):                              
            if z<0:
                return 0
            else:
                return(self.merger[0](z))
        global omega
        g_sw=np.empty(0)
        probability_function=lambda m1,m2,z:probability_function_primary(m1,z)*probability_function_secondary(m2,z)
        @ray.remote(max_retries=-1)
        def omega(f):
            def integrand(z,m1,m2):
                Mc=(((m1*m2)**(3/5)) /(m1+m2)**(1/5))
                nm2=normalizer(lambda m:probability_function_secondary(m,z),m2_min,m2_max)
                nm1=normalizer(lambda m:probability_function_primary(m,z),m1_min,m1_max)
                if nm2==0 or nm1==0:
                    return(0)
                else:
                    if self.detector=='Null':
                        return(10**13*prefactor*f*waveform_factor(f,z,m1,m2)*(Mc**(5/3))*(probability_function(m1,m2,z)/(nm2*nm1))*(1/(cosmo.H(z).to('/s').value*(1+z)))*(1*rem_rate(z)))
                    else:
                        return(10**13*prefactor*f*waveform_factor(f,z,m1,m2)*(Mc**(5/3))*detection_factor(m1=m1,m2=m2,z=z,f=f,detector=self.detector,err=err,source_type=source_type)*(probability_function(m1,m2,z)/(nm2*nm1))*(1/(cosmo.H(z).to('/s').value*(1+z)))*(1*rem_rate(z)))
            def m1_integral(m2, z):
                    a=quad(lambda m1: integrand(z=z, m1=m1, m2=m2),m1_min, m1_max, epsrel=tol_m, epsabs=0,limit=limit)[0]                       
                    return(a)
            def m2_integral(z):
                    return quad(lambda m2: m1_integral(m2=m2, z=z),m2_min,m2_max, epsrel=tol_m, epsabs=0,limit=limit)[0]
            result = quad(lambda z: m2_integral(z), z_lower,z_upper, epsrel=tol_z, epsabs=0,limit=limit)
            return result[0]/10**13
        return np.array(ray.get([omega.remote(i) for i in f]))
    
    def get_background_simulated(self,Tobs=5,threshold=(8,1),source_type='averaged',batch_size=5000,nprocs=32,port_id=7705,Nburn=3000):
        """
        Parameters
        ----------
        Tobs : float, optional
            Observation time in years. The default is 5.
        threshold : Tuple, optional
           tuple containing the threshold value of SNR for which source is detectable and the maximum value of the integrand of match-filtered SNR that will not contribute to background . The default is (8,1).
        source_type : string, optional
            type of source .Could be either 'averaged' or 'optimal' [face on orientation]. The default is 'averaged'.
        batch_size : int, optional
            Batches of sources given to Ray .Better to keep this less than 10000 if you want to use Ray dashboard to track the jobs that are running. The default is 5000.
        nprocs : int, optional
            Number of processors to use for computing the backkground. The default is 32.
        port_id :  int, optional
            Port id is a 4 digit number for Ray Dashboard to initialize. The default is 7705.
        Returns
        -------
        list
            list containing the a list of sources that were used to compute the background and array of background in the given frequency bin

        """
        Rz = lambda z : self.merger[0](z)*(cosmo.differential_comoving_volume(z)*4*np.pi*u.sr).to('Gpc^3')/(1+z)
        events = quad(lambda x : Rz(x).value,self.merger[1],self.merger[2],epsabs=0,epsrel=0.1,limit=100)[0]
        Number_of_events = int(events*Tobs)
        norm_z = quad(lambda x: Rz(x).value, self.merger[1],self.merger[2],epsabs=0,epsrel=0.1,limit=200)[0]
        def p(z): return Rz(z).value/norm_z
        samples = sampler(merger_rate=(lambda z: p(z), self.merger[1],self.merger[2]), pdfm1=(lambda m,z:self.primary_mass[0](m,z)/normalize(lambda m :self.primary_mass[0](m,z),self.primary_mass[1],self.primary_mass[2]), self.primary_mass[1],self.primary_mass[2]),
                          pdfm2=(lambda m,z:self.secondary_mass[0](m,z)/normalize(lambda m :self.secondary_mass[0](m,z),self.secondary_mass[1],self.secondary_mass[2]), self.secondary_mass[1],self.secondary_mass[2]), Nsamples=Number_of_events, Nburn=Nburn)
        @ray.remote(max_retries=-1)
        def flux(samples,Tobs,freq,detector=None,threshold=(8,1),source_type=source_type):
            if detector not in ['LISA','LGWA1','LGWA2','LGWA3','CE1+CE1+ET','CE2+CE2+ET','CE1+CE1','CE2+CE2','Null','LIGO','LVK','CE1','CE2','ET']:
                raise ValueError('Invalid Detector')
            f=freq
            m1,m2,z=samples[0],samples[1],samples[2]
            rho=(3*(cosmo.H0.value*1000*((mpc)**-1))**2)/(8*np.pi*G) 
            dl=cosmo.luminosity_distance(z).to('m').value
            if detector=='Null':
                ans=(1+z)**2/(12*np.pi*c*dl**2 *rho * c**2) * (G*np.pi)**(2/3) * f  * (m1*m2/(m1+m2)**(1/3))*M**(5/3) *waveform_factor(f=f, z=z, m1=m1, m2=m2)
            else:
                ans=(1+z)**2/(12*np.pi*c*dl**2 *rho * c**2) * (G*np.pi)**(2/3) * f  * (m1*m2/(m1+m2)**(1/3))*M**(5/3)*match_filtered_snr(m1=m1, m2=m2, z=z, freq_band=f,threshold=threshold,detector=detector,quantity='detection',source_type=source_type)  *waveform_factor(f=f, z=z, m1=m1, m2=m2)        
            return ans/(Tobs*year)
        def background(samples,freq=None,detector=None,Tobs=Tobs,threshold=threshold,batch_size=batch_size,port_id=port_id,source_type=source_type):
            if detector not in ['LISA','LGWA1','LGWA2','LGWA3','CE1+CE1+ET','CE2+CE2+ET','CE1+CE1','CE2+CE2','Null','LIGO','LVK','CE1','CE2','ET']:
                raise ValueError('Invalid Detector')
            num_batches = (len(samples) + batch_size - 1) // batch_size
            batches = [samples[i:i+batch_size] for i in range(0, len(samples), batch_size)]
            om=[]
            for i in tqdm(iterable=range(len(batches)),desc='Processing',total=len(batches)):
                omega = ray.get([flux.remote(_,detector=detector,Tobs=Tobs,freq=freq,threshold=threshold,source_type=source_type) for _ in batches[i]])
                result_array = [sum(column) for column in zip(*omega)]
                om.append(result_array)
            final_result = [sum(column) for column in zip(*om)]
            return(ipd(freq,final_result,fill_value='extrapolate'))
        a1=background(samples,freq=self.frequency_band,detector=self.detector,Tobs=Tobs,threshold=threshold,batch_size=batch_size)
        return([samples,np.array(a1(self.frequency_band))])

