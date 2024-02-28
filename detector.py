import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo
from scipy.integrate import quad,IntegrationWarning
from scipy.interpolate import interp1d as ipd
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from astropy.cosmology import z_at_value 
from scipy.optimize import minimize_scalar
import emcee
import random
import pandas as pd
from source_parm import mass_distribution
import ray
from time import time
import functools
import warnings
from scipy.ndimage import median_filter
from source_parm import merger_rate
window_size=500
warnings.filterwarnings("ignore",category=RuntimeWarning)
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
freq_lisa = np.loadtxt('LISA_fit.txt',usecols=0)
sensitivity_lisa = np.loadtxt('LISA_fit.txt',usecols=1)
lisa_psd = ipd(freq_lisa,sensitivity_lisa,fill_value='extrapolate',kind='cubic')
freq_ce=np.loadtxt('CE.txt',delimiter=',',usecols=0)
sensitivity_ce1=np.loadtxt('CE.txt',delimiter=',',usecols=1)
sensitivity_ce2=np.loadtxt('CE.txt',delimiter=',',usecols=2)
ce1_psd = ipd(freq_ce,sensitivity_ce1,fill_value='extrapolate',kind='cubic')
ce2_psd = ipd(freq_ce,sensitivity_ce2,fill_value='extrapolate',kind='cubic')




freq_et=np.loadtxt('ET.txt',delimiter=',',usecols=0)
sensitivity_et=np.loadtxt('ET.txt',delimiter=',',usecols=1)
et_psd = ipd(freq_et,sensitivity_et,fill_value='extrapolate',kind='cubic')



# 9 LGWA (Nb)
freq_lgwa=np.loadtxt('LGWA_Soundcheck_psd.txt',usecols=0,delimiter=',')

sensitivity_lgwa_nb = np.sqrt(np.loadtxt('LGWA_Nb_psd.txt',usecols=1,delimiter=','))
lgwa1_psd = ipd(freq_lgwa,sensitivity_lgwa_nb,fill_value='extrapolate',kind='cubic')

# 9 LGWA (Si)

sensitivity_lgwa_si = np.sqrt(np.loadtxt('LGWA_Si_psd.txt',usecols=1,delimiter=','))

lgwa2_psd = ipd(freq_lgwa,sensitivity_lgwa_si,fill_value='extrapolate',kind='cubic')

# 10 LGWA (Soundcheck)
sensitivity_lgwa_soundcheck =np.sqrt( np.loadtxt('LGWA_Soundcheck_psd.txt',usecols=1,delimiter=','))
lgwa3_psd = ipd(freq_lgwa,sensitivity_lgwa_soundcheck,fill_value='extrapolate',kind='cubic')




freq_ligo= np.loadtxt('LIGO.txt',usecols=0,delimiter=',')
sensitivity_ligo = np.loadtxt('LIGO.txt',usecols=1,delimiter=',')
ligo_psd = ipd(freq_ligo,sensitivity_ligo,fill_value='extrapolate',kind='cubic')
###Overlap reduction function L1=LIGO Livingston H1=Ligo Hanford V1=Virgo
orf_hl=ipd(np.loadtxt('ORF.txt',usecols=0,delimiter=','),np.loadtxt('ORF.txt',usecols=1,delimiter=','),fill_value='extrapolate',kind='cubic')
orf_hv=ipd(np.loadtxt('ORF.txt',usecols=0,delimiter=','),np.loadtxt('ORF.txt',usecols=2,delimiter=','),fill_value='extrapolate',kind='cubic')
orf_lv=ipd(np.loadtxt('ORF.txt',usecols=0,delimiter=','),np.loadtxt('ORF.txt',usecols=3,delimiter=','),fill_value='extrapolate',kind='cubic')
orf_ce_et_1=ipd(np.loadtxt('ORF.txt',usecols=0,delimiter=','),(-2/np.sqrt(3))*np.loadtxt('ORF.txt',usecols=3,delimiter=','),fill_value='extrapolate',kind='cubic')
orf_ce_et_2=ipd(np.loadtxt('ORF.txt',usecols=0,delimiter=','),(-2/np.sqrt(3))*np.loadtxt('ORF.txt',usecols=2,delimiter=','),fill_value='extrapolate',kind='cubic')
#### Function to calculate Match-filetered SNR ###
def match_filtered_snr(m1,m2,z,detector=None,threshold=(8,0.01),quantity=None,source_type='averaged',freq_band=None):
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
        Must be any of the detectors from ['LISA','LGWA1','LGWA2','LGWA3','CE1+CE1+ET','CE2+CE2+ET','CE1+CE1','CE2+CE2','LVK']
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
    list of the form [m1,m2,z,snr, 0 or 1] if quantity='statistics' 
    """
    if detector not in ['LISA','LGWA1','LGWA2','LGWA3','CE1+CE1+ET','CE2+CE2+ET','CE1+CE1','CE2+CE2','LVK','Null','CE1','CE2','ET','LIGO']:
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
    elif detector=='LGWA2':
        if source_type=='averaged':
            integrand =(16/5)*np.array([waveform(f) for f in freq_lgwa])**2/sensitivity_lgwa_si
        else:
            integrand =(8)*np.array([waveform(f) for f in freq_lgwa])**2/sensitivity_lgwa_si
        snr=np.sqrt(2*simpson(integrand,freq_lgwa))
    elif detector=='LGWA3':
        if source_type=='averaged':
            integrand =(16/5)*np.array([waveform(f) for f in freq_ce])**2/sensitivity_lgwa_soundcheck
        else:
            integrand =(8)*np.array([waveform(f) for f in freq_ce])**2/sensitivity_lgwa_soundcheck
        snr=np.sqrt(2*simpson(integrand,freq_lgwa))
    elif detector=='CE1':
        if source_type=='averaged':
            integrand =(16/25)*np.array([waveform(f) for f in freq_ce])**2/sensitivity_ce1**2
        else:
            integrand =(8/5)*np.array([waveform(f) for f in freq_ce])**2/sensitivity_ce1**2
        snr=np.sqrt(2*simpson(integrand,freq_ce))
    elif detector=='CE2':
        if source_type=='averaged':
            integrand = (16/25)*np.array([waveform(f) for f in freq_ce])**2/sensitivity_ce2**2
        else:
            integrand = (8/5)*np.array([waveform(f) for f in freq_ce])**2/sensitivity_ce2**2
        snr=np.sqrt(2*simpson(integrand,freq_ce))
    elif detector=='ET':
        if source_type=='averaged':
            integrand = (72/25)*np.array([waveform(f) for f in freq_et])**2/sensitivity_et**2
        else:
            integrand =  (36/5)*np.array([waveform(f) for f in freq_et])**2/sensitivity_et**2
        snr=np.sqrt(simpson(integrand,freq_et))
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
        if detector in ['CE1+CE1+ET','CE2+CE2+ET','LVK','CE1','CE2','ET','LIGO']:
            if snr<threshold[0]:
                detectibility =0
            else:
                detectibility=1
            return([m1/(1+z),m2/(1+z),z,np.round(snr,decimals=3),detectibility])
        else:
            if freq_band==None:
                raise ValueError('Frequency band not given')
            if snr<threshold[0]:
                detectibility =0
            elif snr>=threshold[0] and np.any((integrand(freq_band)<threshold[1]) & (integrand(freq_band)>0)) :
                detectibility=-1
            else:
                detectibility=1
            return([m1/(1+z),m2/(1+z),z,np.round(snr,decimals=3),detectibility])
    else:
        if detector in ['CE1+CE1+ET','CE2+CE2+ET','LVK','CE1','CE2','ET','LIGO']:
            if snr>threshold[0]:
                detection_factor = np.zeros_like(freq_band)
            else:
                detection_factor = np.ones_like(freq_band)
            return detection_factor
        else:
            if freq_band==None:
                raise ValueError('Frequency band not given')
            if snr>threshold[0]:
                detection_factor = np.zeros_like(freq_band)
                mask =  integrand(freq_band) < threshold[1]
                detection_factor[mask] =1
            else:
                detection_factor = np.ones_like(freq_band)
            return detection_factor
@ray.remote(max_retries=-1)
def optimal_snr(m1,m2,detector=None,optimal=8,source_type='optimal'):
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
        Must be any of the detectors from ['LISA','LGWA1','LGWA2','LGWA3','CE1+CE1+ET','CE2+CE2+ET','CE1+CE1','CE2+CE2','LVK']. The default is None.
    optimal : float, optional
        Threshold value of the match-filtered SNR. The default is 8.
    source_type : string, optional
        type of source .Could be either 'averaged' or 'optimal' [face on orientation]. The default is 'averaged'.

    Returns
    -------
    list
        DESCRIPTION.

    """
    if source_type not in ['averaged','optimal']:
        raise ValueError('source_type has to be either averaged or optimal')
    cc=[np.exp(-i) for i in np.arange(0,5,1)]
    res=0
    count=0
    for i in range(len(cc)):
        temp=res
        ans=minimize_scalar(lambda x:abs(match_filtered_snr(m1=m1,m2=m2,z=x,detector=detector,quantity='snr',source_type=source_type)-optimal)+cc[i]*(x-res)**2)
        res=ans.x
        if abs(match_filtered_snr(m1=m1,m2=m2,z=res,detector=detector,quantity='snr',source_type=source_type)-optimal)<0.01:
                return [m1,m2,res]
    init=np.round(0.01,decimals=4)
    check=match_filtered_snr(m1=m1,m2=m2,z=init,detector=detector,quantity='snr',source_type=source_type)
    while check>=optimal:
        if match_filtered_snr(m1=m1,m2=m2,z=init+5,detector=detector,quantity='snr',source_type=source_type)>optimal:
            init=np.round(init+5,decimals=4)
        else:
            init=np.round(init+0.01,decimals=4)
        check=match_filtered_snr(m1=m1,m2=m2,z=init,detector=detector,quantity='snr',source_type=source_type)
    return([m1,m2,np.round(init-0.01,decimals=4)])

class detector():
    """
    A class to define detector object which has a method get_asd() which returns a callable function that represents
    the square root of the PSD of that detector (Power Spectral Density (1/Hz))
    Attributes:
              detector (string) : Name of detector (can be 'LISA','LGWA1','LGWA2','LGWA3','CE1','CE2','ET','ALIGO')
    """
    def __init__(self,detector=None):
        self.detector=detector
    def get_asd(self):
        if self.detector=="LGWA1" or self.detector=="LGWA_Ni" or self.detector=="LGWA_ni":
            return(lgwa1_psd)
        elif self.detector=="LGWA2" or self.detector=="LGWA_Sb" or self.detector=="LGWA_sb":
            return(lgwa2_psd)
        elif self.detector=="LGWA3" or self.detector=="LGWA_Soundcheck" or self.detector=="LGWA_soundcheck":
            return(lgwa3_psd)
        elif self.detector=="ET" or self.detector=="EINSTEIN TELESCOPE":
            return(et_psd)
        elif self.detector=="CE1" or self.detector=="COSMIC EXPLORER1":
            return(ce1_psd)
        elif self.detector=="CE2" or self.detector=="COSMIC EXPLORER2":
            return(ce2_psd)
        elif self.detector=="LISA":
            return(lisa_psd)
        elif self.detector=='ALIGO' or self.detector=='LIGO' or self.detector=='APLUS':
            return(ligo_psd)
def detector_network(conf=None):
    """
    Parameters
    ----------
    conf : String, optional
        Must be any of the detectors from ['LISA','LGWA1','LGWA2','LGWA3','CE1+CE1+ET','CE2+CE2+ET','CE1+CE1','CE2+CE2','LVK']. The default is None.
    Returns
    -------
    Callable fucntion.
    """
    if conf == 'CE1+CE1+ET':
        effective_strain_noise = lambda f : np.sqrt((orf_ce_et_1(f)**2/(ce1_psd(f)**2*et_psd(f)**2)+orf_ce_et_2(f)**2/(ce1_psd(f)**2*et_psd(f)**2)+orf_hl(f)**2/(ce1_psd(f)**2*ce1_psd(f)**2))**-1)
        noise = lambda f : (np.sqrt(50) *np.pi**2 *f**3 /(3*cosmo.H0.to_value('/s')**2)) * effective_strain_noise(f) 
        return(noise)
    elif conf == 'CE2+CE2+ET':
        effective_strain_noise = lambda f : np.sqrt((orf_ce_et_1(f)**2/(ce2_psd(f)**2*et_psd(f)**2)+orf_ce_et_2(f)**2/(ce2_psd(f)**2*et_psd(f)**2)+orf_hl(f)**2/(ce2_psd(f)**2*ce2_psd(f)**2))**-1)
        noise = lambda f : (np.sqrt(50) *np.pi**2 *f**3 /(3*cosmo.H0.to_value('/s')**2)) * effective_strain_noise(f) 
        return(noise)
    elif conf == 'CE1+CE1':
        effective_strain_noise = lambda f : np.sqrt((orf_hl(f)**2/(ce1_psd(f)**2*ce1_psd(f)**2))**-1)
        noise = lambda f : (np.sqrt(50) *np.pi**2 *f**3 /(3*cosmo.H0.to_value('/s')**2)) * effective_strain_noise(f)
        return(noise)
    elif conf == 'CE2+CE2':
        effective_strain_noise = lambda f : np.sqrt((orf_hl(f)**2/(ce2_psd(f)**2*ce2_psd(f)**2))**-1)
        noise = lambda f : (np.sqrt(50) *np.pi**2 *f**3 /(3*cosmo.H0.to_value('/s')**2)) * effective_strain_noise(f)
        return(noise)
    elif conf == 'LISA':
        noise = lambda f : 4*np.pi**2/(3*cosmo.H0.to_value('/s')**2 *(10000/cosmo.H0.value**2)) * f**3 * lisa_psd(f)**2
        return(noise)
    elif conf == 'LGWA1' or conf=='LGWA-Nb':
        effective_strain_noise = lambda f : lgwa1_psd(f)**2
        noise = lambda f : (10 *np.pi**2 *f**3 /(3*cosmo.H0.to_value('/s')**2)) * effective_strain_noise(f)
        return(noise)
    elif conf == 'LGWA2' or conf=='LGWA-Si':
        effective_strain_noise = lambda f : lgwa2_psd(f)**2
        noise = lambda f : (10 *np.pi**2 *f**3 /(3*cosmo.H0.to_value('/s')**2)) * effective_strain_noise(f)
        return(noise)
    elif conf == 'LGWA3' or conf=='LGWA-Soundcheck':
        effective_strain_noise = lambda f : lgwa3_psd(f)**2
        noise = lambda f : (10 *np.pi**2 *f**3 /(3*cosmo.H0.to_value('/s')**2)) * effective_strain_noise(f)
        return(noise)
    elif conf == 'LIGO':
        effective_strain_noise = lambda f : np.sqrt((orf_hl(f)**2/(ligo_psd(f)**2*ligo_psd(f)**2))**-1)
        noise = lambda f : (np.sqrt(50) *np.pi**2 *f**3 /(3*cosmo.H0.to_value('/s')**2)) * effective_strain_noise(f)
        return(noise)
    elif conf == 'LVK':
        effective_strain_noise = lambda f : np.sqrt((orf_lv(f)**2/(ligo_psd(f)**2*ligo_psd(f)**2)+orf_hv(f)**2/(ligo_psd(f)**2*ligo_psd(f)**2)+orf_hl(f)**2/(ligo_psd(f)**2*ligo_psd(f)**2))**-1)
        noise = lambda f : (np.sqrt(50) *np.pi**2 *f**3 /(3*cosmo.H0.to_value('/s')**2)) * effective_strain_noise(f) 
        return(noise)
def pls(fref=0.001,detector=None,Tobs=5,beta_list=np.arange(-8,9,1),snr=1):
    """
    

    Parameters
    ----------
    fref : float, optional
        Reference frequency for PLS. The default is 0.001.
    detector : string, optional
        Must be any of the detectors from ['LISA','LGWA1','LGWA2','LGWA3','CE1+CE1+ET','CE2+CE2+ET','CE1+CE1','CE2+CE2','LVK']. The default is None.
    Tobs : float, optional
        Observation time in years. The default is 5.
    beta_list : numpy array, optional
        array of power law index for the PLS. The default is np.arange(-8,9,1).
    snr : float, optional
        SNR for the PLS. The default is 1.

    Returns
    -------
    Callable function

    """
    if detector == 'LISA':
        freq=np.arange(10**-5,1+10**-5,10**-6)
    elif detector == 'LGWA1' or detector == 'LGWA-Ni':
        freq = np.arange(10**-3,3+10**-5,10**-5)
    elif detector == 'LGWA2' or detector == 'LGWA-Sb' or detector == 'LGWA3':
        freq = np.arange(10**-3,3+10**-5,10**-5)     
    elif detector in ['CE1+CE1+ET', 'CE2+CE2+ET', 'CE1+CE1', 'CE2+CE2', 'LIGO', 'LVK']:
        freq =np.arange(1,1000.25,0.25)
    else:
        pass
    noise = lambda f : detector_network(conf=detector)(f)
    if detector=='LISA':
        ans = [snr*(np.sqrt(Tobs*year*quad(lambda f :( (f/fref)**(2*i))/noise(f)**2,min(freq),max(freq),epsabs=0,epsrel=1,limit=1000)[0])**-1) for i in beta_list]
        power_law=[]
        for i in range(len(freq)):
            temp = np.array([ans[j]*(freq[i]/fref)**(beta_list[j]) for j in range(len(beta_list))])
            power_law.append(max(temp))
        return(ipd(freq,power_law,fill_value='extrapolate'))
    else:
        ans = [snr*(np.sqrt(2*Tobs*year*quad(lambda f :( (f/fref)**(2*i))/noise(f)**2,min(freq),max(freq),epsabs=0,epsrel=1,limit=1000)[0])**-1) for i in beta_list]
        power_law=[]
        for i in range(len(freq)):
            temp = np.array([ans[j]*(freq[i]/fref)**(beta_list[j]) for j in range(len(beta_list))])
            power_law.append(max(temp))
        return(ipd(freq,power_law,fill_value='extrapolate'))
          
def get_background_snr(gw_background=None,band=None,detector=None,sampling_rate=1/4096,Tobs=5):
        """
        Parameters
        ----------
        gw_background : callable function, optional
            Callable function for the background. The default is None.
        band : numpy array, optional
            Array of frequency bin where background has been calculated. The default is None.
        detector : string, optional
            Must be any of the detectors from ['LISA','LGWA1','LGWA2','LGWA3','CE1+CE1+ET','CE2+CE2+ET','CE1+CE1','CE2+CE2','LVK']. The default is None.
        sampling_rate : float, optional
            Sampling rate at which frequency bins will be generated to calculate SNR. The default is 1/4096.
        Tobs : float, optional
            Observation time in years. The default is 5.
        Returns
        -------
        snr : float     
        """
        if sampling_rate==None:
            raise ValueError('Sampling Rate not specified')
        if gw_background==None :
            raise ValueError("Input background not specified")
        if band.any()==None:
            raise ValueError('Invalid Frequency band')
        noise=detector_network(conf=detector)
        if detector=='LISA':
            freq=np.arange(min(band),max(band)+sampling_rate,sampling_rate) 
            snr=np.sqrt(Tobs*year*np.sum(np.array(gw_background(freq)**2/noise(freq)**2))*sampling_rate)
        else:
            freq=np.arange(min(band),max(band)+sampling_rate,sampling_rate) 
            snr=np.sqrt(2*Tobs*year*np.sum(np.array(gw_background(freq)**2/noise(freq)**2))*sampling_rate)         
        return(snr)


        
        
        
        
        