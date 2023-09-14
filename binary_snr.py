import numpy as np
import cosmology as cs
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import warnings
import random
from scipy.integrate import quad
from scipy import interpolate
import scipy.stats as distribution
warnings.filterwarnings("ignore",category=RuntimeWarning)
import csv
mpc=3.08568*10**22
year=365*24*3600
c=299792458
M=2*10**30
G=6.67*10**-11
density_detector = []

# We consider noise characteristic strain (1/sqrt(Hz)) of various detectors in the given Frequency band
# Note that we have mentioned the source (website or arxiv paper)
# There are 3 parts : a)Noise strain b)Frequency bins c)omega_strain calculated using the formula
# omega_noise = ((5*sqrt(2)*(pi**2))/(3*H**2))*(f**3)*S(f) where S(f)=(loaded data)**2
# This is because loaded data gives you sqrt(S)
# 1 A+ LVK                  https://dcc.ligo.org/LIGO-T1800042/public
with open('a_plus_noise.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))
freq_a_plus = np.empty(0)
sensitivity_a_plus = np.empty(0)
for i in range(len(data)):
    freq_a_plus = np.append(freq_a_plus, float(data[i][0]))
    sensitivity_a_plus = np.append(sensitivity_a_plus, float(data[i][1]))
#omega_a_plus = ((5*np.sqrt(2)*(np.pi**2))/(3*H**2)) * \
   # (freq_a_plus**3)*sensitivity_a_plus**2
# 2 DECIGO arXiv:2202.04253v1
with open('decigo_noise.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))
freq_decigo = np.empty(0)
sensitivity_decigo = np.empty(0)
for i in range(len(data)):
    freq_decigo = np.append(freq_decigo, float(data[i][0]))
    sensitivity_decigo = np.append(sensitivity_decigo, float(data[i][1]))
# 3 LISA   arXiv:1803.01944v2
with open('lisa_noise.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))
freq_lisa = np.empty(0)
sensitivity_lisa = np.empty(0)
for i in range(len(data)):
    freq_lisa = np.append(freq_lisa, float(data[i][0]))
    sensitivity_lisa = np.append(sensitivity_lisa, float(data[i][1]))
# 4 CE_A (20 km Baseline)
with open('cosmic_explorer_A_noise.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))
freq_ce_A = np.empty(0)
sensitivity_ce_A = np.empty(0)
for i in range(len(data)):
    freq_ce_A = np.append(freq_ce_A, float(data[i][0]))
    sensitivity_ce_A = np.append(sensitivity_ce_A, float(data[i][1]))
# 4 CE_B (40 km baseline)
with open('cosmic_explorer_B_noise.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))
freq_ce_B = np.empty(0)
sensitivity_ce_B = np.empty(0)
for i in range(len(data)):
    freq_ce_B = np.append(freq_ce_B, float(data[i][0]))
    sensitivity_ce_B = np.append(sensitivity_ce_B, float(data[i][1]))
# 6 ET
with open('ET_noise.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))
freq_et = np.empty(0)
sensitivity_et = np.empty(0)
for i in range(len(data)):
    freq_et = np.append(freq_et, float(data[i][0]))
    sensitivity_et = np.append(sensitivity_et, float(data[i][3]))
# 7 Test 
freq_test=np.arange(10**-4,0.1,10**-5)
sensitivity_test=(10**-20)*np.ones(len(freq_test))

###Overlap reduction function L1=LIGO Livingston H1=Ligo Hanford V1=Virgo
f_hl=np.loadtxt('orf_h1_l1.txt',usecols=0)
orf_hl=np.loadtxt('orf_h1_l1.txt',usecols=1)
f_hv=np.loadtxt('orf_h1_v1.txt',usecols=0)
orf_hv=np.loadtxt('orf_h1_v1.txt',usecols=1)
f_lv=np.loadtxt('orf_l1_v1.txt',usecols=0)
orf_lv=np.loadtxt('orf_l1_v1.txt',usecols=1)
def phenomA(f,m1,m2,z,flag=None):
    fm = [2.9740*10**(-1), 4.4810*10**(-2), 9.5560*10**(-2)]
    fri = [5.9411*10**(-1), 8.9794*10**(-2), 19.111*10**(-2)]
    fc = [8.4845*10**(-1), 12.848*10**(-2), 27.299*10**(-2)]
    fw = [5.0801*10**(-1), 7.7515*10**(-2), 2.2369*10**(-2)]
    M_total = m1+m2
    eta = m1*m2/(M_total**2)
    M_chirp=(eta**(3/5))*M_total
    M_x=(G*M_chirp*M)/c**3
    f_merg = ((c**3)*(fm[0]*eta*eta+fm[1]*eta+fm[2]))/(np.pi*G*M_total*M)
    f_ring = ((c**3)*(fri[0]*eta*eta+fri[1]*eta+fri[2]))/(np.pi*G*M_total*M)
    f_cut = ((c**3)*(fc[0]*eta*eta+fc[1]*eta+fc[2]))/(np.pi*G*M_total*M)
    f_w = ((c**3)*(fw[0]*eta*eta+fw[1]*eta+fw[2]))/(np.pi*G*M_total*M)
    w=(np.pi*f_w/2)*(f_merg/f_ring)**(2/3)
    ans = []
    if flag=='cutoff':
        return(f_cut)
    factor=((np.sqrt(5/24))*(M_x**(5/6))*(f_merg**(-7/6)))/((np.pi**(2/3))*(1/c))
    for _ in f:
        if _ < f_merg:
            ans.append((_/f_merg)**(-7/6))
        elif _ >= f_merg and _ < f_ring:
            ans.append((_/f_merg)**(-2/3))
        elif _ >= f_ring and _ < f_cut:
            ans.append((w/(2*np.pi))*(f_w/((_-f_ring)**2+((f_w**2)/4))))
        else:
            ans.append(0)
    return(np.array(ans)*factor)
class detector():
    def __init__(self,detector=None):
        self.detector=detector
    def get_asd(self):
        if self.detector=="LIGO" or self.detector=="VIRGO" or self.detector=="KAGRA":
            return(freq_a_plus,sensitivity_a_plus)
        elif self.detector=="ET" or self.detector=="EINSTEIN TELESCOPE":
            return(freq_et,sensitivity_et)
        elif self.detector=="CE1" or self.detector=="COSMIC EXPLORER1":
            return(freq_ce_A,sensitivity_ce_A)
        elif self.detector=="CE2" or self.detector=="COSMIC EXPLORER2":
            return(freq_ce_B,sensitivity_ce_B)
        elif self.detector=="DECIGO":
            return(freq_decigo,sensitivity_decigo)
        elif self.detector=="LISA":
            return(freq_lisa,sensitivity_lisa)
        else:
            return(freq_test,sensitivity_test)
def detector_network(conf=None,H0=70):
    if conf not in [1,2,3,4,5,6,7]:
        raise ValueError('conf has to be a integer between 1 and 7')
    if conf==1: ###CE1+CE1###
       freq=np.arange(5,100.25,0.25)
       p1=interpolate.interp1d(detector('CE1').get_asd()[0],detector('CE1').get_asd()[1],kind='linear')(freq)**2
       H=H0*1000/mpc
       orf=interpolate.interp1d(x=f_hl, y=orf_hl,kind='cubic')(freq)
       fac=(10*(np.pi**2)*freq**3)/(3*(H**2)*orf)
       Cov=(p1)*(p1)*fac**2
       return([freq,Cov])
    elif conf==2: ###CE2+CE2###
       freq=np.arange(5,100.25,0.25)
       p1=interpolate.interp1d(detector('CE2').get_asd()[0],detector('CE2').get_asd()[1],kind='linear')(freq)**2
       H=H0*1000/mpc
       orf=interpolate.interp1d(x=f_hl, y=orf_hl,kind='cubic')(freq)
       fac=(10*(np.pi**2)*freq**3)/(3*(H**2)*orf)
       Cov=(p1)*(p1)*fac**2
       return([freq,Cov])
    elif conf==3: ###CE1+CE2###
       freq=np.arange(5,100.25,0.25)
       p1=interpolate.interp1d(detector('CE1').get_asd()[0],detector('CE1').get_asd()[1],kind='linear')(freq)**2
       p2=interpolate.interp1d(detector('CE2').get_asd()[0],detector('CE2').get_asd()[1],kind='linear')(freq)**2
       H=H0*1000/mpc
       orf=interpolate.interp1d(x=f_hl, y=orf_hl,kind='cubic')(freq)
       fac=(10*(np.pi**2)*freq**3)/(3*(H**2)*orf)
       Cov=(p1)*(p2)*fac**2
       return([freq,Cov])
    elif conf==4: ###CE1+CE1+ET###
       freq=np.arange(5,100.25,0.25)
       p1=interpolate.interp1d(detector('CE1').get_asd()[0],detector('CE1').get_asd()[1],kind='linear')(freq)**2
       p2=interpolate.interp1d(detector('CE2').get_asd()[0],detector('CE2').get_asd()[1],kind='linear')(freq)**2
       p3=interpolate.interp1d(detector('ET').get_asd()[0],detector('ET').get_asd()[1],kind='linear')(freq)**2
       H=H0*1000/mpc
       orf12=interpolate.interp1d(x=f_hl, y=orf_hl,kind='cubic')(freq)
       orf13=interpolate.interp1d(x=f_hv, y=orf_hv,kind='cubic')(freq)
       orf23=interpolate.interp1d(x=f_lv, y=orf_lv,kind='cubic')(freq)
       fac=(10*(np.pi**2)*freq**3)/(3*(H**2))
       cov_eff=p1*p1/orf12**2+p1*p3*((np.sin(np.pi/3))**2)/orf13**2+p1*p3*((np.sin(np.pi/3))**2)/orf23**2
       Cov=cov_eff*fac**2
       return([freq,Cov])
    elif conf==5: ###CE2+CE2+ET###
       freq=np.arange(5,100.25,0.25)
       p1=interpolate.interp1d(detector('CE1').get_asd()[0],detector('CE1').get_asd()[1],kind='linear')(freq)**2
       p2=interpolate.interp1d(detector('CE2').get_asd()[0],detector('CE2').get_asd()[1],kind='linear')(freq)**2
       p3=interpolate.interp1d(detector('ET').get_asd()[0],detector('ET').get_asd()[1],kind='linear')(freq)**2
       H=H0*1000/mpc
       orf12=interpolate.interp1d(x=f_hl, y=orf_hl,kind='cubic')(freq)
       orf13=interpolate.interp1d(x=f_hv, y=orf_hv,kind='cubic')(freq)
       orf23=interpolate.interp1d(x=f_lv, y=orf_lv,kind='cubic')(freq)
       fac=(10*(np.pi**2)*freq**3)/(3*(H**2))
       cov_eff=p2*p2/orf12**2+p2*p3*((np.sin(np.pi/3))**2)/orf13**2+p2*p3*((np.sin(np.pi/3))**2)/orf23**2
       Cov=cov_eff*fac**2
       return([freq,Cov])
    elif conf==6: ###CE1+CE2+ET###
       freq=np.arange(5,100.25,0.25)
       p1=interpolate.interp1d(detector('CE1').get_asd()[0],detector('CE1').get_asd()[1],kind='linear')(freq)**2
       p2=interpolate.interp1d(detector('CE2').get_asd()[0],detector('CE2').get_asd()[1],kind='linear')(freq)**2
       p3=interpolate.interp1d(detector('ET').get_asd()[0],detector('ET').get_asd()[1],kind='linear')(freq)**2
       H=H0*1000/mpc
       orf12=interpolate.interp1d(x=f_hl, y=orf_hl,kind='cubic')(freq)
       orf13=interpolate.interp1d(x=f_hv, y=orf_hv,kind='cubic')(freq)
       orf23=interpolate.interp1d(x=f_lv, y=orf_lv,kind='cubic')(freq)
       fac=(10*(np.pi**2)*freq**3)/(3*(H**2))
       cov_eff=p1*p2/orf12**2+p1*p3*((np.sin(np.pi/3))**2)/orf13**2+p2*p3*((np.sin(np.pi/3))**2)/orf23**2
       Cov=cov_eff*fac**2
       return([freq,Cov])
    elif conf==7 :###LISA###
       H=H0*1000/mpc
       f=detector('LISA').get_asd()[0]
       freq=np.round(np.arange(10**-4,0.1,0.00025),decimals=5)
       sh=interpolate.interp1d(f,detector('LISA').get_asd()[1]**2)(freq)
       fac=(2*(np.pi**2)*freq**3)/(3*(H**2))
       Cov=(fac*sh)**2
       return([freq,Cov])         
b=[]
class bbh():
    def __init__(self,m1=30,m2=30,z=1,H0=70,om_m=0.3,om_l=0.7,detector=None):
        self.m1=m1*(1+z)
        self.m2=m2*(1+z)
        self.z=z
        self.H0=H0
        self.om_m=om_m
        self.om_l=om_l
        self.detector=detector
    def get_strain(self):
        A=phenomA(detector(self.detector).get_asd()[0],self.m1, self.m2,self.z)
        A=A/(cs.cosmology(self.H0,self.om_m,self.om_l).get_lum_dist(self.z))
        return(A)
    def get_snr(self,Tobs=1):
        C=self.get_strain()
        Sh=(detector(self.detector).get_asd()[1])**2
        freq=detector(self.detector).get_asd()[0]
        R=1/5
        if self.detector=='test':
            R=1
        if self.detector=='LISA':
            R=1
            M_total = self.m1+self.m2
            eta = self.m1*self.m2/(M_total**2)
            M_chirp=(eta**(3/5))*M_total
            if M_total<=0:
                return(0)
            M_x=(G*M_chirp*M)/c**3
            f_isco=(c**3)/(G*(6**(3/2))*M_total*M*np.pi)
            tau=(eta*Tobs*year*(c**3))/(G*5*M_total*M)
            f_initial=((c**3)/(8*np.pi*G*M_total*M*(tau**(3/8))))*(1+((11*eta/32)+743/2688)*tau**(-1/4))
            f_initial=min(max(f_initial,10**-4),0.1)
            if f_isco>0.1:
                f_end=0.1
            else:
                f_end=f_isco
            if f_initial==0.1:
                return 0
            init=np.argwhere(freq>=f_initial)[0][0]
            end=np.argwhere(freq>=f_end)[0][0]
            freq=freq[init:end]
            Sh=Sh[init:end]
            C=C[init:end]
        if self.detector=='ET' :
            R=9/20
        if self.detector=='DECIGO':
            R=9/20
        dx_values = np.diff(freq)
        return(np.sqrt(((16*R)/5)*np.trapz((C**2)/Sh,freq,dx_values)))
    def get_view(self,Tobs=1):
        C=self.get_strain()
        Sh=(detector(self.detector).get_asd()[1])**2
        freq=detector(self.detector).get_asd()[0]
        curve_color = "#{:02x}{:02x}{:02x}".format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        R=1/5
        if self.detector=='test':
            R=1
        if self.detector=='LISA':
            R=1
            M_total = self.m1+self.m2
            eta = self.m1*self.m2/(M_total**2)
            M_chirp=(eta**(3/5))*M_total
            M_x=(G*M_chirp*M)/c**3
            f_isco=(c**3)/(G*(6**(3/2))*M_total*M*np.pi)
            tau=(eta*Tobs*year*(c**3))/(G*5*M_total*M)
            f_initial=((c**3)/(8*np.pi*G*M_total*M*(tau**(3/8))))*(1+((11*eta/32)+743/2688)*tau**(-1/4))
            f_initial=min(max(f_initial,10**-4),0.1)
            if f_isco>0.1:
                f_end=0.1
            else:
                f_end=f_isco
            if f_initial==0.1:
                return 0
            init=np.argwhere(freq>=f_initial)[0][0]
            end=np.argwhere(freq>=f_end)[0][0]
            freq=freq[init:end]
            Sh=Sh[init:end]
            C=C[init:end]
        if self.detector=='ET':
            R=9/20
        if self.detector=='DECIGO':
            R=9/20
        plt.loglog(detector(self.detector).get_asd()[0],np.sqrt(detector(self.detector).get_asd()[0]*(detector(self.detector).get_asd()[1]**2)/R),color='k',label=r'$h_n(f,{},{})$'.format(self.detector,self.m1/(1+self.z)))
        plt.loglog(freq,np.sqrt(16/5)*C*freq,color=curve_color,label=r'$h_c(f,T={})$'.format(Tobs))
        plt.fill_between(freq,np.sqrt(16/5)*C*freq,np.sqrt(freq*(Sh)/R),where=(np.sqrt(16/5)*C*freq>np.sqrt(freq*
        (Sh)/R)),color=curve_color,alpha=0.4)
def optimal_snr(m1,m2,H0=70,om_m=0.3,om_l=0.7,detector=None,optimal=8):
    c=[np.exp(-i) for i in np.arange(0,50,1)]
    res=0
    count=0
    for i in range(len(c)):
        temp=res
        ans=minimize_scalar(lambda x:abs(bbh(m1=m1,m2=m2,z=x,H0=H0,om_m=om_m,om_l=om_l,detector=detector).get_snr()-8)+c[i]*(x-res)**2)
        res=ans.x
        if abs(bbh(m1=m1,m2=m2,z=res,H0=H0,om_m=om_m,om_l=om_l,detector=detector).get_snr()-8)<0.01:
            if res>120:
                warnings.warn('z_min>120 and will be clipped at 120',category=UserWarning)
                return 120
            else:
                return res
    init=0.001
    check=bbh(m1=m1,m2=m2,z=init,H0=H0,om_m=om_m,om_l=om_l,detector=detector).get_snr()
    while check>8:
        check=bbh(m1=m1,m2=m2,z=init,H0=H0,om_m=om_m,om_l=om_l,detector=detector).get_snr()
        if bbh(m1=m1,m2=m2,z=init+5,H0=H0,om_m=om_m,om_l=om_l,detector=detector).get_snr()>8:
            init=init+5
        else:
            init=init+0.001
    return(init)
def get_background_snr(gw_background=None,conf=None,H0=70,Tobs_space=3,Tobs_ground=1):
        """
        Gives integrated SNR of the given background for the given detector and observation time(in years)
        Parameters:
            gw_background(callable function):A callable function representing the gravitational background
            conf(int):An integer between 1 and 12 
            H0 (float) : Present day value of Hubble's Constant in km/(Mpc)
            Tobs_space(float) : Observation time for space based detector in years .Default is 3
            Tobs_ground(float) : Observation time for ground based detector in years .Default is 1
        Returns:
            float:SNR 
        """
        if gw_background==None :
            raise ValueError("Input background not specified")
        if conf not in [1,2,3,4,5,6,7]:
             raise ValueError('conf has to be a integer between 1 and 7')
        if conf in [1,2,3,4,5,6]:
             f,cov=detector_network(conf=conf,H0=70)[0],detector_network(conf=conf,H0=70)[1]
             snr=np.sqrt(2*Tobs_ground*year)*np.sqrt((f[1]-f[0])*np.sum((gw_background(f)**2)/cov))
             return(snr)
        if conf==7:
            f,cov=detector_network(conf=conf,H0=70)[0],detector_network(conf=conf,H0=70)[1]
            snr=np.sqrt(Tobs_space*year)*(0.00025)*np.sqrt(np.sum((gw_background(f)**2)/cov))
            return(snr)
