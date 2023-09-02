import numpy as np
import cosmology as cs
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import warnings
import random
warnings.filterwarnings("ignore")
from tqdm import tqdm
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
        plt.legend()
    def optimal_snr(m1,m2,H0=70,om_m=0.3,om_l=0.7,detector=None,optimal=8):
    c=[np.exp(-i) for i in np.arange(0,50,1)]
    res=0
    count=0
    for i in range(len(c)):
        temp=res
        ans=minimize_scalar(lambda x:abs(bbh(m1=m1,m2=m2,z=x,H0=H0,om_m=om_m,om_l=om_l,detector=detector).get_snr()-8)+c[i]*(x-res)**2)
        res=ans.x
        if abs(bbh(m1=m1,m2=m2,z=res,H0=H0,om_m=om_m,om_l=om_l,detector=detector).get_snr()-8)<0.1:
            return res
    else:
        raise ValueError("Invalid")

