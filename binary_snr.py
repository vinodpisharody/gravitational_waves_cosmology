import numpy as np
import cosmology as cs
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings("ignore")
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
sensitivity_lisa = np.sqrt(sensitivity_lisa)
# 4 CE
with open('cosmic_explorer_noise.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))
freq_ce = np.empty(0)
sensitivity_ce = np.empty(0)
for i in range(len(data)):
    freq_ce = np.append(freq_ce, float(data[i][0]))
    sensitivity_ce = np.append(sensitivity_ce, float(data[i][1]))
# 4 ET
with open('ET_noise.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))
freq_et = np.empty(0)
sensitivity_et = np.empty(0)
for i in range(len(data)):
    freq_et = np.append(freq_et, float(data[i][0]))
    sensitivity_et = np.append(sensitivity_et, float(data[i][3]))
def phenomA(f,m1,m2,z):
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
    def get_psd(self):
        if self.detector=="LIGO" or self.detector=="VIRGO" or self.detector=="KAGRA":
            return(freq_a_plus,sensitivity_a_plus)
        elif self.detector=="ET" or self.detector=="EINSTEIN TELESCOPE":
            return(freq_et,sensitivity_et)
        elif self.detector=="CE" or self.detector=="COSMIC EXPLORER":
            return(freq_ce,sensitivity_ce)
        elif self.detector=="DECIGO":
            return(freq_decigo,sensitivity_decigo)
        elif self.detector=="LISA":
            return(freq_lisa,sensitivity_lisa)
class bbh():
    def __init__(self,m1=30,m2=30,z=1,H0=67.9,om_m=0.3,om_l=0.7,detector=None):
        self.m1=m1*(1+z)
        self.m2=m2*(1+z)
        self.z=z
        self.H0=H0
        self.om_m=om_m
        self.om_l=om_l
        self.detector=detector
    def get_strain(self):
        A=phenomA(detector(self.detector).get_psd()[0],self.m1, self.m2,self.z)
        A=A/(cs.cosmology(self.H0,self.om_m,self.om_l).get_lum_dist(self.z))
        return(A)
    def get_snr(self,response='mean'):
       C=self.get_strain()
       if response=='median':
           Sh=(detector(self.detector).get_psd()[1])**2
           freq=detector(self.detector).get_psd()[0]
           dx_values = np.diff(freq)
           return(np.sqrt((1.71/4)*np.trapz((C**2)/Sh,freq,dx_values)))
       else:
           Sh=(detector(self.detector).get_psd()[1])**2
           freq=detector(self.detector).get_psd()[0]
           dx_values = np.diff(freq)
           return(np.sqrt((2.56/4)*np.trapz((C**2)/Sh,freq,dx_values)))             
def optimal_snr(m1,m2,H0,om_m,om_l,detector='LIGO',optimal=8):
    c=[np.exp(-1),np.exp(-2),np.exp(-3),np.exp(-4),np.exp(-5)]
    ans=0
    for i in range(len(c)):
        ans=np.round(minimize_scalar(lambda x:abs(bbh(m1=m1,m2=m2,z=x,H0=H0,om_m=om_m,om_l=om_l,detector=detector).get_snr()-8)+c[i]*(x-ans)**2).x,8)
    return ans
