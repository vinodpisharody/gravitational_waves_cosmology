import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import cosmology as cs
from scipy.interpolate import interp1d 
import scipy.stats as distribution
mpc=3.08568*10**22
year=365*24*3600
c=299792458
def pgap(m,alpha,beta,m_upper,m_lower,eta):
    A=1
    h=lambda m,m_low:(1+(m_low/m)**eta)**-1
    l=lambda m,m_high:(1+(m/m_high)**eta)**-1
    ans=[]
    if isinstance(m,np.ndarray) or isinstance(m,list):
        for _ in m:
            if _<=m_lower:
              ans.append(((m_lower/_)**alpha)*(1-A*(l(_,m_upper)*h(_,m_lower))))  
            else:
                ans.append(((m_lower/_)**beta)*(1-A*(l(_,m_upper)*h(_,m_lower))))
        return ans
    else:
            _=m
            if _<=m_lower:
              return(((m_lower/_)**alpha)*(1-A*(l(_,m_upper)*h(_,m_lower))))  
            else:
                return(((m_lower/_)**beta)*(1-A*(l(_,m_upper)*h(_,m_lower))))
def lgap(m,alpha,sigma,m_upper,m_lower,eta):
    A=1
    h=lambda m,m_low:(1+(m_low/m)**eta)**-1
    l=lambda m,m_high:(1+(m/m_high)**eta)**-1
    ans=[]
    if isinstance(m,np.ndarray) or isinstance(m,list):
        for _ in m:
            if _<=m_lower:
              ans.append((1/alpha)*distribution.lognorm.pdf(_/alpha,sigma)*(1-A*(l(_,m_upper)*h(_,m_lower))))  
            else:
                ans.append((((1/alpha)*distribution.lognorm.pdf(_/alpha,sigma))*(1-A*(l(_,m_upper)*h(_,m_lower)))))
        return ans
    else:
            _=m
            if _<=m_lower:
              return((1/alpha)*distribution.lognorm.pdf(_/alpha,sigma)*(1-A*(l(_,m_upper)*h(_,m_lower))))  
            else:
                return((1/alpha)*distribution.lognorm.pdf(_/alpha,sigma)*(1-A*(l(_,m_upper)*h(_,m_lower))))
class events():
    def __init__(self,a=2.7,z_peak=2.9,b=5.6,z=None,H0=70,om_m=0.3
                 ,om_l=0.7,z_pivot=0,R_pivot=30):
        self.a=a
        self.z_peak=z_peak
        self.b=b
        self.z=z
        self.H0=H0
        self.om_m=om_m
        self.om_l=om_l
        self.z_pivot=z_pivot
        self.R_pivot=R_pivot

    def get_star_formation(self):
        zz=np.arange(self.z[0],self.z[1]+0.1,0.1)
        sfr=[((1+_)**self.a)/(1+((1+_)/self.z_peak)**self.b) for _ in zz]
        return(interp1d(zz,sfr,fill_value='extrapolate'))
    def get_merger_rate(self,delay_time_distribution=lambda td:td**-1,t_min=0.1,t_max=13.5 ):
        cosmo=cs.cosmology(H0=self.H0)
        sfr=lambda z:((1+z)**self.a)/(1+((1+z)/self.z_peak)**self.b)
        R=[]
        H=self.H0/(mpc*year)
        zz=np.arange(self.z[0],self.z[1]+0.1,0.1)
        delay_time_normalize_factor=quad(lambda x:delay_time_distribution(x),t_min,t_max)[0]            ## For normalizing the delay time distribution
        for i in range(np.size(zz)):
            z_initial=cosmo.get_redshift(cosmo.get_look_time(zz[i],in_gyr=True)+t_min,quant=0)
            z_final=cosmo.get_redshift(cosmo.get_look_time(zz[i],in_gyr=True)+t_max,quant=0)
            if z_final>40:
                z_final=40
            if z_initial>40:
               R.append(0)
            else:
                integrand= lambda z:(((1+z)*(H)*((self.om_m*((1+z)**3))+self.om_l)**(0.5))**(-1))*(sfr(z)/delay_time_normalize_factor)*(delay_time_distribution((cosmo.get_look_time(z,in_gyr=True)-cosmo.get_look_time(zz[i],in_gyr=True))))
                R.append(quad(integrand,z_initial,z_final)[0])
        R=np.array(R)
        if R[np.argwhere(zz>=self.z_pivot)[0][0]]==0:
            return (interp1d(zz,np.zeros(np.size(R))))
        return(interp1d(zz,self.R_pivot*R/R[np.argwhere(zz>=self.z_pivot)[0][0]],fill_value='extrapolate'))
class mass_dist():
    def __init__(self):
        pass
    def get_stellar(m=(5,100),l=0.005,mc=45,sigma=1,beta=1): 
        m=np.arange(m[0],m[1]+0.1,0.1)
        arr=(1-l)*(1/m**beta)+l*(1/(np.sqrt(2*np.pi*(sigma**2))))*np.exp(-((m-mc)**2)/(2*sigma**2))
        return(interp1d(m,arr/np.trapz(arr,m,dx=0.1)))
    def get_pop3_1(m=(30,700),beta=1,eta=50,lower=45,upper=260):
        m=np.arange(m[0],m[1]+0.1,0.1)
        arr=pgap(m,alpha=beta,beta=beta,m_upper=upper,m_lower=lower,eta=eta)
        return(interp1d(m,arr/np.trapz(arr,m,dx=0.1)))
    def get_pop3_2(m=(30,700),mc=10,sigma=0.5,eta=50,lower=45,upper=260):
        m=np.arange(m[0],m[1]+0.1,0.1)
        arr=lgap(m,alpha=mc,sigma=sigma,m_upper=upper,m_lower=lower,eta=eta)
        return(interp1d(m,arr/np.trapz(arr,m,dx=0.1)))

