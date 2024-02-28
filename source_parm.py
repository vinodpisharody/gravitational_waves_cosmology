import numpy as np
from scipy.integrate import quad, IntegrationWarning
import pandas as pd
from scipy.interpolate import interp1d as ipd
from numba import jit
from astropy import constants as const 
from astropy import units as u 
from astropy.cosmology import z_at_value 
from astropy.cosmology import Planck18 as cosmo
from scipy.stats import norm
import warnings
c=const.c.value
G=const.G.value
M=const.M_sun.value
gpc=const.pc.value*10**9
mpc=const.pc.value*10**6
year=365*24*3600
gyr=year*10**9
myr=gyr/1000
H=cosmo.H0.value*1000*(mpc)**-1
om_m=cosmo.Om0
om_l=cosmo.Ode0
om_k = cosmo.Ogamma0
warnings.filterwarnings("ignore",category=RuntimeWarning)

class mass_distribution():
    def __init__(self,zm,mpisn0=40,alpha=1.5,gamma=-1,t_min=0.1,k=1,b=1,s=5,l=0.05,m_min=2.5,m_max=100):
        self.zm=zm
        self.mpisn0=mpisn0
        self.alpha=alpha
        self.gamma=gamma
        self.t_min=t_min
        self.k=k
        self.b=b
        self.s=s
        self.l=l
        self.m_min=m_min
        self.m_max=m_max
    def theta(self,zf,m,alt_model=None,flag=0):
        """
        theta function that takes in mass of black hole and parameters for the linear model of m_pisn/Alternatively you can 
        define a callable fucntion for m_pisn
        Parameters:
            zf(float):Redshift where black holes are forming
            m(float):Mass of the black hole in source frame
            alt_model(callable function) : optional alternate model for M_pisn
        returns
            int 0 or 1 depending on whether m>=m_pisn(zf) or m<m_pisn(zf) respectively 
        """
        if alt_model==None:
            mpisn=lambda z:self.mpisn0 -self.alpha*self.gamma*z
            if flag==1:
                return(mpisn(zf))
            if m<=mpisn(zf):
                return(1)
            else:
                return 0 
        else:
            if flag==1:
                return(alt_model(zf))
            if m<=alt_model(zf):
                return(1)
            else:
                return 0 

    def window_function(self,m,delay_time_distribution=None,zmax=75,alt_model=None,catch_error=False):
        if delay_time_distribution==None:
            delay_time_distribution = lambda x : 1/x**self.k  
        time_delay_norm=quad(lambda x:delay_time_distribution(x),self.t_min,cosmo.age(0).value,limit=200,epsabs=0,epsrel=0.1)[0]            ## For normalizing the delay time distribution
        E_z =lambda zz: np.sqrt(om_m*(1+zz)**3 + om_k*(1+zz)**2+ cosmo.Ode(zz))
        dt_dz= lambda z:-((1+z)*E_z(z)*H)**-1
        if catch_error==True:
            try:
                z_initial=z_at_value(func=cosmo.lookback_time, fval=cosmo.lookback_time(self.zm)+self.t_min*u.Gyr,zmax=zmax).value
            except:
                z_initial=zmax
        else:
            z_initial=z_at_value(func=cosmo.lookback_time, fval=cosmo.lookback_time(self.zm)+self.t_min*u.Gyr,zmax=zmax).value
        wint= lambda zf:(self.theta(zf=zf,m=m,alt_model=alt_model)/time_delay_norm)*(delay_time_distribution((cosmo.lookback_time(self.zm)-cosmo.lookback_time(zf)).value))*dt_dz(zf)
        ans=quad(lambda zf:wint(zf),z_initial,zmax,limit=200,epsrel=0.1,epsabs=0)[0]
        return(ans)


    def dist(self,m):
        b,u,s,l=self.b,self.theta(zf=self.zm,m=m,flag=1),self.s,self.l
        nm_gauss=quad(lambda m : 1/(np.sqrt(2*np.pi*s**2))*(np.exp(-(((m-u)/s)**2)/2)),self.m_min,self.m_max)[0]
        nm_pow=quad(lambda m :1/m**b,self.m_min,self.m_max)[0]
        return((1-l)/((m**b)*nm_pow)+l*(1/(np.sqrt(2*np.pi*s**2)*nm_gauss))*(np.exp(-(((m-u)/s)**2)/2)))


    def distribution(self,zmax=75,p=None,delay_time_distribution=None,alt_model=None,catch_error=False):
        N=quad(lambda m:self.window_function(m=m,delay_time_distribution=delay_time_distribution,zmax=zmax,alt_model=alt_model,catch_error=catch_error),self.m_min,self.m_max,limit=200,epsrel=0.1,epsabs=0)[0]
        ans=[]
        m=np.linspace(self.m_min,self.m_max,100)
        if p==None:
            for _ in m:
                ans.append(self.dist(m=_)*self.window_function(m=_,delay_time_distribution=delay_time_distribution,zmax=zmax,alt_model=alt_model,catch_error=catch_error)/N)
            ans=np.array(ans)
            norm=quad(lambda x: ipd(m,ans)(x),min(m),max(m),limit=200,epsrel=0.1,epsabs=0)[0]
            ans=ans/norm
            return([m,ans])
        else:
            for _ in m:
                ans.append(p(_)*self.window_function(m=_,delay_time_distribution=delay_time_distribution,zmax=zmax,alt_model=alt_model,catch_error=catch_error)/N)
            ans=np.array(ans)
            norm=quad(lambda x: ipd(m,ans)(x),min(m),max(m),limit=200,epsrel=0.1,epsabs=0)[0]
            ans=ans/norm
            return(m,ans)


# import matplotlib.pyplot as plt
# for i in [1.1,6.1]:
#     m=np.linspace(5,100,100)
#     W=mass_distribution(zm=i,t_min=0.1,b=1.5,l=0.05,k=1,m_min=5,m_max=60,gamma=-0.44,alpha=1.5,mpisn0=45)
#     x,y=W.distribution(p=lambda x:1/x**3.2,zmax=20,alt_model=lambda z : 45 -1.5*(-0.44*z+0.01+4))
#     plt.plot(x,y)
# plt.legend()
# plt.yscale('log')

class merger_rate():
    def __init__(self,t_min=1,t_max=cosmo.age(0),a=2.7,b=2.9,c=5.6,k=1,z_pivot=0,R_pivot=23):
        self.t_min=t_min
        self.t_max=t_max
        self.a=a
        self.b=b
        self.c=c
        self.z_pivot=z_pivot
        self.R_pivot=R_pivot
        self.k=k
    def merger_rate(self,delay_time_distribution=None,zmax=1000,zz=np.arange(0,50.5,0.5),alt_model=None):
        if delay_time_distribution==None:
            delay_time_distribution = lambda x : 1/x**self.k  
        R=[]
        if alt_model==None:
            md=lambda z:((1+z)**self.a)/(1+((1+z)/self.b)**self.c)
        else:
            md =lambda z: alt_model(z)
        time_delay_norm=quad(lambda x:delay_time_distribution(x),self.t_min,self.t_max.value)[0]            ## For normalizing the delay time distribution
        E_z =lambda zz: np.sqrt(om_m*(1+zz)**3 + om_l)
        dt_dz= lambda z:((1+z)*E_z(z)*H)**-1
        for i in range(np.size(zz)):
            try:
                z_initial=z_at_value(func=cosmo.lookback_time, fval=cosmo.lookback_time(zz[i])+self.t_min*u.Gyr,zmax=zmax).value
            except:
                z_initial=zmax
            try:
                z_final=z_at_value(func=cosmo.lookback_time, fval=cosmo.lookback_time(zz[i])+self.t_max,zmax=zmax).value
            except:
                z_final=zmax
            m1_integral= lambda zf:dt_dz(zf)*(md(zf)/time_delay_norm)*(delay_time_distribution((cosmo.lookback_time(zf)-cosmo.lookback_time(zz[i])).value))
            kkk=quad(m1_integral,z_initial,z_final,limit=1000,epsabs=0,epsrel=0.1)[0]
            R.append(kkk)
        R=np.array(R)
        R_interpolated = ipd(zz,R)
        if R_interpolated(self.z_pivot)==0:
            return np.zeros(np.size(R))
        return(zz,self.R_pivot*R/R_interpolated(self.z_pivot))
    def get_number_of_events(self,Tobs=1,delay_time_distribution=None,zmax=1000,zz=np.arange(0,50.5,0.5),alt_model=None):
        z,R=self.merger_rate(delay_time_distribution=delay_time_distribution,zmax=zmax,zz=zz,alt_model=alt_model)
        Rm =ipd(z,R)
        Rz = lambda z : Rm(z)*(cosmo.differential_comoving_volume(z)*4*np.pi*u.sr).to('Gpc^3')/(1+z)
        events = quad(lambda x : Rz(x).value,min(zz),max(zz),epsabs=0,epsrel=0.1,limit=100)[0]
        return(int(events*Tobs))



# zz,R1=merger_rate(t_min=0.1,R_pivot=23,z_pivot=0).merger_rate()
# zz,R3a=merger_rate(t_min=0.01,a=-5.92,c=-8.55,b=12.83,R_pivot=2,z_pivot=8.5).merger_rate()
# zz,R3b=merger_rate(t_min=0.01,a=3.7,c=12.2,b=20,R_pivot=2,z_pivot=15).merger_rate()
# zz,R3c=merger_rate(t_min=0.01,a=3.7,c=12.2,b=13.5,R_pivot=2,z_pivot=10).merger_rate()
# R4=R1+R3a
# # R1=_time_delay(t_min=20,a1=1.7,c1=4.2,b1=15,R_pivot=2,z_pivot=15)

# # R2=_time_delay(t_min=50,a1=3.2,c1=5.5,b1=7.5,R_pivot=0.01,z_pivot=0)

# plt.plot(zz,R1,label=r'Model 1 (POP-I/II)')
# plt.plot(zz,R3a,label=r'Model 3a POP-III [Liu-Bromm]')
# plt.plot(zz,R3b,label=r'Model 3b POP-III [H22]')
# plt.plot(zz,R3c,label=r'Model 3c POP-III [H22]')
# plt.plot(zz,R4,label=r'Model 4 POP I/II + III',linestyle='dashed')
# interp=ipd(zz,R4)
# plt.scatter(zz,interp(zz),color='r',s=0.7)
# # # plt.plot(zz,R2,label=r'$z_{peak}$=5')
# # plt.xlim(0,50)
# plt.ylim(bottom=10**-5,top=50)
# plt.title('Stellar Mass BBH Merger Rate')
# max_index = np.argmax(R3b)
# x_max, y_max = zz[max_index], R3b[max_index]
# plt.scatter(x_max, y_max, color='red', label=f'Maximum Point\n({x_max:.2f}, {y_max:.2f})')

# max_index = np.argmax(R3c)
# x_max, y_max = zz[max_index], R3c[max_index]
# plt.scatter(x_max, y_max, color='red', label=f'Maximum Point\n({x_max:.2f}, {y_max:.2f})')



# plt.yscale('log')
# plt.legend(fontsize='small')
# plt.grid(visible=True,which='both',color='gray',alpha=0.6)
# plt.xlabel('z')
# plt.ylabel(r'R(z) [$Gpc^{-3}year^{-1}$]')


