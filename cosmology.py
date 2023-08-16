import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings("ignore")
mpc=3.08568*10**22
year=365*24*3600
c=299792458
class cosmology():
    """
            A class representing the cosmology
            Attributes:
                H0 (float) : Present day value of Hubble's Constant in km/(Mpc*year)
                om_m (float) : Matter density
                om_m (float) : Vacuum density
    """
    def __init__(self,H0=67.8,om_m=0.300,om_l=0.7):
        self.H0=H0
        self.om_m=om_m
        self.om_l=om_l
        if self.om_m<0 or self.om_l<0 or self.om_m+self.om_l!=1:
            raise ValueError(r'matter density,vacuum density must be non-negative and add to 1')
    def get_lum_dist(self,z,in_mpc=False):
        """
        Gives the Luminosity distance(in meters)
        Parameters:
            z(float):The redshift upto which you want the luminosity distance
            in_mpc(Bool):Gives the Luminosity in Mpc
        Returns:
            float:Luminosity distance for the given redshift
        """
        H=self.H0*1000*(mpc)**-1
        out=(c/H)*(quad(lambda x:(((self.om_m*((1+x)**3))+self.om_l)**(0.5))**(-1),0.0,z)[0])*(1+z)
        if in_mpc==True:
            return out/mpc
        else :
            return out
    def get_look_time(self,z,in_gyr=False):
        """
        Gives the Lookback time (in seconds)
        Parameters:
            z(float):The redshift at which you want the lookback time
            in_gyr(Bool):Gives the lookback time in Gyrs
        Returns:
            float:Lookback time of the given redshift
        """
        H=self.H0*1000*(mpc)**-1
        out=(1/H)*(quad(lambda x:((1+x)*(((self.om_m*((1+x)**3))+self.om_l)**(0.5)))**(-1),0.0,z)[0])
        if in_gyr==True:
            return out/(year*(10**(9)))
        else :
            return out
    def get_redshift(self,y=None,quant=-1):
        """
        Gives the redshift 
        Parameters:
            y (float):The lookback time (in Gyr) or luminosity distance (in Mpc)
            quant(int): 0 if you have given lookback time and 1 if you have given luminosity distance
        Returns:
            float:redshift
        """
        if y==None or y<0:
            raise ValueError("Invalid Input")
        if quant in [0,1]:
            if quant==0:
                return(np.round(minimize_scalar(lambda x:abs(self.get_look_time(x,in_gyr=True)-y)).x,3))
            else:
                return(np.round(minimize_scalar(lambda x:abs(self.get_lum_dist(x,in_mpc=True)-y)).x,3))
        else:
            raise ValueError("quant can only be 0 or 1")