import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d as ipd
from background import background
import os
import ray
from detector import detector_network
import warnings
import pygtc
year=365*3600*24
def generate_variations(true_values,step_size=False):
    variations = []
    step_size_h=[]
    # Generate variations for each true value
    for i, true_value in enumerate(true_values):
        variation = true_values[:]
        if true_value>=1 or true_value<=0.1:
            variation[i] = true_value - 10**(int(np.floor(np.log10(abs(true_value))))-1)  # Decrement the parameter by 1
            variations.append(variation[:])
            step_size_h.append(10**(int(np.floor(np.log10(abs(true_value))))-1))
    
            variation[i] = true_value + 10**(int(np.floor(np.log10(abs(true_value))))-1)  # Increment the parameter by 1
            variations.append(variation[:])
        else:
            variation[i] = true_value - 10**(int(np.floor(np.log10(abs(true_value)))))  # Decrement the parameter by 1
            variations.append(variation[:])
            step_size_h.append(10**(int(np.floor(np.log10(abs(true_value))))) )
            variation[i] = true_value + 10**(int(np.floor(np.log10(abs(true_value)))))  # Increment the parameter by 1
            variations.append(variation[:])
    if step_size==True:
        return step_size_h
    return np.round(np.array(variations),decimals=7)
class fisher():
    def __init__(self,merger,primary_mass,secondary_mass,R0=lambda m1,m2:1,truth=None,cache=False,cache_dir=None,detector=None,frequency_band=None):
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
        self.R0=R0
        if cache==False:
            self.cache=cache
            warnings.warn('Generated data wont be saved')
        else:
            self.cache=cache
            if cache_dir==None:
                warnings.warn('Cache Directory not specified .Using deafult value')
                current_dir = os.getcwd()
                self.cache_dir = os.path.join(current_dir,'fisher_cache')
                if not os.path.exists( self.cache_dir ):
                    os.makedirs( self.cache_dir )
            else:
                self.cache_dir=cache_dir
        variables=[item for item in list(self.merger[0].__code__.co_varnames) if item !='z']+[item for item in list(self.primary_mass[0].__code__.co_varnames) if item !='z' and item!='m1' and item!='m2' and item!='m']
        if len(variables)!=len(truth):
            raise ValueError('Number of Truth Value does not match the number of free parameters')
        else:
            self.truth=truth
            self.variable=len(variables)
    def get_fisher_matrix(self,Tobs=5,sampling_rate=None):
        if sampling_rate==None:
            raise ValueError('Invalid Value for Sampling rate ')
        parm_count=len(self.truth)
        variations = generate_variations(self.truth)
        futures=[]
        r=self.merger[0]
        p1=self.primary_mass[0]
        p2=self.secondary_mass[0]
        # Extract variable names from g and h
        variable_names_r = r.__code__.co_varnames
        variable_names_p1 = p1.__code__.co_varnames
        filename=''
        for i in range(len(self.truth)):
            filename=filename+str(self.truth[i])
        if os.path.exists(r'{}/imbh_{}_{}.csv'.format(self.cache_dir,self.detector,filename)):
            ans=np.array(pd.read_csv(r'{}/imbh_{}_{}.csv'.format(self.cache_dir,self.detector,filename)).transpose())
        else:
            for variation in variations:
                truth=tuple(variation)
                # Create new lambda functions with true values as default arguments
                if len([_ for _ in list(variable_names_r) if _!='z' and _!='m1' and _!='m2'])==0:
                    p1_true = lambda z, m1, *args, **kwargs: p1(z, m1, *(truth[len([_ for _ in list(variable_names_r)  if _!='z' and _!='m1' and _!='m2']):]), *args, **kwargs)
                    p2_true = lambda z, m2, *args, **kwargs: p2(z, m2, *(truth[len([_ for _ in list(variable_names_r)  if _!='z' and _!='m1' and _!='m2']):]), *args, **kwargs)
                    r_true=r
                elif len([_ for _ in list(variable_names_p1)  if _!='z' and _!='m1' and _!='m2'])==0:
                    r_true = lambda z, *args, **kwargs: r(z, *(truth[:len([_ for _ in list(variable_names_p1) if _ != 'z' and _ != 'm1' and _ != 'm2']) + 1] + args), **kwargs)
                    p1_true=p1
                    p2_true=p2
                else:
                    r_true = lambda z, *args, **kwargs: r(z, *(truth[:len([_ for _ in list(variable_names_r)  if _!='z' and _!='m1' and _!='m2'])] + args), **kwargs)
                    p1_true = lambda z,m1, *args, **kwargs: p1(z, m1, *(truth[len([_ for _ in list(variable_names_r)  if _!='z' and _!='m1' and _!='m2']):]), *args, **kwargs)
                    p2_true = lambda z,m2, *args, **kwargs: p2(z,m2, *(truth[len([_ for _ in list(variable_names_r)  if _!='z' and _!='m1' and _!='m2']):]), *args, **kwargs)
                zz=np.arange(self.merger[1],self.merger[2]+0.5,0.5)
                R1=r_true(zz)
                merger_rate_interpolated=lambda z:np.interp(z,zz,R1)
                futures.append(background.remote(
                    merger=(lambda z:merger_rate_interpolated(z=z),self.merger[1],self.merger[2]),
                    primary_mass=(lambda m, z:p1_true(m1=m,z=z),self.primary_mass[1],self.primary_mass[2]),
                    secondary_mass=(lambda m, z:p2_true(m2=m,z=z),self.secondary_mass[1],self.secondary_mass[2]),
                    R0=lambda m1,m2:self.R0,
                    frequency_band=self.frequency_band,detector=self.detector,
                    verbose=True
                ))
                # Print the result of g * h(x=1, y=1)
            ans=ray.get([cc.get_background_analytic.remote(tol_z=0.1,tol_m=10) for cc in futures])
            if self.cache==True:
                dataframe=pd.DataFrame(ans)
                dataframe=dataframe.transpose()
                column=[f'parm{i}' for i in range(1,len(self.truth)+1) for _ in range(2)]
                dataframe.columns=column
                filename=''
                for i in range(len(self.truth)):
                    filename=filename+str(self.truth[i])
                dataframe.to_csv(r'{}/imbh_{}_{}.csv'.format(self.cache_dir,self.detector,filename),index=False)
        F=np.zeros([parm_count,parm_count])
        parm_h = generate_variations(self.truth,step_size=True)
        mu=[ipd(self.frequency_band,(ans[2*i+1]-ans[2*i])/(2*parm_h[i]),fill_value='extrapolate') for i in range(int(len(ans)/2))]
        noise=detector_network(conf=self.detector)
        f_points=np.arange(min(self.frequency_band),max(self.frequency_band)+sampling_rate,sampling_rate)
        for i in range(parm_count):
            for j in range(i,parm_count):
                if self.detector=='LISA':
                    F[i][j]=np.sum(Tobs*year*sampling_rate*mu[i](f_points)*mu[j](f_points)*1/(noise(f_points)**2))
                    F[j][i]=F[i][j]
                else:
                    F[i][j]=np.sum(2*Tobs*year*sampling_rate*mu[i](f_points)*mu[j](f_points)*1/(noise(f_points)**2))
                    F[j][i]=F[i][j]
        return(F)
    def get_corner_plot(self,Tobs=5,sampling_rate=None,prior_information=None,points=20000,label=None):
        if label==None:
            warnings.warn('No label given for the chain .So assuming label to be name of the current detector')
            label=self.detector
        if sampling_rate==None:
            raise ValueError('Invalid Value for Sampling rate ')
        if prior_information==None:
            F=self.get_fisher_matrix(Tobs=Tobs,sampling_rate=sampling_rate)
        else:
            F=np.matrix(prior_information) + np.matrix(self.get_fisher_matrix(Tobs=Tobs,sampling_rate=sampling_rate))
        cov=np.linalg.inv(F)
        rng = np.random.default_rng()
        mean = np.array(self.truth)
        samples=rng.multivariate_normal(mean,cov, size=points)
        names = ['parm{}'.format(i+1) for i in range(len(self.truth))]
        chainLabels = [label]
        truths = (tuple(self.truth))
        GTC = pygtc.plotGTC(chains=[samples],smoothingKernel=2.1,nBins=75,
                    paramNames=names,
                    chainLabels=chainLabels,plotDensity=False,
                    truths=truths,customLabelFont={'family':'Arial', 'size':16},customLegendFont={'family':'Arial', 'size':10},
                    priors=None,
                    figureSize='APJ_page',filledPlots=True,colorsOrder=['purples','blues','greens','reds','cyans','oranges'])



