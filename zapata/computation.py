import os
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sc
import numpy.linalg as lin
import xarray as xr
import pandas as pd
import cartopy.crs as car

import scipy.ndimage as ndimage

import zapata.lib as lib
import zapata.data as era

from geocat.viz import cmaps as gvcmaps
from geocat.viz import util as gvutil
import tqdm as tm
import mpl_toolkits.axes_grid1 as tl

def zonal_var(dataset,var,season,option='LonTime',verbose=False):
    """
    A routine to average xarray 
    
    This routine will accept xarray up to four dimensions (lat,lon,pressure, time) and return the averaged arrays with compatible dimensions.
    
    Parameters
    ----------
    dataset :      
        Name of the dataset, ``ERA5``, ``GPCP``       
    var :   
        Variable
    season :     
        Month or Season. Resolved from `dat_param`
    option :       
        Control Averaging   
            -  None        No Averaging   
            - 'LonTime'    Longitude and Time   
            - 'Lon'        Longitude    
            - 'Time'       Time averaging   
    verbose:    
        Tons of Output
    
    Returns
    --------
    
    average :
        Average. The dimension is depending on the averaging option chosen
    
    Examples
    --------
    
    >>> zonal_var('ERA5','Z','JAN',option='LonTime')   #Longitude and Time Average for Z from ERA5
    >>> zonal_var('GPCP','TPREP','DJF',option='Time',verbose=True)   # Time average 
    """

    info=era.DataGrid()
    lev=info[dataset][var]['level']
    nlev=len(lev)
    xx=era.read_xarray(dataset='ERA5',var=var,level=str(lev[0]),season=season,verbose=verbose)
    
    if option == 'LonTime':
        zon=xr.DataArray.expand_dims(xx.mean(dim='lon').mean(dim='time'),dim='pressure').assign_coords(pressure=[lev[0]])
        print(' Averaging on longitude and time ')
    elif option == 'Time':
        zon=xr.DataArray.expand_dims(xx.mean(dim='time'),dim='pressure').assign_coords(pressure=[lev[0]])
        print(' Averaging on  time ')
    elif option == 'Lon':
        zon=xr.DataArray.expand_dims(xx.mean(dim='lon'),dim='pressure').assign_coords(pressure=[lev[0]])
        print(' Averaging on longitude ')
    else:
        zon=xr.DataArray.expand_dims(xx,dim='pressure').assign_coords(pressure=[lev[0]])

    for i in tm.tnrange(1,nlev):
        xx=era.read_xarray(dataset='ERA5',var=var,level=str(lev[i]),season=season)
        if option == 'LonTime':
            xx1=xr.DataArray.expand_dims(xx.mean(dim='lon').mean(dim='time'),dim='pressure').assign_coords(pressure=[lev[i]])
        elif option == 'Time':
            xx1=xr.DataArray.expand_dims(xx.mean(dim='time'),dim='pressure').assign_coords(pressure=[lev[i]])
        elif option == 'Lon':
            xx1=xr.DataArray.expand_dims(xx.mean(dim='lon'),dim='pressure').assign_coords(pressure=[lev[i]])
        else:
            xx1=xr.DataArray.expand_dims(xx,dim='pressure').assign_coords(pressure=[lev[i]])       
        zon=xr.concat([zon,xx1],dim='pressure')
        
    return zon

def smooth_xarray(X,sigma=5,order=0,mode='wrap'):
    """
    Smooth xarray X with a gaussian filter . 

    It uses a routine from scipy ndimage ( ``ndimage.gaussian_filter``). 
    The filter is applied to all dimensions.
    See the doc page of ( ``ndimage.gaussian_filter``) for a full documentation.
    The filter can be used for periodic fields, then the correct setting of `mode` is 'wrap'

    Parameters
    -----------
    X :  
        Input Xarray    
    
    sigma:  
        Standard deviation for the Gaussian kernel

    order:  
        Order of the smoothing, 0 is a simple convolution
    
    mode:   

        The mode parameter determines how the input array is extended when the filter overlaps a border. 
        By passing a sequence of modes with length equal to the number of dimensions of the input array, 
        different modes can be specified along each axis. Default value is ‘reflect’. 
        The valid values and their behavior is as follows:

        ‘reflect’ (d c b a | a b c d | d c b a)
        The input is extended by reflecting about the edge of the last pixel.
        
        ‘constant’ (k k k k | a b c d | k k k k)
        The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.
        
        ‘nearest’ (a a a a | a b c d | d d d d)
        The input is extended by replicating the last pixel.

        ‘mirror’ (d c b | a b c d | c b a)
        The input is extended by reflecting about the center of the last pixel.

        ‘wrap’ (a b c d | a b c d | a b c d)
        The input is extended by wrapping around to the opposite edge.

    Returns
    --------
    smooth_array:   
        Smoothed array

    Examples
    --------
    _Smooth a X[lat,lon] array with nearest repetition in `lat` and periodicity in `lon`_
    >>> smooth_array(X,sigma=5,order=0,mode=['nearest','wrap']) 
    
    """
    lat = X.lat
    lon = X.lon
    temp = ndimage.gaussian_filter(X, sigma=sigma, order=order,mode=mode)
    zarray = xr.DataArray(temp,dims=('lat','lon'),coords={'lat':lat,'lon':lon})
    return zarray

def anomaly(var,option='anom',freq='month'):
    """
    Compute Anomalies according to 'option'.

    Long description here.

    Parameters
    ----------

    var :   
        xarray to compute anomalies

    option :    
        'deviation' _Subtract the total time mean of the time series_   
        'anom'      _Compute anomalies from monthly climatology_    
        'anomstd'   _Compute standardized anomalies from monthly climatology_

    freq :  
        Frequency of data   

    Returns
    -------

    anom :  xarray

    """
    frequency = 'time.' + freq
    if option == 'deviation':
        anom = var - var.mean(dim='time')
    elif option == 'anom':
        clim = var.groupby(frequency).mean("time")
        anom = var.groupby(frequency) - clim
    elif option == 'anomstd':
        clim = var.groupby(frequency).mean("time")
        climstd = var.groupby(frequency).std("time")
        anom = xr.apply_ufunc(lambda x, m, s: (x - m) / s,
                var.groupby(frequency),
                clim,
                climstd )
    else:
        print(' Wrong option in `anomaly` {}'.format(option))
        raise SystemExit

    return anom
class Xmat():
    """ This class creates xarrays in mathematical form.

    The xarray is stacked along `dims` dimensions
    with the spatial values as column vectors and time as the 
    number of columns
    
    Examples    
    --------    
    Create a stacked data matrix along the 'lon' 'lat dimension
    >>> Z = Xmat(X, dims=('lat','lon'))
    """

    __slots__ = ('A','_ntime')

    def __init__(
        self,
        X,
        dims: Union[Hashable, Sequence[Hashable], None] = None,
        ):
        self.A = X.stack(z=dims).transpose()
        """Stacked Matrix of type `xarray`"""
        self._ntime = len(X.time.data)
        print(' Created mathematical matrix A, \n \
                stacked along dimensions {} '.format(dims))
        
        

    def __call__(self, v ):
        ''' Matrix vector evaluation.'''
        f = self.a @ v
        return f

    def __repr__(self):
        '''  Printing Information '''
        print(' \n Math Data Matrix \n {} \n'.format(self.A))
        print(f' Shape of A numpy array {self.A.shape}')
        return  '\n'
     
    def svd(self, N=10):
        '''Compute SVD of Data Matrix A.
        
        The calculation is done in a way that the modes are equivalent to EOF

        Parameters
        ----------
        
        N :  
            Number of modes desired.     
            If it is larger than the number of `time` levels    
            then it is set to the maximum

        Results
        -------

        out : dictionary
            Dictionary including    
        * **Pattern**               EOF patterns    
        * **Singular_Values**       Singular Values 
        * **Coefficient**           Time Coefficients   
        * **Varex**                 Variance Explained  

        Examples         
        --------     
        >>> out = Z.svd(N=10)   
        '''
        #Limit to maximum modes to time levels
        Neig = np.min([N,self._ntime])
        print(f'Computing {Neig} Modes')
        # Prepare arrays
        len_modes = self._ntime
        u = self.A.isel(time=range(Neig)).rename({'time': 'Number'}).assign_coords(Number= range(Neig))
        u.name = 'Modes'
        
        #Compute modes
        _u,_s,_v=sc.svd(self.A,full_matrices=False)
    
        #EOF Patterns
        u.data = _u[:,0:Neig]
        #Singular values
        s = xr.DataArray(_s[0:Neig], dims='Modes',coords=[np.arange(Neig)])
        #Coefficients
        vcoeff = xr.DataArray(_v[0:Neig,:], dims=['Modes','Time'],coords=[np.arange(Neig),self.A.time.data])
        # Compute variance explained
        _varex = _s**2/sum(_s**2)
        varex = xr.DataArray(_varex[0:Neig], dims='Modes',coords=[np.arange(Neig)])

        #Output
        out = xr.Dataset({'Pattern':u,'Singular_Values': s, 'Coefficient': vcoeff, 'Varex': varex})
        return out

    def corr(self,y, Dim =('time') ):
        """
        Compute correlation of data matrix `A` with index `y`.

        This method compute the correlation of the data matrix
        with an index of the same length of the `time` dimension of `A`

        Parameters
        ----------
        y : xarray  
            Index, should have the same dimension length `time` 
        
        Examples
        --------
        Correlation of data matrix `Z` with `index`
        >>> corr = Z.corr(index)
        """
        index= y - y.mean(dim=Dim)
        _corr = (self.A - self.A.mean(dim=Dim)).dot(index)
        return _corr / (self.A.std(dim=Dim) * y.std(dim=Dim))/self._ntime
    
    def cov(self,y, Dim =('time') ):
        """
        Compute covariance of data matrix `A` with `index`.

        This method compute the correlation of the data matrix
        with an index of the same length of the `time` dimension of `A`

        Example
        -------
        Covariance of data matrix `Z` with `index`
        >>> cov = Z.cov(index)

        """
        index= (y - y.mean(dim=Dim))
        _cov = (self.A - self.A.mean(dim=Dim)).dot(index)/self._ntime
        return _cov

    def anom(self,**kw):
        """ Creates anomalies.

        This is using the function `anomaly` from `zapata.computation` 
        """

        self.A = anomaly(self.A,**kw)
        return 
    
      
