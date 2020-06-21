import os
import math
import numpy as np

import scipy.linalg as sc
import scipy.special as sp
import scipy.interpolate as spint
import scipy.spatial.qhull as qhull

import xarray as xr


class Atmosphere_Interpolator():
    """ 
    This class creates weights for interpolation for atmospheric fields.

    This class create an interpolator operator from a *source grid* 
    `src_grid` to a *target grid* `tgt_grd`. The interpolator then
    can be used to perform the actual interpolation.

    The *source grid* must be a `xarray` `DataSet` containing coordinates
    `latitude` and `longitude`.

    The *target grid* must be a `xarray` `DataArray` with variables `lat` and `lon`

    Works only on single `DataArray`
    
    Parameters
    ----------

    src_grid : xarray
        Source grid
    tgt_grid : xarray
        Target grid
    
    Attributes
    ----------

    tgt_grid :
        Target grid 
    mask :
        Mask of the target grid
    vt :
        Weights
    wt :
        Weights
    
    Examples    
    --------    
    Create the weights for interpolation

    >>> w= zint.Ocean_Interpolator(src_grid,tgt_grid) 
    
    Interpolate

    >>> target_xarray=w.interp(src_xarray)

    """

    __slots__ = ('vt','wt','mask','tgt_grid')

    def __init__(self, src_grid, tgt_grid):
        # Put here info on grids to be obtained from __call__
        #Source Grid
        lato=src_grid.latitude.data.flatten()
        lono=src_grid.longitude.data.flatten()
        latlon=np.asarray([lato,lono])

        #Target Grid
        self.tgt_grid = tgt_grid
        mask = tgt_grid.stack(ind=('lat','lon')).dropna('ind')
        self.mask = mask
        #Order target coordinate
        latlon_to=np.asarray([mask.lat.data,mask.lon.data])
        
        #Generates Weights
        tri = qhull.Delaunay(latlon.T)
        simplex = tri.find_simplex(latlon_to.T)
        self.vt = np.take(tri.simplices, simplex, axis=0)

        temp = np.take(tri.transform, simplex, axis=0)
        delta = latlon_to.T - temp[:, 2]
        bary = np.einsum('njk,nk->nj', temp[:, :2, :], delta)
        self.wt = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

    def __call__(self):
        return

    def __repr__(self):
        '''  Printing other info '''
        return '\n' 
    
    def interp(self, xdata):
        ''' 
        Perform actual interpolation to the target grid
        
        '''
        dat= np.expand_dims(xdata.data.flatten(),axis=1)
        new = np.einsum('nj,nj->n', np.take(dat, self.vt), self.wt)
        temp=self.mask.copy()
        temp.data = new
        temp.name = xdata.name
        return temp.unstack()



class Ocean_Interpolator():
    """ 
    This class creates weights for interpolation of ocean fields.

    This class create an interpolator operator from a *source grid* 
    `src_grid` to a *target grid* `tgt_grd`. The interpolator then
    can be used to perform the actual interpolation.

    The *source grid* must be a `xarray` `DataSet` containing coordinates
    `latitude` and `longitude`.

    The *target grid* must be a `xarray` `DataArray` with variables `lat` and `lon`

    Works only on single `DataArray`
    
    Parameters
    ----------

    src_grid : xarray
        Source grid
    tgt_grid : xarray
        Target grid
    
    Attributes
    ----------

    tgt_grid :
        Target grid 
    mask :
        Mask of the target grid
    vt :
        Weights
    wt :
        Weights
    
    Examples    
    --------    
    Create the weights for interpolation

    >>> w= zint.Ocean_Interpolator(src_grid,tgt_grid) 
    
    Interpolate

    >>> target_xarray=w.interp(src_xarray)

    """

    __slots__ = ('vt','wt','mask','tgt_grid')

    def __init__(self, src_grid, tgt_grid):
        # Put here info on grids to be obtained from __call__
        #Source Grid
        lato=src_grid.latitude.data.flatten()
        lono=src_grid.longitude.data.flatten()
        latlon=np.asarray([lato,lono])

        # #Recognize Grid
        # if 'CMCC-CM2-HR4' in src_grid.name :
        #     print(f'Interpolating from high resolution 1/4 grid \t {src_grid.name}')

        # else: 
        #     sys.exit('Unrecognized Grid')

        # if 'grid_V' or 'grid_U' in src_grid.name :
        #     print(f' Velocity Grid ')
        # else:
        #     print(f' Temperature Grid ')
        
        #Target Grid
        self.tgt_grid = tgt_grid
        mask = tgt_grid.stack(ind=('lat','lon')).dropna('ind')
        self.mask = mask
        #Order target coordinate
        latlon_to=np.asarray([mask.lat.data,mask.lon.data])
        
        #Generates Weights
        tri = qhull.Delaunay(latlon.T)
        simplex = tri.find_simplex(latlon_to.T)
        self.vt = np.take(tri.simplices, simplex, axis=0)

        temp = np.take(tri.transform, simplex, axis=0)
        delta = latlon_to.T - temp[:, 2]
        bary = np.einsum('njk,nk->nj', temp[:, :2, :], delta)
        self.wt = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

    def __call__(self):
        return

    def __repr__(self):
        '''  Printing other info '''
        return '\n' 
    
    def interp_T(self, xdata):
        ''' 
        Perform interpolation for T Grid point to the target grid.
        This methods can be used for scalar quantities.

        Parameters
        ----------
        xdata :  xarray
            2D array to be interpolated, it must be on the `src_grid`
        
        Returns
        -------
        out :  xarray
            Iterpolated xarray on the target grid
        
        '''
        dat= np.expand_dims(xdata.data.flatten(),axis=1)
        new = np.einsum('nj,nj->n', np.take(dat, self.vt), self.wt)
        temp=self.mask.copy()
        temp.data = new
        temp.name = xdata.name
        return temp.unstack()

    def interp_UV(self, xdata):
        ''' 
        Perform actual interpolation to the target grid
        Under development
        '''
        
        return 

