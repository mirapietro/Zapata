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
    This class creates weights for interpolation of atmospheric fields.
    No mask is used.

    This class create an interpolator operator to a *target grid* `tgt_grd`. The interpolator then
    can be used to perform the actual interpolation.

    The *target grid* must be a `xarray` `DataArray` with variables `lat` and `lon`

    Parameters
    ----------

    grid : str
        Choice of output grids  

        * `1x1`  -- Regular 1 degree 
        * `025x025`  -- Coupled model grid, nominally 0.25, 

    option : str
        'linear', interpolation method

    Attributes
    ----------

    name : str
        Name of the interpolator
    grid : str
        Option for the grid
    option : str
        Interpolation method

    
    Note
    ====

    It is a thin wrapper around `xarray` `interp_like` method.
    
    Examples    
    --------    
    Create the weights for interpolation

    >>> w= zint.Atmosphere_Interpolator('1x1','linear') 
    
    Interpolate temperature

    >>> target_xarray=w.interp_f(src_xarray)

    
    """

    __slots__ = ('name','tgt','method')

    

    def __init__(self, grid, option='linear'):
        
        self.method = option
        ''' str: Interpolation method selected `linear` or `nearest`'''
        # Put here info on grids to be obtained from __call__
        self.name = 'Atmosphere_Interpolator'
        '''str: Name of the Interpolator'''
        if grid == '1x1':
            # Selected regular 1 Degree grid
            lon1x1 = np.linspace(0,359,360)
            lat1x1 = np.linspace(-90,90,180)
            mm=np.ones([lat1x1.shape[0],lon1x1.shape[0]])
            tgt = xr.DataArray(mm,dims=['lat','lon'],\
                            coords={'lat':lat1x1,'lon':lon1x1})  
        elif grid == '025x025':
            homedir = os.path.expanduser("~")
            file = homedir + '/Dropbox (CMCC)/data_zapata/'+ 'masks_CMCC-CM2_VHR4_AGCM.nc'
            dst=xr.open_dataset(file,decode_times=False)
            lat25 = dst.yc.data[:,0]
            lon25 = dst.xc.data[0,:]
            tgt = xr.DataArray(dst.mask,dims=['lat','lon'],\
                            coords={'lat':lat25,'lon':lon25})  
        else:
            SystemError(f'Wrong Option in {self.name} --> {grid}')  
        self.tgt = tgt
        '''xarray: Target grid'''    
        return
        
    def __call__(self):
        return

    def __repr__(self):
        '''  Printing other info '''
        return '\n' 
    
    def interp_scalar(self, xdata):
        ''' 
        Perform interpolation  to the target grid.
        This methods can be used for scalar quantities.

        Parameters
        ----------
        xdata :  xarray
            2D array to be interpolated, it must be on the `src_grid`
        
        Returns
        -------
        out :  xarray
            Interpolated xarray on the target grid
        
        '''
        
        res = xdata.interp_like(self.tgt,method=self.method)
        return res



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
    option : str
        'global', for global grids (to be implemented regional )

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
    
    Interpolate temperature

    >>> target_xarray=w.interp_T(src_xarray)

    Interpolate U,V

    >>> target_xarray=w.interp_UV(U_xarray,V_xarray)

    """

    __slots__ = ('vt','wt','option','mask','masku','maskv','mask_H','lath','lonh','mdir' \
                     ,'tangle')

    def __init__(self, src_grid, tgt_grid, option = 'Global'):
        # Put here info on grids to be obtained from __call__
        # This currently works with mask files
        # 'masks_CMCC-CM2_VHR4_AGCM.nc'
        # and
        # 'ORCA025L50_mesh_mask.nc'
        
        # Path to auxiliary Ocean files
        homedir = os.path.expanduser("~")
        self.mdir = homedir + '/Dropbox (CMCC)/data_zapata'
        ''' Mask Directory'''
        da=xr.open_dataset( self.mdir + '/ORCA025_angle.nc',decode_times=False)
        self.tangle = da.tangle.data*np.pi/180.0
        '''Angles at t-points in the ocean C-grid'''
        # Global grid consider North Pole
        if option == 'Global':
            self.option = 'Global'
            '''Option for region (not implemented)'''

        #Source Grids for T, U, V
        self.mask = src_grid.tmask.data[0,0,...]>0
        ''' T-mask'''
        self.masku = src_grid.umask.data[0,0,...]>0
        ''' U-mask'''
        self.maskv = src_grid.vmask.data[0,0,...]>0
        ''' V-mask'''

        #Fix Polar Fold
        self.mask[-1:,:] = False
        self.mask[-2:,:] = False
        self.mask[-3:,:] = False
   
        latou =  src_grid.gphiu[0,...].data[self.masku].flatten()
        lonou = (src_grid.glamu[0,...].data[self.masku].flatten() + 360) % 360
        
        latov =  src_grid.gphiv[0,...].data[self.maskv].flatten()
        lonov = (src_grid.glamv[0,...].data[self.maskv].flatten() + 360) % 360

        lato =  src_grid.nav_lat.data[self.mask].flatten()
        lono = (src_grid.nav_lon.data[self.mask].flatten() + 360) % 360
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
        #Get coordinates for output
        self.lath=tgt_grid.yc.data[:,0]
        '''Latitudes of the target grid'''
        self.lonh=tgt_grid.xc.data[0,:]
        '''Longitudes of the target grid'''

        #Insert NaN for land
        temp1 = tgt_grid.mask.where(tgt_grid.mask < 1)
        self.mask_H = temp1.stack(ind={'nj','ni'})
        '''Land-sea mask of the target grid'''
        
        _maskt = self.mask_H[~xr.ufuncs.isnan(self.mask_H)] 
        
        #Order target coordinate
        latlon_to=np.asarray([_maskt.yc.data,_maskt.xc.data])
        
        #Generates Weights
        tri = qhull.Delaunay(latlon.T)
        simplex = tri.find_simplex(latlon_to.T)
        self.vt = np.take(tri.simplices, simplex, axis=0)
        '''Weights'''
        _temp = np.take(tri.transform, simplex, axis=0)
        delta = latlon_to.T - _temp[:, 2]
        bary = np.einsum('njk,nk->nj', _temp[:, :2, :], delta)
        self.wt = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))
        '''Weights'''

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
            Interpolated xarray on the target grid
        
        '''
        t_data = xdata.data[self.mask].flatten()
        res = self._interp_distance(t_data)
        return res

    def interp_UV(self, udata, vdata):
        ''' 
        Perform interpolation for U,V Grid point to the target grid.
        This methods can be used for vector quantities.

        The present method interpolates the U,V points to the T points, 
        rotates them and then interpolates to the target grid.

        Parameters
        ----------
        udata,vdata :  xarray
            2D array to be interpolated, it must be on the `src_grid`
        
        Returns
        -------
        out :  xarray
            Interpolated xarray on the target grid

        '''
        

        # Allocate space
        uvelT = np.zeros(udata.data.shape)
        vvelT = np.zeros(vdata.data.shape)

        # Interpolate zonal component from grid U to grid T
        tmpmsk=self.masku[:,:-1]+self.masku[:,1:]
        utmp=udata[:,:-1]*self.masku[:,:-1]+udata[:,1:]*self.masku[:,1:]
         
        uvelT[:,1:]=np.where(self.mask[:,1:]!=0, utmp/tmpmsk, utmp)
        uvelT[:,0]=uvelT[:,-2] # E-W periodicity

        # Interpolate meridional component from grid V to grid T
        tmpmsk=self.maskv[:-1,:]+self.maskv[1:,:]
        vtmp=vdata[:-1,:]*self.maskv[:-1,:]+vdata[1:,:]*self.maskv[1:,:]
        vvelT[1:,:]=np.where(self.mask[1:,:]!=0, vtmp/tmpmsk, vtmp)

        # rotate  velocity according their grid points
        uvel,vvel = self._rotate_UV(uvelT,vvelT)
        #print(f'---uvelT---vvel--- {uvelT}  {uvel} {self.masku.shape} {utmp.shape}')
       
        # Interpolate to the new grid
        u1 = self._interp_distance(uvel[self.mask].flatten())
        v1 = self._interp_distance(vvel[self.mask].flatten())

        return u1,v1

    def _interp_distance(self,_data):
        ''' 
        Internal Routine to perform interpolation
        on T points according a simple distance method
        '''
        
        # Apply sum over weights
        dat= np.expand_dims(_data,axis=1)
        new = np.einsum('nj,nj->n', np.take(dat, self.vt), self.wt)
        #
        temp=self.mask_H.copy()
        temp[~xr.ufuncs.isnan(temp)]  = new
        # The transpose is necessary for the ordering of variables in mask_H
        temp1 = temp.unstack().data
       
        res = xr.DataArray(temp1,dims=('lat','lon'),coords = {'lat':self.lath,'lon': self.lonh})
        res[:,0]=res[:,-1]
        # Polar interpolation still problematic, mask values at the North Pole
        if self.option == 'Global':
            res[-1:-8:-1,:]= -1.78
            
        return res

    def _rotate_UV(self,u,v):
        ''' 
        Internal Routine to perform rotation of vectors fields
        on U,V points priori interpolation to the T points.
        '''
        # 
        uu = u * np.cos(self.tangle) - v * np.sin(self.tangle)
        vv = v * np.cos(self.tangle) + u * np.sin(self.tangle)
        return uu,vv

