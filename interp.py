'''
Interpolation for Ocean and Atmospheric Data
============================================

Routines and methods for interpolation of ocean and atmospheric model
field, either on regular grids or rotated, multi-pole grids.
For the ocean, is set for the CMCC Ocean model at the nominal resolution of 0.25
respectively.

The staggering requires different interpolators operators for scalar points at T-points
and vector quantities carried at (u,v) points. For the moment a simple interpolation is carried out
but a more accurate vector interpolation is under development.

The interpolation is obtained by triangulation of the starting grid and separate interpolation to the new grid.
The weights are preserved and they can be used for repeated application of the same set of grids.

Classes
-------
| **Atmosphere_Interpolator**
| **Ocean_Interpolator**



'''

import os, sys
import math
import numpy as np
import pickle
import gzip

import scipy.linalg as sc
import scipy.special as sp
import scipy.interpolate as spint
import scipy.spatial.qhull as qhull
from scipy.spatial import Delaunay


import zapata.lib as zlib

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


    Notes
    =====

    It is a thin wrapper around `xarray` `interp_like` method.

    Examples
    --------
    Create the weights for interpolation

    >>> w= zint.Atmosphere_Interpolator('1x1','linear')

    Interpolate temperature

    >>> target_xarray=w.interp_f(src_xarray)


    """

    __slots__ = ('name','tgt','choice')



    def __init__(self, grid, option='linear'):
        self.choice = option
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

        res = xdata.interp_like(self.tgt,method=self.choice)
        return res

class Ocean_Interpolator():
    """This class creates weights for interpolation of ocean fields.

    This class creates an interpolation operator from a *source grid*
    `src_grid` to a *target grid* `tgt_grd`. The interpolator then
    can be used to perform the actual interpolation. The model uses
    an Arakawa C-grid, that is shown in the following picture. The f-points
    correspond to the points where the Coriolis terms are carried.

    .. image:: ../resources/NEMOgrid.png
        :scale: 25 %
        :align: right

    The Arakawa C-grid used in the ocean model shows also the ordering
    of the points, indicating which points correspond to the (i,j) index.

    The *source grid* must be a `xarray` `DataSet` containing coordinates
    `latitude` and `longitude`.

    The *target grid* must be a `xarray` `DataArray` with variables `lat` and `lon`

    Border land points at all levels are covered by a convolution value using a
    window that can be changed in `sea_over_land`.

    Works only on single `DataArray`

    Parameters
    ----------

    src_grid : xarray
        Source grid
    tgt_grid : xarray
        Target grid
    level : float
        Depth to generate the interpolator
    window : int
        Window for sea over land
    period : int
        Minimum number of points in the sea-over-land process
    nloops : int
        Number of sea-over-land loops
    verbose: bool
        Lots of output

    Attributes
    ----------

    mask :
        Mask of the source grid
    vt :
        Weights
    wt :
        Weights
    mdir :
        Directory for masks files
    ingrid :
        Input Grid, `src_grid_name`
    outgrid :
        Output Grid `tgt_grid_name`
    window :
        Window for convolution Sea-Over-Land (default 3)
    period :
        Minimum number of points into the window (default 1)
    nloops :
        Number of sea-over-land loops
    empty :
        True if the chosen level in the target grid is all land
    T_only :
        Read only the T grid
    isglobal:
        The input grid is global
    T_lon :
        Longitudes of input T-mask
    T_lat :
        Latitudes of input T-mask
    U_lon :
        Longitudes of input U-mask
    U_lat :
        Latitudes of input U-mask
    V_lon :
        Longitudes of input V-mask
    V_lat :
        Latitudes of input V-mask
    tangle :
        Angles of the T points of the input grid
    mask_reg :
        Mask of the Target grid
    cent_long :
        Central Longitude of the Target grid
    name :
        Name of the Interpolator Object
    level :
        Level of the Interpolator Object


    Methods
    -------

    Interp_T :
        Interpolate Scalar quantities at T points

    Interp_UV :
        Interpolate Vector Velocities at (U,V)

    mask_sea_over_land :
        Mask border point for Sea over land

    fill_sea_over_land :
        Fill U,V values over land

    to_file :
        Writes interpolator object to file (pickled format)

    Examples
    --------
    Create the weights for interpolation

    >>> w = zint.Ocean_Interpolator(src_grid_name, tgt_grid_name)

    Interpolate temperature

    >>> target_xarray = w.interp_T(src_xarray, method='linear')

    Intek': grid.umask, 'vmask': grid.vmaskpolate U,V

    >>> target_xarray = w.interp_UV(U_xarray, V_xarray, method='linear')


    """

    __slots__ = ('mask','masku','maskv','mdir', 'mask_reg', \
                'sea_index','sea_index_U','sea_index_V', \
                'latlon', 'masT_vec', 'maskub','maskvb','masktb',\
                'latlon_reg','sea_index_reg','regmask_vec', \
                'T_lat','T_lon','U_lat','U_lon','V_lat','V_lon',\
                'name','cent_long','tri_sea_T','tri_sea_U','tri_sea_V','tangle',\
                'ingrid','outgrid','level','window','period','nloops','empty','T_only','isglobal',\
                'Umask_reg','Ulatlon_reg','Usea_index_reg','Uregmask_vec',\
                'Vmask_reg','Vlatlon_reg','Vsea_index_reg','Vregmask_vec')

    def __init__(self, src_grid_name, tgt_grid_name, level=1, verbose=False, window=3, period=1,
                 nloops=1, T_only=True, isglobal=True):
        # Put here info on grids to be obtained from __call__
        # This currently works with mask files
        # 'masks_CMCC-CM2_VHR4_AGCM.nc'
        # and
        # 'ORCA025L50_mesh_mask.nc'

        # Path to auxiliary Ocean files
        homedir = os.path.expanduser("~")
        #self.mdir = homedir + '/Dropbox (CMCC)/data_zapata'
        self.mdir = '/work/oda/pm28621/C-GLORS_Zapata/'
        self.ingrid = src_grid_name
        self.outgrid = tgt_grid_name

        # Parameter for sea over land
        self.window = window
        self.period = period
        self.nloops = nloops

        #Resolve grids
        s_in = self._resolve_grid(src_grid_name, level)

        self.empty = False
        if s_in['tmask'].sum() == 0.:
            self.empty = True
            print(f'Level {level} is empty')
            return

        tk = s_in['tmask']
        self.T_lon = s_in['lonT']
        self.T_lat = s_in['latT']
        mask = tk.assign_coords({'lat':self.T_lat,'lon':self.T_lon}) #.drop_vars(['U_lon','U_lat','V_lon','V_lat','T_lon','T_lat']).rename({'z':'deptht'})

        if s_in['umask'] is not None:
            tk = s_in['umask']
            self.U_lon = s_in['lonU']
            self.U_lat = s_in['latU']
            masku = tk.assign_coords({'lat':self.U_lat,'lon':self.U_lon}) #.drop_vars(['U_lon','U_lat','V_lon','V_lat','T_lon','T_lat']).rename({'z':'deptht'})

        if s_in['vmask'] is not None:
            tk = s_in['vmask']
            self.V_lon = s_in['lonV']
            self.V_lat = s_in['latV']
            maskv = tk.assign_coords({'lat':self.V_lat,'lon':self.V_lon}) #.drop_vars(['U_lon','U_lat','V_lon','V_lat','T_lon','T_lat']).rename({'z':'deptht'})

        self.name = 'UV  Velocity'
        self.level = str(level)

        # Fix Polar Fold
        if isglobal:
            mask[-3:,:] = False

        # T angles
        self.tangle = s_in['tangle']

        # apply seaoverland over source grid(s)
        self.mask = self.mask_sea_over_land(mask)
        if s_in['umask'] is not None:
            self.masku = self.mask_sea_over_land(masku)
        if s_in['vmask'] is not None:
            self.maskv = self.mask_sea_over_land(maskv)

        print(f' Generating interpolator for {self.name}')
        if verbose:
            print(self.mask, self.masku, self.maskv)

        # Get triangulation for all grids
        self.latlon, self.sea_index, self.masT_vec = get_sea(self.mask, level)
        self.tri_sea_T  = Delaunay(self.latlon)
        print(f' computed the triangulation for T grid')

        if s_in['umask'] is not None:
            latlon_U, self.sea_index_U, masU_vec = get_sea(self.masku, level)
            self.tri_sea_U  = Delaunay(latlon_U)
            print(f' computed the triangulation for U grid')

        if s_in['vmask'] is not None:
            latlon_V, self.sea_index_V, masT_vec = get_sea(self.maskv, level)
            self.tri_sea_V  = Delaunay(latlon_V)
            print(f' computed the triangulation for V grid')

        # Target (T) Grid
        s_out = self._resolve_grid(tgt_grid_name, level, verbose=verbose)
        self.mask_reg = s_out['tmask']
        self.cent_long = s_out['cent_long']
        self.latlon_reg, self.sea_index_reg, self.regmask_vec = get_sea(self.mask_reg, level)

        if not T_only:
            # Target (U) Grid
            self.Umask_reg = s_out['umask']
            self.Ulatlon_reg, self.Usea_index_reg, self.Uregmask_vec = get_sea(self.Umask_reg, level)
            # Target (V) Grid
            self.Vmask_reg = s_out['vmask']
            self.Vlatlon_reg, self.Vsea_index_reg, self.Vregmask_vec = get_sea(self.Vmask_reg, level)


    def __call__(self):
        print(f' Interpolator for T, U, V GLORS data ')
        print(f' This is for level at depth {self.level} m')
        print(f' Main methods interp_T and interp_UV')
        return


    def __repr__(self):
        '''  Printing other info '''
        print(f' Interpolator for T, U, V GLORS data ')
        print(f' This is for level at depth {self.level} m')
        return '\n'


    def interp_T(self, xdata, method='linear', grd='T'):
        '''
        Perform interpolation for T Grid point to the target grid.
        This methods can be used for scalar quantities.

        Parameters
        ----------
        xdata :  xarray
            2D array to be interpolated, it must be on the `src_grid`

        method : str
            Method for interpolation
                * 'linear'  , Use linear interpolation
                * 'nearest' , use nearest interpolation

        grd : str
            can be either T (default), U or V, it defines the target grid

        Returns
        -------
        out :  xarray
            Interpolated xarray on the target grid
        '''
        regmask_vec = self.Uregmask_vec if grd=='U' else self.Vregmask_vec if grd=='V' else self.regmask_vec
        latlon_reg = self.Ulatlon_reg if grd=='U' else self.Vlatlon_reg if grd=='V' else self.latlon_reg
        sea_index_reg = self.Usea_index_reg if grd=='U' else self.Vsea_index_reg if grd=='V' else self.sea_index_reg

        # Fill T values over land
        xdata = self.fill_sea_over_land(xdata)

        # stack the input and keep only the sea
        Tstack = xdata.stack(ind=('y','x'))
        sea_T = Tstack[self.sea_index]
        # prepare the (stacked) output
        temp = xr.full_like(regmask_vec, np.nan)

        # define the interpolator
        if method == 'linear':
            interpolator = spint.LinearNDInterpolator(self.tri_sea_T, sea_T)
        elif method == 'nearest':
            interpolator = spint.NearestNDInterpolator(self.tri_sea_T, sea_T)
        else:
            SystemError(f'Error in interp_T , wrong method  {method}')
        # Interpolate
        T_reg = interpolator(latlon_reg)
        # save the output
        temp[sea_index_reg] = T_reg.data
        out = temp.unstack()

        # Fix dateline problem
        if self.outgrid == 'L44_025_REG_GLO':
            delx=0.25
            ddelx=3*delx
            out[:,1439] = out[:,1438] + delx*(out[:,1]-out[:,1438])/ddelx
            out[:,0] = out[:,1438] + 2*delx*(out[:,1]-out[:,1438])/ddelx

        return out


    def interp_UV(self, udata, vdata, method = 'linear'):
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

        # Insert NaN
        udata = xr.where(udata < 200, udata, np.nan)
        vdata = xr.where(vdata < 200, vdata, np.nan)

        # Fill U, V values over land
        udata = self.fill_sea_over_land(udata)
        vdata = self.fill_sea_over_land(vdata)

        # Compute interpolation for U,V
        Ustack = udata.stack(ind=('y','x'))
        Vstack = vdata.stack(ind=('y','x'))
        sea_U = Ustack[self.sea_index_U]
        sea_V = Vstack[self.sea_index_V]

        # Interpolate to T_grid
        if method == 'linear':
            int_U = spint.LinearNDInterpolator(self.tri_sea_U, sea_U)
            int_V = spint.LinearNDInterpolator(self.tri_sea_V, sea_V)
        elif method == 'nearest':
            int_U = spint.NearestNDInterpolator(self.tri_sea_U, sea_U)
            int_V = spint.NearestNDInterpolator(self.tri_sea_V, sea_V)
        else:
            SystemError(f'Error in interp_UV , wrong method  {method}')

        U_on_T = int_U(self.latlon)
        V_on_T = int_V(self.latlon)

        UT = self.masT_vec.copy()
        VT = self.masT_vec.copy()
        UT[self.sea_index] = U_on_T.data
        VT[self.sea_index] = V_on_T.data

        UT = UT.unstack()
        VT = VT.unstack()

        # Rotate Velocities
        fac = self.tangle*math.pi/180.
        uu = (UT * np.cos(fac) - VT * np.sin(fac)).drop_vars(['nav_lon','nav_lat'])
        vv = (VT * np.cos(fac) + UT * np.sin(fac)).drop_vars(['nav_lon','nav_lat'])

        # Interpolate to regular grid
        Uf = self.interp_T(uu, method=method, grd='U')
        Vf = self.interp_T(vv, method=method, grd='V')

        # Fix dateline problem
        if self.outgrid == 'L44_025_REG_GLO':
            delx=0.25
            ddelx=3*delx
            Uf[:,1439] = Uf[:,1438] + delx*(Uf[:,1]-Uf[:,1438])/ddelx
            Uf[:,0] = Uf[:,1438] + 2*delx*(Uf[:,1]-Uf[:,1438])/ddelx

        return Uf,Vf


    def to_file(self, filename):
        '''
        This method writes to file the interpolating object in
        pickled format
        '''

        with gzip.open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)


    def mask_sea_over_land(self, mask):
        '''
        Apply `Sea over land` to a mask.

        Sea over land is obtained by forward and backward filling NaN land
        value with an arbitrary value.

        Parameters
        ==========
        mask:
            original mask

        Returns
        =======
        masknan:
            seaoverlanded mask

        '''
        # np.nan where the mask=0, 1 elsewhere
        masknan = xr.where(mask != 0, 1, np.nan)
        # number of iterations of seaoverland
        for loop in range(self.nloops):
            # initialize the tuple to store the shifts
            mask_shift = ()
            # shift in all 8 directions
            for x in range(-1, 2):
                for y in range(-1, 2):
                    if ((x != 0) | (y != 0)): # skip the no-shifting
                        # store the shifting in the tuple
                        mask_shift = mask_shift + (masknan.shift({'x':x, 'y':y}),)
            # take the max over 'shift' (either 1 or np.nan)
            masknan = xr.concat(mask_shift, dim='shift').max(dim='shift')

        return masknan


    def _resolve_grid(self, ingrid, level=None, verbose=False):
        '''
        Internal routine to resolve grid informations
        '''

        if ingrid == 'L75_025_TRP_GLO':
            print(f' Tripolar L75 0.25 Grid -- {ingrid}')
            grid = xr.open_dataset(self.mdir + '/L75_025_TRP_GLO/tmask_UVT_latlon_coordinates.nc')
            grid = grid.sel(z=level)
            geo = xr.open_dataset(self.mdir + '/L75_025_TRP_GLO/NEMO_coordinates.nc')
            angle = xr.open_dataset(self.mdir + '/L75_025_TRP_GLO/ORCA025L75_angle.nc')
            struct={'tmask': grid.tmask, 'umask': grid.umask,'vmask': grid.vmask, 'tangle': angle.tangle, \
                    'lonT': geo.glamt,'latT':geo.gphit,'lonU':geo.glamu, \
                    'latU':geo.gphiu,'lonV':geo.glamv,'latV':geo.gphiv  }

        elif ingrid == 'L44_025_TRP_GLO':
            print(f' Tripolar L44 0.25 Grid -- {ingrid}')
            grid = xr.open_dataset(self.mdir + '/L44_025_TRP_GLO/tmask44_UVT_latlon_coordinates.nc')
            grid = grid.sel(z=level)
            angle = xr.open_dataset(self.mdir + '/L75_025_TRP_GLO/ORCA025L75_angle.nc')
            geo = xr.open_dataset(self.mdir + '/L75_025_TRP_GLO/NEMO_coordinates.nc')
            struct={'tmask': grid.tmask, 'umask': grid.umask,'vmask': grid.vmask, 'tangle': angle.tangle, \
                    'lonT': geo.glamt,'latT':geo.gphit,'lonU':geo.glamu, \
                    'latU':geo.gphiu,'lonV':geo.glamv,'latV':geo.gphiv  }

        elif ingrid == 'L50_025_TRP_GLO':
            print(f' Tripolar L50 0.25 Lat-Lon Grid -- {ingrid}')
            grid = xr.open_dataset('/work/oda/pm28621/data/Reanalysis/CGLORS/mesh_mask.nc')
            # Take only a slice of it
            grid = grid.isel(x=slice(1000,1350),y=slice(600,750))
            grid = grid.isel(z=level, t=0)
            # remove the Red Sea
            grid = grid.where(np.logical_or(grid.y > 27, grid.x < 280), 0.)
            # remove the Black Sea
            grid = grid.where(np.logical_or(grid.y < 83, grid.x < 257), 0.)
            angle = xr.open_dataset('/work/oda/pm28621/data/Reanalysis/CGLORS/ORCA025L75_angle.nc')\
                        .isel(x=slice(1000,1350),y=slice(629,779))
            struct={'tmask': grid.tmask, 'umask': grid.umask,'vmask': grid.vmask,
                    'tangle': angle.tangle, 'lonT': grid.glamt,'latT': grid.gphit,'lonU':grid.glamu,
                    'latU': grid.gphiu,'lonV': grid.glamv,'latV': grid.gphiv  }

        elif ingrid == 'L50_1o24_BDY_MED':
            print(f' LOBC L50 1/24 Lat-Lon Grid (NOT MASKED) -- {ingrid}')
            grid = xr.open_dataset('/data/oda/pm28621/med_LOBC_out/grids/tmask50_UVT_latlon_coordinates.nc')\
                        .rename({'T_lat':'lat','T_lon':'lon'})
            # take only the boundary (T), do not mask
            nc_bndT_tmp = xr.open_dataset('/data/oda/pm28621/med_LOBC_out/grids/bndT_2D.nc')
            Tgrid = grid['lat'].isel(x=nc_bndT_tmp.nbidta-1, y=nc_bndT_tmp.nbjdta-1) > 0.
            # take only the boundary (U), do not mask
            nc_bndU_tmp = xr.open_dataset('/data/oda/pm28621/med_LOBC_out/grids/bndU_2D.nc')
            Ugrid = grid['lat'].isel(x=nc_bndU_tmp.nbidta-1, y=nc_bndU_tmp.nbjdta-1) > 0.
            # take only the boundary (V), do not mask
            nc_bndV_tmp = xr.open_dataset('/data/oda/pm28621/med_LOBC_out/grids/bndV_2D.nc')
            Vgrid = grid['lat'].isel(x=nc_bndV_tmp.nbidta-1, y=nc_bndV_tmp.nbjdta-1) > 0.
            # prepare the struct
            cent_long = 720
            struct={'tmask': Tgrid, 'tangle': None, 'cent_long': cent_long,
                    'umask': Ugrid, 'vmask': Vgrid}

        elif ingrid == 'L50_1o24_REG_MED':
            print(f' Regular L50 1/24 Lat-Lon Grid -- {ingrid}')
            grid = xr.open_dataset('/data/oda/pm28621/med_LOBC_out/grids/tmask50_UVT_latlon_coordinates.nc')\
                    .rename({'T_lat':'lat','T_lon':'lon'})
            grid = grid.isel(z=level)
            cent_long = 720
            struct={'tmask': grid.tmask, 'tangle': None, 'cent_long': cent_long,
                    'umask': grid.umask, 'vmask': grid.vmask}

        # this is a source grid
        elif ingrid == 'L102_025_WOA':
            print(f' Regular L102 0.25 Lat-Lon Grid -- {ingrid}')
            grid = xr.open_dataset('/users_home/oda/pm28621/newREA_IC/grids/WOA_meshmask.nc')
            grid = grid.isel(x=slice(630,867), y=slice(475,550))
            grid = grid.isel(z=level)
            struct={'tmask': grid.tmask, 'umask': None,'vmask': None,
                    'tangle': None, 'lonT': grid.lon2d,'latT': grid.lat2d,'lonU':None,
                    'latU': None, 'lonV': None,'latV': None }
        # this is a target grid (for interpolation of WOA over SDN MED)
        elif ingrid == 'L102_0125_SDN_MED':
            print(f' Regular L102 0.125 Lat-Lon Grid -- {ingrid}')
            grid = xr.open_dataset('/users_home/oda/pm28621/newREA_IC/grids/hybrid_grids/REGSDN_0125h_102v.nc')\
                    .rename({'T_lat':'lat','T_lon':'lon'})
            # do not mask (this is an intermediate grid)
            Tgrid = grid['lat'] > 0.
            struct={'tmask': Tgrid, 'umask': None,'vmask': None,
                    'tangle': None, 'cent_long':None}
        # this is a target grid (for interpolation of WOA over SDN NA)
        elif ingrid == 'L102_025_SDN_NA':
            print(f' Regular L102 0.125 Lat-Lon Grid -- {ingrid}')
            grid = xr.open_dataset('/users_home/oda/pm28621/newREA_IC/grids/hybrid_grids/REGSDN_NA_025h_102v.nc')\
                    .rename({'T_lat':'lat','T_lon':'lon'})
            # do not mask (this is an intermediate grid)
            Tgrid = grid['lat'] > 0.
            struct={'tmask': Tgrid, 'umask': None,'vmask': None,
                    'tangle': None, 'cent_long':None}
        # this is a target grid (for interpolation of WOA over MFS NEMO)
        elif ingrid == 'L102_1o24_REG_MED':
            print(f' Regular L102 1/24 Lat-Lon Grid -- {ingrid}')
            grid = xr.open_dataset('/users_home/oda/pm28621/newREA_IC/grids/tmask102_T_latlon_coordinates.nc')\
                        .rename({'T_lat':'lat','T_lon':'lon'})
            # do not mask (this is an intermediate grid)
            Tgrid = grid['lat'] > 0.
            cent_long = 720
            struct={'tmask': Tgrid, 'tangle': None, 'cent_long': cent_long,
                    'umask': None, 'vmask': None}

        # this is a source grid
        elif ingrid == 'L107_025_SDN_NA_v1':
            print(f' Regular L107 0.25 Lat-Lon Grid -- {ingrid}')
            grid = xr.open_dataset('/users_home/oda/pm28621/newREA_IC/grids/SDN_NA_v1_meshmask.nc')
            grid = grid.isel(y=slice(50,170), x=slice(240,-1))
            grid = grid.isel(z=level)
            struct={'tmask': grid.tmask, 'umask': None,'vmask': None,
                    'tangle': None, 'lonT': grid.lon2d,'latT': grid.lat2d,'lonU':None,
                    'latU': None, 'lonV': None,'latV': None }
        # this is a target grid (for interpolation of SDN_NA over WOA)
        elif ingrid == 'L107_025_WOA':
            print(f' Regular L107 0.25 Lat-Lon Grid -- {ingrid}')
            grid = xr.open_dataset('/users_home/oda/pm28621/newREA_IC/grids/hybrid_grids/REGWOA_025h_107v.nc')\
                    .rename({'T_lat':'lat','T_lon':'lon'})
            # do not mask (this is an intermediate grid)
            Tgrid = grid['lat'] > 0.
            struct={'tmask': Tgrid, 'umask': None,'vmask': None,
                    'tangle': None, 'cent_long':None}
        # this is a target grid (for interpolation of SDN_NA over MFS NEMO)
        elif ingrid == 'L107_1o24_REG_MED':
            print(f' Regular L107 1/24 Lat-Lon Grid -- {ingrid}')
            grid = xr.open_dataset('/users_home/oda/pm28621/newREA_IC/grids/tmask107_T_latlon_coordinates.nc')\
                        .rename({'T_lat':'lat','T_lon':'lon'})
            # do not mask (this is an intermediate grid)
            Tgrid = grid['lat'] > 0.
            cent_long = 720
            struct={'tmask': Tgrid, 'tangle': None, 'cent_long': cent_long,
                    'umask': None, 'vmask': None}

        # this is a source grid
        elif ingrid == 'L92_0125_SDN_MED':
            print(f' Regular L92 0.25 Lat-Lon Grid -- {ingrid}')
            grid = xr.open_dataset('/users_home/oda/pm28621/newREA_IC/grids/SDN_MED_meshmask.nc')
            grid = grid.isel(z=level)
            struct={'tmask': grid.tmask, 'umask': None,'vmask': None,
                    'tangle': None, 'lonT': grid.lon2d,'latT': grid.lat2d,'lonU':None,
                    'latU': None, 'lonV': None,'latV': None }
        # this is a target grid (for interpolation of SDN_MED over WOA)
        elif ingrid == 'L92_025_WOA':
            print(f' Regular L92 0.25 Lat-Lon Grid -- {ingrid}')
            grid = xr.open_dataset('/users_home/oda/pm28621/newREA_IC/grids/hybrid_grids/REGWOA_025h_92v.nc')\
                    .rename({'T_lat':'lat','T_lon':'lon'})
            # do not mask (this is an intermediate grid)
            Tgrid = grid['lat'] > 0.
            struct={'tmask': Tgrid, 'umask': None,'vmask': None,
                    'tangle': None, 'cent_long':None}
        # this is a target grid (for interpolation of SDN_MED over MFS NEMO)
        elif ingrid == 'L92_1o24_REG_MED':
            print(f' Regular L92 1/24 Lat-Lon Grid -- {ingrid}')
            grid = xr.open_dataset('/users_home/oda/pm28621/newREA_IC/grids/tmask92_T_latlon_coordinates.nc')\
                    .rename({'T_lat':'lat','T_lon':'lon'})
            # do not mask (this is an intermediate grid)
            Tgrid = grid['lat'] > 0.
            cent_long = 720
            struct={'tmask': Tgrid, 'tangle': None, 'cent_long': cent_long,
                    'umask': None, 'vmask': None}

        elif ingrid == 'L75_025_REG_GLO':
            print(f' Regular L75 0.25 Lat-Lon Grid -- {ingrid}')
            grid = xr.open_dataset(self.mdir + '/L75_025_REG_GLO/GLO-MFC_001_025_mask_bathy.nc').sel(z=level). \
                rename({'longitude':'lon','latitude':'lat'})
            struct={'tmask': grid.mask, 'tangle': None, 'cent_long': 720}

        elif ingrid == 'L44_025_REG_GLO':
            print(f' Regular L44 0.25 Lat-Lon Grid from WOA -- {ingrid}')
            grid = xr.open_dataset(self.mdir + '/WOA/m025x025L44.nc')
            grid = grid.sel(depth=level)
            cent_long = 720
            struct={'tmask': grid.m025x025L44, 'tangle': None ,'cent_long' : cent_long}

        elif ingrid == 'L50_025_TRP_GLO_Elisa':
            print(f' Tripolar L50 0.25 Lat-Lon Grid (cropped) -- {ingrid}')
            grid = xr.open_dataset('/work/oda/pm28621/data/Reanalysis/CGLORS/mesh_mask.nc')
            # Take only a slice of it
            grid = grid.isel(x=slice(1064,1064+233),y=slice(622,622+140))
            grid = grid.isel(z=level, t=0)
            angle = xr.open_dataset('/work/oda/pm28621/data/Reanalysis/CGLORS/ORCA025L75_angle.nc')\
                        .isel(x=slice(1064,1064+233),y=slice(622,622+140))
            struct={'tmask': grid.tmask, 'umask': grid.umask,'vmask': grid.vmask,
                    'tangle': angle.tangle, 'lonT': grid.glamt,'latT': grid.gphit,'lonU':grid.glamu,
                    'latU': grid.gphiu,'lonV': grid.glamv,'latV': grid.gphiv  }

        elif ingrid == 'L50_025_REG_MED_Elisa':
            print(f' Regular L50 0.25 Lat-Lon Grid (cropped) -- {ingrid}')
            grid = xr.open_dataset('/data/oda/pm28621/Reanalysis/CGLORSv5/outputs//cglors_meshmask/remap_MED_meshmask.nc')
            grid = grid.isel(z=level, time=0)
            cent_long = 720
            struct={'tmask': grid.tmask, 'tangle': None, 'cent_long': cent_long,
                    'umask': grid.tmask, 'vmask': grid.tmask}

        else:
             SystemError(f'Wrong Option in _resolve_grid --> {ingrid}')

        if verbose:
            print(f'Elements of {ingrid} extracted \n ')
            for i in struct.keys():
              print(f' {i} \n')

        return struct


    def fill_sea_over_land(self, var_in,
                           xdim='x', ydim='y',
                           ismax = False):
        '''
        Put values Sea over land.

        The nan points are filled with the average (or max) value
        of the shifted versions (8 possible directions) of `var_in`.
        The `nloops` values controls the number of cycles.

        `nloops` is assigned as attribute of the interpolator.

        Parameters
        ==========
        var_in:
            Field to be treated
        xdim:
            Name of the x dimension
        ydim:
            Name of the y dimension
        ismax:
            Take either the mean (ismax=False) or the max (=True) over 'shift'

        Returns
        =======
        var_out:
            Filled array
        '''
        var_out = var_in.copy()

        for loop in range(self.nloops):
            # Initialize the tuple to store the shifts
            var_shift = ()
            # Shift in all 8 directions
            for x in range(-1, 2):
                for y in range(-1, 2):
                    if ((x != 0) | (y != 0)): # Skip the no-shifting
                        # Store the shifting in the tuple
                        var_shift = var_shift + (var_out.shift({xdim:x, ydim:y}),)
            # Take either the mean or the max over 'shift'
            if ismax:
                var_mean = xr.concat(var_shift, dim='shift').max(dim='shift')
            else:
                var_mean = xr.concat(var_shift, dim='shift').mean(dim='shift')
            # Replace input masked points (nan values) with new ones
            # from the mean (or max) matrix
            var_out = var_out.where(~np.isnan(var_out), other=var_mean)
            # Nothing more to flood
            if np.sum(np.isnan(var_out)) == 0:
                print('WARNING. Field does not have anymore land points,', str(loop + 1),
                      'steps were sufficient to flood it completely.', file=sys.stderr)
                break

        return var_out


def get_sea(maskT, level):
    '''
    Obtain indexed coordinates for sea points.
    It works for 3D and 2D (fixed level) masks.
    '''

    # Try Interpolation
    tm = zlib.putna(-0.1, 0.1, maskT)
    sea_index = ~np.isnan(tm).stack(ind=maskT.dims)

    maskT_vec = tm.stack(ind=maskT.dims)
    land_point = maskT_vec[~sea_index]
    sea_point = maskT_vec[sea_index]
    land_point.name = 'Land'
    sea_point.name = 'Sea'

    # Compute triangularization for sea points (only if 2D)
    latlon = np.asarray([sea_point.lat.data, sea_point.lon.data]).T

    return latlon, sea_index, maskT_vec


def from_file(file):
    '''
    Read interpolator object from `file`
    '''

    with gzip.open(file, 'rb') as input:
        w = pickle.load(input)
    return w


