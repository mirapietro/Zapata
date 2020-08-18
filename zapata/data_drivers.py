'''
Drivers for data retrieve of specific dataset
=============================================
'''

import os, sys, re
import numpy as np
import xarray as xr
import pandas as pd
import glob


def era5_numpy(dataset, var, level, period):
    '''
    Driver for data retrieve of ERA5 in numpy format
    Read requested data into an xarray DataArray

    Parameters
    ----------
    dataset : dict
        Dataset informative structure
    var : string
         variable name
    level : list
        vertical levels float value
    period : list
        Might be None or a two element list with initial and final year

    Returns
    -------
    out : DataArray
        Output data from dataset

    '''
    from natsort import natsorted
    from zapata.data import get_data_files
  
    out = None

    # get all levels if not specified
    if level is None:
        level = dataset['levels']
 
    for lev in level:

        files = get_data_files(dataset, var, [lev], period)
        
        # use natural sorting of files
        files['files'] = natsorted(files['files'])

        # get lon/lat coordinates
        if dataset['metrics']['lon'][-3:] == 'npy':
            lon = np.load(dataset['path'] + '/' + dataset['metrics']['lon'])
            lat = np.load(dataset['path'] + '/' + dataset['metrics']['lat'])
        else:
            lon = eval(dataset['metrics']['lon'])
            lat = eval(dataset['metrics']['lat'])

        # define time axis
        time_bnd = files['period']
        if dataset['data_freq'] == 'yearly':
            prds = (time_bnd[1] - time_bnd[0] + 1) ; frq = 'YS'
        elif dataset['data_freq'] == 'monthly':
            prds = (time_bnd[1] - time_bnd[0] + 1) * 12 ; frq = 'M'
        elif dataset['data_freq'] == 'daily':
            prds = (time_bnd[1] - time_bnd[0] + 1) * 365 ; frq = 'D'
        time = pd.date_range(str(time_bnd[0])+'-01-01', periods=prds,freq=frq)
        del prds, frq

        # get first file
        ndat = np.load(files['files'][0])

        # append other data
        if len(files['files']) > 1:
           ndat =  np.expand_dims(ndat, axis=0)
           for ff in files['files'][1:]:
              tmp = np.expand_dims(np.load(ff), axis=0)
              ndat = np.append(ndat, tmp, axis=0)
              del tmp

        # create xarray
        if len(lat.shape) > 1:
            tmp = xr.DataArray(ndat, name=files['var'], coords={'time': time, 'latitude': (['lat','lon'], lat),
                          'longitude': (['lat','lon'],lon)}, dims=['time', 'lat', 'lon'])
        else:
            tmp = xr.DataArray(ndat, name=files['var'], coords=[time, lat, lon], dims=['time', 'lat','lon'])

        # add level coordinate
        if lev not in ['SURF',]:
           tmp = tmp.expand_dims(dim=['lev'], axis=1).assign_coords(lev=[lev]) 

        if out is None:
            out = tmp
        else:
            out = xr.concat(out, tmp, dim='lev')
     
    out.attrs['realm'] = files['component']

    return out
