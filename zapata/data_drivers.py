'''
Drivers for data retrieve of specific dataset.

Reading and processing of inputs from dataset is performed by module :meth:`data.py<zapata.data>`

===================================
'''

import os, sys, re
import numpy as np
import xarray as xr
import pandas as pd
import glob


def cglorsv7(dataset, var, level, period, season):
    '''
    Driver for data retrieve of C-GLORS V7 global ocean reanalyses
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
    season : string
        Month ('JAN'), season ('DJF', 'AMJ') or annual ('ANN')

    Returns
    -------
    out : DataArray
        Output data from dataset

    '''
    from zapata.data import get_data_files, fix_coords, mask_data

    out = None

    # get files list to read
    files = get_data_files(dataset, var, level, period)

    # open files as a dataset
    ds = xr.open_mfdataset(files['files'], engine='netcdf4', combine = 'by_coords', coords='minimal', compat='override', parallel=True)
    out = ds[files['var']]
    out.attrs['realm'] = files['component']

    # read and assign 2D coordinates
    dc = xr.open_dataset(files['coords']['file'])
    lon = dc[files['coords']['lon']]
    lat = dc[files['coords']['lat']]

    out = out.assign_coords({"nav_lon":(("y","x"), lon.data)})
    out = out.assign_coords({"nav_lat":(("y","x"), lat.data)})

    # rename dimensions and coordinates
    out = fix_coords(out, files['coord_map'])

    # apply mask to data if provided
    if 'mask' in files.keys():
        out = mask_data(out, files['mask']['name'], files['mask']['file'], files['mask']['coord_map'])

    out.attrs['realm'] = files['component']

    return out


def era5_numpy(dataset, var, level, period, season):
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
    season : string
        Month ('JAN'), season ('DJF', 'AMJ') or annual ('ANN')

    Returns
    -------
    out : DataArray
        Output data from dataset

    '''
    from tqdm import tqdm
    from natsort import natsorted
    from zapata.data import get_data_files
    from zapata.data import define_time_frames
  
    out = None

    if level is None:
        level = dataset['levels']

    if var in ['tp', 'MSL', 'SST']:
        level = ['SURF',]
 
    for lev in level:

        if not isinstance(lev, str):
            lev = int(lev)

        files = get_data_files(dataset, var, [lev], period)

        # use natural sorting of files list
        inp_files = natsorted(files['files'])

        # define selected data time axis
        time_bnd = files['period']
        prds = (time_bnd[1] - time_bnd[0] + 1) * 12 ; frq = 'M'
        timeline = pd.date_range(str(time_bnd[0])+'-01-01', periods=prds,freq=frq)
        del prds, time_bnd 

        # filter by year/season/month if provided
        if season:
            # admissible time frames
            time_frames = define_time_frames(season)
            if len(time_frames[season]) > 1:
                sel_idx = time_frames[season][2]
            else:
                sel_idx = time_frames[season][0]

            sel_data = []
            time = []
            for fid,ff in enumerate(inp_files):
                #get filename month
                ff_month = int(ff.split('_')[-2])
                if ff_month in sel_idx:
                    sel_data.append(ff)
                    time.append(timeline[fid].strftime('%Y-%m-%d'))
            inp_files = sel_data
            time = pd.to_datetime(time)
            del fid, ff, ff_month, sel_data, sel_idx
        else:
            time = timeline

        # get lon/lat coordinates
        if dataset['metrics']['lon'][-3:] == 'npy':
            lon = np.load(dataset['path'] + '/' + dataset['metrics']['lon'])
            lat = np.load(dataset['path'] + '/' + dataset['metrics']['lat'])
        else:
            lon = eval(dataset['metrics']['lon'])
            lat = eval(dataset['metrics']['lat'])

        # get first file
        ndat = np.load(inp_files[0])

        # append other data over time
        if len(inp_files) > 1:
           ndat =  np.expand_dims(ndat, axis=0)
           for ff in tqdm(inp_files[1:]):
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
