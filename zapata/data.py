'''
Data retrieve and resampling module
===================================
'''

import os, sys, re
import numpy as np
import xarray as xr
import pandas as pd
import zapata.lib as lib
import netCDF4 as net
import yaml, glob


# NCEP Reanalysis standard pressure levels
# 1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10
# 1000  925  850  700  600, 500  400  300, 250, 200, 150, 100      50          10
#


def read_xarray(dataset=None, var=None, period=None, level=None, season=None, region=None, verbose=False):
    '''
    Load into a DataArray the requested variable from dataset source.

    Parameters
    ----------
    dataset : string
        Name of dataset
    var : string
         variable name
    period : list
        A two element list with initial and final years
    level : float
        level value (if not in levels list the closest one will be used)
    season : string
        Month ('JAN'), season ('DJF', 'AMJ') or annual ('ANN')
    region : list
        Region corners [LonMax, LonMin, LatMax, LatMin]
    verbose : Boolean
        True/False -- Tons of Output

    Returns
    -------
    out : DataArray
        extracted data

    Examples
    --------

    >>> da = read_xarray(dataset='ERA5_MM',var='T',period=[2000 2010], level='500')
    >>> da = read_xarray(dataset='C-GLORSv7', var='votemper', period=[2000 2010], season='DJF')
    '''
    datacat = inquire_catalogue(dataset)

    files = get_data_files(datacat, var, level, period)
    
    out = load_dataarray(datacat, files)
 
    # temporal sampling
    if season is not None:
        out = da_time_mean(out, season)

    # horizontal sampling (need test)
    if region is not None:
        out = out.sel(lon = slice(region[0],region[1]), lat = slice(region[2],region[3]))

    # vertical sampling
    if not files['islevel'] and level is not None:

        # find closest level if not in levels list
        if level not in datacat['levels']:
            idx = np.abs(out.lev - level).argmin()
            level = out.lev[idx]

        out = out.sel(lev = level)

    return out


def da_time_mean(da, sample):
    '''
    Sample datarray based on month/season and compute average over timewindows

    Parameters
    ----------
    da : DataArray
        Input data
    sample : string
        Identifier of temporal sampling (e.g., JAN, FEB, ...,  ANN, DJF, MAM ...)

    Returns
    -------
    out : DataArray
        Time sampled DataArrray

    Examples
    --------

    >>> da = da_time_mean(da, 'JJA')
    >>> da = da_time_mean(da, 'ANN')
    '''
    indexes = None

    # admissible time groups
    time_grp ={'DJF': ['Q-NOV', [0, 4]], 'MAM':['Q-NOV', [1, 4]], 'JJA':['Q-NOV', [2, 4]], 'SON':['Q-NOV', [3, 4]],
        'JFM':['Q-DEC', [0, 4]], 'AMJ':['Q-DEC', [1, 4]], 'JAS':['Q-DEC', [2, 4]], 'OND':['Q-DEC', [3, 4]],
        'ANN':['A', [0, 1]], 
        'JAN':[1,], 'FEB':[2,], 'MAR': [3,], 'APR':[4,], 'MAY':[5,], 'JUN':[6,],
        'JUL':[7,], 'AUG':[8,], 'SEP': [9,], 'OCT':[10,], 'NOV':[11,], 'DEC':[12,]}

    # reduce data to months
    da = da.resample(time='M').mean(dim='time')

    if sample in time_grp.keys():
        if len(time_grp[sample]) > 1:
            weights = subyear_weights(da.time, time_grp[sample][0])
            da = (da * weights).resample(time=time_grp[sample][0]).sum(dim='time')
            idx = time_grp[sample][1]
            da = da.isel(time=slice(idx[0], None, idx[1]))
            da.attrs.update({'time_resample':sample})
        else:
            months = ( da.time.dt.month == time_grp[sample])
            da = da.sel(time=months)
            da.attrs.update({'time_resample':sample})

    else:
        print('requested temporal sampling' + sample + ' is not in admissible time groups.')
        sys.exit(1)

    return da


def subyear_weights(time, freq):
    '''
    Compute month weights according to frequency anchored offsets (season/year)

    Parameters
    ----------
    time : DataArray
        Time array of input data
    freq : string
        Identifier of pandas temporal anchored offsets (e.g., A, Q-DEC)

    Returns
    -------
    weights : DataArray
        weights to be applied at data before subyear time averaging

    '''

    weights = None 
    months = time.dt.days_in_month
    monbyfreq = pd.PeriodIndex(time.data,  freq=freq)

    cummon = months * 0. 
    for ii in monbyfreq.unique():
        idx = (monbyfreq == ii)
        asum = months.sel(time=idx).sum(dim='time').data
        cummon.data[idx] = asum
    
    weights = months / cummon

    return weights


def load_dataarray(dataset, files):
    '''
    Read requested data into an xarray DataArray according to dataset data_format

    Parameters
    ----------
    dataset : dict
        Dataset informative structure
    files : dict
        Dataset input files, variable name, requested period

    Returns
    -------
    out : DataArray
        Output data from dataset

    '''

    if dataset['data_format'] == 'netcdf':
        # open files as a dataset
        ds = xr.open_mfdataset(files['files'], engine='netcdf4', combine = 'by_coords', coords='minimal', compat='override')
        out = ds[files['var']]

    elif dataset['data_format'] == 'numpy':
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
           del tmp, ff

        # create xarray
        if len(lat.shape) > 1:
            out = xr.DataArray(ndat, name=files['var'], coords={'time': time, 'latitude': (['lat','lon'], lat),
                          'longitude': (['lat','lon'],lon)}, dims=['time', 'lat', 'lon'])
        else:
            out = xr.DataArray(ndat, name=files['var'], coords=[time, lat, lon], dims=['time', 'lat','lon'])
 
    else:
        print('Cannot handle data format ' + dataset['data_format'])
        sys.exit(1)

    # rename dimensions and coordinates if mapping provided
    if 'coord_map' in files.keys():
        for coord in files['coord_map'].keys():
            if files['coord_map'][coord][0] in out.coords.keys():
                out = out.rename({files['coord_map'][coord][0]:coord})
                # if dim name is different from coord a second value is given
                if len(files['coord_map'][coord])>1:
                    out = out.rename({files['coord_map'][coord][1]:coord})

    return out


def get_data_files(dataset, var, level, period):
    '''
    Retrieve list of input files from requested dataset and variable.

    Parameters
    ----------
    dataset : dict
        Dataset informative structure
    var : string
         variable name
    level : float
        vertical levels float value
    period : list
        Might be None or a two element list with initial and final year

    Returns
    -------
    files: dict
        Dataset input files, variable name, requested period

    '''
    if dataset is None:
        print('No dataset provided.')
        sys.exit(1)

    var_info = dataset_request_var(dataset, var, level, period)

    # compose list of files
    datapath = dataset['path']
    datatree = dataset['subtree']
    filename = dataset['components'][var_info[0]]['filename']

    if period is None:
        period = dataset['year_bounds']

    if datatree is not None:
        # check if variable is arranged by levels
        islevel = True if re.search('<lev>',datatree) else False
        # check if year in subtree
        subyear = True if re.search('year',datatree) else False
        #TODO do we need to handle months in subtree?
        if re.search('mon',datatree):
            print('Cannot handle dataset subtree with months')
            sys.exit(1)
    else:
        islevel = False
        subyear = False
        datatree = ''

    nameyear = True if re.search('year',filename) else False
        
    # standard set of wildcards
    wildcards={'var':var, 'lev':str(level), 'mon':'*', 'comp':var_info[0], 'data_stream':var_info[1]}
    for ii in wildcards.keys():
        datatree = datatree.replace('<' + ii +'>',wildcards[ii])
        filename = filename.replace('<' + ii +'>',wildcards[ii])
    
    # compose files list
    in_files=[]
    for yy in np.arange(period[0], period[1]+1):
        thispath = '/'.join([datapath, datatree])
        if subyear:
            thispath = thispath.replace('<year>',str(yy))
        thisname = filename
        if nameyear:
            thisname = thisname.replace('<year>',str(yy))
        tmpfile = sorted(glob.glob('/'.join([thispath, thisname])))
        in_files.extend(tmpfile)
        del thispath, thisname, tmpfile
    
    if not in_files:
        print('Input files not found for ' + dataset['name'] + ' located in ' + datapath)
        sys.exit(1)

    # create output dictionary
    files={}
    files['files'] = in_files
    files['var'] = var
    files['period'] = period
    files['islevel'] = islevel

    # coordinate mapping
    data_stream = dataset['components'][var_info[0]]['data_stream'][var_info[1]]
    if 'coord_map' in data_stream.keys():
        files['coord_map'] = data_stream['coord_map']

    return files


def dataset_request_var(dataset, var, level, period):
    '''
    Perform consistency control on user dataset requests and find matching variable.

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
    var_info : list
        requested variable information: [component, data stream, type]

    '''
    # check for level bounds
    level_bnd = [min(dataset['levels']), max(dataset['levels'])]
    if level is not None:
        for lev in level:
            if lev < level_bnd[0] or lev > level_bnd[1]:
                print('Requested level ' + str(lev) + ' is not within dataset bounds [%s, %s]' % tuple(level_bnd))
                sys.exit(1)

    # check for time bounds
    time_bnd = dataset['year_bounds']
    if period is not None and len(time_bnd) > 1:
           if period[0] < time_bnd[0] or period[1] > time_bnd[1]:
               print('Requested time period is not within dataset bounds [%s, %s]' % tuple(time_bnd))
               sys.exit(1)

    # find matching variable
    var_match=[]
    for cc in dataset['components'].keys():
        for dd in dataset['components'][cc]['data_stream'].keys():
             for xy in dataset['components'][cc]['data_stream'][dd].keys():
                 thevars = dataset['components'][cc]['data_stream'][dd][xy]
                 if var in thevars:
                     var_match.append([cc, dd, xy])
    del cc, dd, xy, thevars

    if len(var_match) > 1:
        print('Requested variable ' + var + ' is available from multiple data streams. Something is wrong.')
        sys.exit(1)
    elif len(var_match) == 0:
        print('Requested variable ' + var + ' is not available in the dataset %s', dataset['name'])
        sys.exit(1)
    else:
        print('Retrieve variable ' + var + ' from component %s of data stream %s as %s field' % tuple(var_match[0]))
        var_info = var_match[0]


    return var_info


def inquire_catalogue(dataset=None):
    '''
    Retrieve requested dataset informative structure from general catalogue (YAML file).

    If no dataset is requested, print a list of available datasets name and description.

    Parameters
    ----------
    dataset : string
        Name of data set

    Returns
    -------
    out : dict
        requested dataset informative structure

    Examples
    --------

    >>> datacat = inquire_catalogue(dataset='C-GLORSv7')
    >>> inquire_catalogue()
        List of datasets in catalogue:
        C-GLORSv7 : Ocean Global Reanalyses at 1/4° resolution monthly means
        ERA5_MM : ERA5 Monthly Mean on Pressure Levels
    '''
    out = None
    pwd = os.path.dirname(os.path.abspath(__file__))

    # Load catalogue
    catalogue = yaml.load(open(pwd + '/catalogue.yml'), Loader=yaml.FullLoader)

    # Print list of available datasets    
    if dataset is None:
        print('List of datasets in catalogue:\n')
        for cat in catalogue.keys():
            print('%s : %s' % tuple([cat, catalogue[cat]['description']]))
        print('\n')
        return

    # retrieve dataset information
    if dataset not in catalogue.keys():
        print('Requested dataset ' + dataset + ' is not available in catalogue.')
        sys.exit(1)
    else:
        print('Access dataset ' + dataset + '\n')
        out = catalogue[dataset]
        out['name'] = dataset

    return out


def read_month(dataset, vardir,var1,level,yy,mm,type,option,verbose=False):
    """
    A routine to read one month of data from various datasets.
    
    This routine will read data one month at a time from various data sets
    described in *DataGrid()*
    
    Parameters
    ----------
    dataset :   
        Name of the dataset, ``ERA5``, ``GPCP``  

    vardir :   
        Path to the dataset 

    var1 :   
        Variable to extract 

    level :   
        Level of the Variable   

    yy :    
        Year
    
    mm :    
        Month

    type :   
        Type of data to reay. Currently hardwired to ``npy``

    option :    
        'Celsius'     For temperature Transform to Celsius
    
    verbose: 
        Tons of Output
    
    Returns
    --------
    
    average :
        Monthly data. 
    
    Examples
    --------
    
    >>> read_month('ERA5','.','Z','500',1979,12,'npy',[],verbose=verbose)
    >>> read_month('GPCP','.','TPREP','SURF',1979,12,'nc',[],verbose=verbose) 
    >>> read_month('ERA5','.','T','850',1979,12,'npy',option=Celsius,verbose=verbose)
    """
    info=DataGrid()
    if dataset == 'ERA5':
 #       def adddir(name,dir):
 #   return dir +'/' + name.split('.')[0]+'.npy'
        fil1=lib.adddir(lib.makemm(var1,str(level),yy,mm),info[dataset]['place'])
        if verbose: print(fil1)
        if var1 == 'T' and option == 'Celsius':
            data1=np.load(fil1) - 273.16
        else:
            data1=np.load(fil1)
    elif dataset == 'GPCP':       
        file = info[dataset]['place'] + '/gpcp_cdr_v23rB1_y' + str(yy) + '_m' + '{:02d}'.format(mm) + '.nc'      
        data1 = net.Dataset(file).variables["precip"][:,:]
    else:
        Print(' Error in read_month, datset set as {}'.format(dataset))
    return data1

def date_param():
    """ 
    Data Bank to resolve Month and Season averaging information

    Examples
    --------
    >>> index = data_param()
    >>> mon=index['DJF']['month_index']

    """
    months=['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
    DJF ={'label':'DJF','month_index':[12,1,2]}
    JFM ={'label':'JFM','month_index':[1,2,3]}
    AMJ ={'label':'AMJ','month_index':[4,5,6]}
    JJA ={'label':'JAS','month_index':[6,7,8]}
    JAS ={'label':'JAS','month_index':[7,8,9]}
    SON ={'label':'SON','month_index':[10,11,12]}
    ANN ={'label':'ANN','month_index':[i for i in range(1,13)]}
    JAN ={'label':'JAN','month_index':[1]}
    FEB ={'label':'FEB','month_index':[2]}
    MAR ={'label':'JAN','month_index':[3]}
    APR ={'label':'APR','month_index':[4]}
    MAY ={'label':'MAY','month_index':[5]}
    JUN ={'label':'JUN','month_index':[6]}
    JUL ={'label':'JUL','month_index':[7]}
    AUG ={'label':'AUG','month_index':[8]}
    SEP ={'label':'SEP','month_index':[9]}
    OCT ={'label':'OCT','month_index':[10]}
    NOV ={'label':'NOV','month_index':[11]}
    DEC ={'label':'DEC','month_index':[12]}
    out={'DJF':DJF,
        'JFM': JFM,
        'AMJ': AMJ,
        'JAS': JAS,
        'JJA': JJA,
        'SON': SON,
        'ANN': ANN,
        'JAN': JAN,
        'FEB': FEB,
        'MAR': MAR,
        'APR': APR,
        'MAY': MAY,
        'JUN': JUN,
        'JUL': JUL,
        'AUG': AUG,
        'SEP': SEP,
        'OCT': OCT,
        'NOV': NOV,
        'DEC': DEC,
        'MONTHS': months
        }
    return out

def DataGrid(option=None):
    """
    Routine that returns a Dictionary with information on the reuqested Data Set.

    Currently two data sets are supported   

    * ERA5 -- Subset of monthly data of ERA5 (compiled by AN 2019)   
    * GPCP -- Monthly data of precipitation data set 

    Info can be retrieved as ``grid[dataset][var]['start']`` for the starting years.
    See source for full explanation of the content.

    Parameters
    ----------
    Option :       
        * 'Verbose'      Tons of Output   
        * 'Info'         Info on data sets  

    Examples
    --------

    >>> DataGrid('info')
    >>> dat = DataGrid('verbose')
    """

    homedir = os.path.expanduser("~")
    if option == 'verbose': print('Root Directory for Data ',homedir)
    
    U ={      'level': [10, 50, 100,150, 200,250,300,400,500,600,700,850,925,1000],
              'start': 1979,
              'end': 2018,
              'label': 'U',
              'longname': 'Zonal Wind',
              'factor':1
               }
    V ={      'level': [10,50,100,150,200,250,300,400,500,600, 700,850,925,1000],
              'start': 1979,
              'end': 2018,
              'label': 'V',
              'longname': 'Meridional Wind',
              'factor':1
               }
    T ={      'level': [10,50,100,150, 200,250,300,400, 500,600, 700,850,925, 1000],
              'start': 1979,
              'end': 2018,
              'cv': 4,
              'label': 'T',
              'longname': 'Temperature',
              'factor':1
               }
    W ={      'level': [10,50,100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
              'start': 1979,
              'end': 2018,
              'cv': 0.01,
              'label': 'W',
              'longname': 'Vertical Velocity',
              'factor':1
               }
    Z ={      'level': [ 200, 500],
              'start': 1979,
              'end': 2018,
              'cv': 0.01,
              'label': 'Z',
              'longname': 'Geopotential Height',
              'factor':1
               }
    tp ={     'level': ['SURF'],
              'start': 1979,
              'end': 2018,
              'cv': 0.01,
              'label': 'TP',
              'longname': 'Precipitation',
              'factor':60
               }
    MSL ={     'level': ['SURF'],
              'start': 1979,
              'end': 2018,
              'cv': 10,
              'label': 'MSL',
              'longname': 'Mean Sea Level Pressure',
              'factor':1/100.
               }
    SST ={    'level': ['SURF'],
              'start': 1979,
              'end': 2018,
              'cv': 10,
              'label': 'SST',
              'longname': 'Sea Surface Tenperature',
              'factor':1
               }
    THETA ={      'level': [10,50,100,150, 200,250,300,400, 500,600, 700,850,925, 1000],
              'start': 1979,
              'end': 2018,
              'cv': 4,
              'label': 'T',
              'longname': 'Potential Temperature',
              'factor':1
               }
    dataera5={'nlat': 721,
              'nlon': 1440,
              'latvec':[i for i in np.arange(-90,90.1,0.25)],
              'lonvec': [i for i in np.arange(0,360.1,0.25)],
              'latnp': np.asarray([i for i in np.arange(-90,90.1,0.25)][::-1]),  # For plotting
              'lonnp': np.asarray([i for i in np.arange(0,360.1,0.25)]),
              'clim': homedir + '/Dropbox (CMCC)/ERA5/CLIM',
              'place': homedir +'/Dropbox (CMCC)/ERA5/DATA/ERA5_MM',
              'source_url': 'http://confluence.ecmwf.int/display/CKB/ERA5+data+documentation#ERA5datadocumentation-Parameterlistings',
              'desc':'ERA5 Monthly Mean for U,V,T,W,SLP 1979-2018',
              'special_value': 9999.,
              'U': U,
              'T': T,
              'V': V,
              'W': W,
              'Z': Z,
              'tp': tp,
              'MSL': MSL,
              'SST': SST,
              'THETA': THETA
              }
    
    precip_gpcp ={  'level': 'SURF',
              'start': 1979,
              'end': 2018,
              'label': 'precip',
              'factor':1,
              'longname': 'Precipitation GPCP',
               }
    
    datagpcp={'nlat': 72,
              'nlon': 144,
              'latvec':[i for i in np.arange(-88.75,90.1,2.5)],
              'lonvec': [i for i in np.arange(1.25,360.1,2.5)],
              'latnp': np.asarray([i for i in np.arange(-88.75,90.1,2.5)][::-1]),  # For plotting
              'lonnp': np.asarray([i for i in np.arange(1.25,360.,2.5)]),
              'precip': precip_gpcp,
              'place': homedir +'/Dropbox (CMCC)/ERA5/DATA/GPCP/TPREP',
              'clim': homedir + '/Dropbox (CMCC)/ERA5/DATA/GPCP/TPREP',
              'desc': 'Precipitation from the GPCP Project',
              'source_url': 'http://gpcp.umd.edu/'
              }


    grid={'ERA5': dataera5,
          'GPCP': datagpcp
         }
    if option == 'info':
        for i in list(grid.keys()):
            print(grid[i]['desc'])
            print(grid[i]['place'])
            print(grid[i]['source_url']+'\n')
        return
    
    return grid


def readvar_grid(region='globe',dataset='ERA5',var='Z',level='500',season='JAN',Celsius=False,verbose=False):
    """
    Read Variable from data sets
    
    Parameters
    ----------

    region :    
        *globe* for global maps, or [east, west, north, south]
        for limited region, longitude 0-360
    dataset :   
         name of data set
    var :   
         variable name
    level : 
        level, either a value or 'SURF' for surface fields

    season :    
        Month ('JAN') or season (,'DJF') or annual 'ANN')
     
    Celsius :   
        True/False for temperature transform to Celsius
    
    verbose :   
        True/False -- tons of output

    Returns
    -------

    xdat : numpy    
        array data 
    nlon :  
        Number of longitudes
    nlat :  
        Number of Latitudes
    lat :   
        Latitudes
    lon :   
        Longitudes

    Examples
    --------

    >>> readvar_grid(region='globe',dataset='ERA5',var='Z',level='500',season='JAN',Celsius=False,verbose=False)
    >>> readvar_grid(region='globe',dataset='ERA5',var='SST',level='SURF',season='JAN',Celsius=True,verbose=False)
    """
    
    vardir = '.'

    dat=date_param()

    grid = DataGrid()
    nlat = grid[dataset]['nlat']
    nlon = grid[dataset]['nlon']

    lat = grid[dataset]['latnp']
    lon = grid[dataset]['lonnp']
    #Correct for longitude in ERA5
    if dataset =='ERA5':
        lon=lon[:-1]
    sv=None
    try:
        sv=grid[dataset]['special_value']
        if verbose: print('  Using Special Value ---->', sv)
    except:
        print('  Special Value not defined for dataset {}'.format(dataset))
        
    ys=grid[dataset][var]['start']
    ye=grid[dataset][var]['end']
    ys=1979
    ye=2018
    lname=grid[dataset][var]['longname']

    nyears= ye-ys
    years =[i for i in range(ys,ye+1)]

    factor=grid[dataset][var]['factor']

    xdat= np.zeros([nlat,nlon,nyears+1])

    for tim in years:
        itim =years.index(tim)
        dat=date_param()
        mon=dat[season]['month_index']
        if verbose:
            print(' Plotting ' + var + ' from dataset ' + dataset)
            print('Printing year  ', tim)
        #
        if len(mon) > 1:
            if verbose: print('Mean on these months: {}'.format(mon))
            temp= np.zeros([nlat,nlon,len(mon)])
            for k in range(len(mon)):
                temp[:,:,k]=read_month(dataset,vardir,var,level,tim,mon[k],'npy',[],verbose=verbose) 
            xdat[:,:,itim]=np.mean(temp,axis=2)
        else:
            xdat[:,:,itim]=read_month(dataset, vardir,var,level,tim,mon[0],'npy',[],verbose=verbose) 

    return xdat,nlon, nlat,lat,lon,sv


def read_dataset(dataset='ERA5',region='globe',var='Z',level='500',season='DJF',verbose=False):
    '''
    Similar to `read_xarray` but returns a ``xarray DataSet``
    
    '''
    
    out = read_xarray(dataset=dataset, region=region, \
                            var=var,level=level,season=season,verbose=verbose)
    ds = xr.Dataset({var: out})
    return ds

def readvar_year(region='globe',period='all',dataset='ERA5',var='Z',level='500',
                 Celsius=False,verbose=False):
    """
    Read Variable from data banks, all month, no averaging
    
    Parameters
    ----------

    region :    
        'globe' for global maps, or [east, west, north, south]
        for limited region, longitude 0-360
    dataset :   
         name of data set
    var :   
         variable name
    level : 
        level, either a value or 'SURF' for surface fields

    period :    
        Time period to be read  
            * 'all' Every time level in databank  
            * [start_year,end_year] period in those years
     
    Celsius :   
        True/False for temperature transform to Celsius
    
    verbose :   
        True/False -- tons of output

    Returns
    -------

    xdat : numpy    
        array data 
    nlon :  
        Number of longitudes
    nlat :  
        Number of Latitudes
    lat :   
        Latitudes
    lon :   
        Longitudes

    Examples
    --------

    >>> readvar_year(region='globe',dataset='ERA5',var='Z',level='500',season='JAN',Celsius=False,verbose=False)
    >>> readvar_year(region='globe',dataset='ERA5',var='SST',level='SURF',season='JAN',Celsius=True,verbose=False)
    """
    
    vardir = '.'

    dat=date_param()

    grid = DataGrid()
    nlat = grid[dataset]['nlat']
    nlon = grid[dataset]['nlon']

    lat = grid[dataset]['latnp']
    lon = grid[dataset]['lonnp']
    #Correct for longitude in ERA5
    if dataset =='ERA5':
        lon=lon[:-1]
    sv=None
    try:
        sv=grid[dataset]['special_value']
        if verbose: print('  Using Special Value ---->', sv)
    except:
        print('  Special Value not defined for dataset {}'.format(dataset))

    ys=grid[dataset][var]['start']
    ye=grid[dataset][var]['end']
    ys=1979
    ye=2018
    lname=grid[dataset][var]['longname']
    #Choose period
    if period =='all':
        nyears= ye-ys
        years =[i for i in range(ys,ye+1)]
    else:
        nyears= period[1]-period[0]
        years =[i for i in range(period[0],period[1]+1)]

    factor=grid[dataset][var]['factor']

    xdat= np.zeros([nlat,nlon,12*(nyears+1)])
   
    itim=0
    dat=date_param()
    mon=dat['ANN']['month_index']
    if verbose : print(' Reading ' + var + ' from databank ' + dataset)
    for tim in years:
        print('Reading year  ', tim)
        for imon in mon:       
            if verbose : print('Reading mon  ', imon)
            xdat[:,:,itim]=read_month(dataset,vardir,var,level,tim,mon[imon-1],'npy',[],verbose=verbose)            
            if verbose : print('Reading time  ', itim)
            itim = itim + 1

    return xdat,nlon, nlat,lat,lon,sv
