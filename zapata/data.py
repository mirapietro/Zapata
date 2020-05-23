import os
import numpy as np
import xarray as xr
import pandas as pd
import zapata.lib as lib
import netCDF4 as net

# NCEP Reanalysis standard pressure levels
# 1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10
# 1000  925  850  700  600, 500  400  300, 250, 200, 150, 100      50          10
# 
def read_month(dataset, vardir,var1,level,yy,mm,type,option,verbose=False):
    """
    A routine to read one month of data from various datasets.
    
    This routine will read data one month at a time from various data sets
    described in _DataGrid()_
    
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

def DataGrid(verbose=False):
    """
    Routine that returns a Dictionary with information on the reuqested Data Set.

    Currently two data sets are supported   

    * ERA5 -- Subset of monthly data of ERA5 (compiled by AN 2019)   
    * GPCP -- Monthly data of precipitation data set 

    Info can be retrieved as ``grid[dataset][var]['start']`` for the starting years.
    See source for full explanation of the content.

    Parameters
    ----------

    verbose :   
        True/False      Tons of Output


    Examples
    --------

    >>> info = data_param()
    >>> info = data_param(verbose=True)
    """

    homedir = os.path.expanduser("~")
    if verbose: print('Root Directory for Data ',homedir)
    
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
              'clim': homedir + '/Dropbox (CMCC)/ERA5/DATA/GPCP/TPREP'
              }


    grid={'ERA5': dataera5,
          'GPCP': datagpcp
         }
    return grid


def GPCPInfo():
    """Location data for the GPCP data set. """
    source_url='http://gpcp.umd.edu/'
    out = {'Info': source_url}
    return out   

def ERA5Info():
    """Location data for the ERA5 data set. """
    source_url='://confluence.ecmwf.int/display/CKB/ERA5+data+documentation#ERA5datadocumentation-Parameterlistings'
    out = {'Info': source_url}
    return out

def readvar_grid(region='globe',dataset='ERA5',var='Z',level='500',season='JAN',Celsius=False,verbose=False):
    """
    Read Variable from data sets
    
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

    try:
        grid[dataset]['special_value']
        if verbose: print('  Using Special Value ---->', grid[dataset]['special_value'])
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

    return xdat,nlon, nlat,lat,lon

def read_xarray(dataset='ERA5',region='globe',var='Z',level='500',season='DJF',verbose=False):
    '''
    Read npy files from data and generates xarray.

    This a xarray implementation of read_var. It always grabs the global data.

    Parameters
    ----------
    dataset :   
        name of data set   
    region: 
        select region   
        * 'globe', _Entire globe_
        * [East, West, North, South], _Specific region_
    var :   
         variable name
    level : 
        level, either a value or 'SURF' for surface fields

    season :    
        Month ('JAN') or season (,'DJF') or annual 'ANN'), or 'ALL' for every year
    verbose:    
        True/False -- Tons of Output

    Returns
    -------

    out : xarray   
        array data 
    
    '''
    
    if season != 'ALL':
        xdat,nlon, nlat,lat,lon=readvar_grid(region='globe',dataset=dataset, \
                            var=var,level=level,season=season,Celsius=False,verbose=verbose)
        times = pd.date_range('1979-01-01', periods=40,freq='YS')
    elif season == 'ALL':
        xdat,nlon, nlat,lat,lon=readvar_year(region='globe',dataset=dataset, \
                            var=var,level=level,period='all',Celsius=False,verbose=verbose)
        times = pd.date_range('1979-01-01', periods=480,freq='MS')
    
    out = xr.DataArray(xdat, coords=[lat, lon, times], dims=['lat','lon','time'])

    if region != 'globe':
        out = out.sel(lon = slice(region[0],region[1]), lat = slice(region[2],region[3]))

    
    return out

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
            * 'all' _Every time level in databank_  
            * [start_year,end_year] _period in those years_
     
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

    try:
        grid[dataset]['special_value']
        if verbose: print('  Using Special Value ---->', grid[dataset]['special_value'])
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

    return xdat,nlon, nlat,lat,lon