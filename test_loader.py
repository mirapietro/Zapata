import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sc
import numpy.linalg as lin
import xarray as xr
import pandas as pd
import cartopy.crs as car


#import lib
import zapata.data as zdat
import zapata.computation as zcom
import zapata.mapping as zmap
from geocat.viz import cmaps as gvcmaps
from geocat.viz import util as gvutil


# check catalogue contents
zdat.inquire_catalogue()

#zdat.inquire_catalogue(dataset='C-GLORSv7', info=True)
## Read data
#xx=zdat.read_xarray(dataset='C-GLORSv7', var='votemper', period=[2000, 2001], season='JFM', level=[500.])


zdat.inquire_catalogue(dataset='ERA5_MM', info=True)
yy = zdat.read_xarray(dataset='ERA5_MM', var='SST', period=[2000, 2001], season='JFM', level=['SURF',])
# Read data and compose zonal means (local)
ud=zcom.zonal_var('ERA5_MM','T', period=[2000, 2001], season='DJF',level=[100,], option='Time')


print('Done')

