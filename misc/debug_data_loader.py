import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sc
import numpy.linalg as lin
import xarray as xr
import pandas as pd
import cartopy.crs as car

sys.path.append("../")

#import lib
import zapata.data as zdat
import zapata.computation as zcom
import zapata.mapping as zmap
from geocat.viz import cmaps as gvcmaps
from geocat.viz import util as gvutil

# check catalogue contents
zdat.inquire_catalogue()

#mycat='BSFS-NRT_daily'
#mycat='C-GLORSv7'
mycat='ERA5_MM'
#mycat='dummy_dataset'

zdat.inquire_catalogue(dataset=mycat, info=True)

## Read data
if mycat == 'dummy_dataset':
    xx=zdat.read_data(dataset=mycat, var='Z', period=[2000, 2010], season='DJF', level=[500])

if mycat == 'BSFS-NRT_daily':
    xx=zdat.read_data(dataset=mycat, var='vomecrty', period=[2019, 2019], season='ANN', level=[3.])

if mycat == 'C-GLORSv7':
    #xx=zdat.read_data(dataset=mycat, var='votemper', period=[2000, 2000], season='JFM', level=[500.])
    xx=zdat.read_data(dataset=mycat, var='sosstsst', period=[2000, 2000], season='JFM')

if mycat == 'ERA5_MM':
    xx=zdat.read_data(dataset=mycat, var='Z', season='DJF', level=[200])
    xx=zcom.zonal_var(mycat, 'T', period=[2000, 2001], season='DJF',level=[100,], option='Time')

print(xx)

print('Done')

