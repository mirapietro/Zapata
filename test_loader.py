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


# Read data and compose zonal means¶
ud=zcom.zonal_var('ERA5_MM','T', period=[2000, 2001], season='DJF',level=[100,])


print('Done')

