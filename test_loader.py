import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sc
import numpy.linalg as lin
import xarray as xr
import pandas as pd
import cartopy.crs as car


#import lib
import zapata.data as zera
import zapata.computation as zcom
import zapata.mapping as zmap
from geocat.viz import cmaps as gvcmaps
from geocat.viz import util as gvutil


# Read data and compose zonal means¶
ud=zcom.zonal_var('ERA5','U','DJF')

print('Done')

