import os, sys
import math
import numpy as np
import numpy.ma as ma
import xarray as xr

sys.path.insert(1, '/users_home/oda/pm28621/Zapata/')
import interp as zint

def Oliveri_SoL(input_matrix, depth=1):  
# (2016) depth is to select the number of consequential mask points to fill
    # depth loop
    for d in range(depth):
        if np.sum(input_matrix.mask) == 0:  # nothing to fill
            return input_matrix
        else:
            # Create a m x n x 8 3D matrix in which, third dimension fixed, the other dimensions
            #  contains values that are shifted in one of the 8 possible direction compared to the original matrix
            shift_matrix = ma.array(np.empty(shape=(input_matrix.shape[0], input_matrix.shape[1], 8)),
                                    mask=True, fill_value=1.e20, dtype=float)
            # up shift
            shift_matrix[: - 1, :, 0] = input_matrix[1:, :]
            # down shift
            shift_matrix[1:, :, 1] = input_matrix[0: - 1, :]
            # left shift
            shift_matrix[:, : - 1, 2] = input_matrix[:, 1:]
            # right shift
            shift_matrix[:, 1:, 3] = input_matrix[:, : - 1]
            # up-left shift
            shift_matrix[: - 1, : - 1, 4] = input_matrix[1:, 1:]
            # up-right shift
            shift_matrix[: - 1, 1:, 5] = input_matrix[1:, : - 1]
            # down-left shift
            shift_matrix[1:, : - 1, 6] = input_matrix[: - 1, 1:]
            # down-right shift
            shift_matrix[1:, 1:, 7] = input_matrix[: - 1, : - 1]
            # Mediate the shift matrix among the third dimension
            mean_matrix = ma.mean(shift_matrix, 2)
            # Replace input missing values with new ones belonging to the mean matrix
            input_matrix = ma.array(np.where(mean_matrix.mask + input_matrix.mask, mean_matrix, input_matrix),
                                    mask=mean_matrix.mask, fill_value=1.e20, dtype=float)
            input_matrix = ma.masked_where(mean_matrix.mask, input_matrix)
    return input_matrix

def test_zapata_SoL(in_a:np.array,
                    window = 3,
                    period = 1,):
   
   in_a = xr.DataArray(in_a)

   # mask_sea_over_land
   masknan = xr.where(np.isnan(in_a), np.nan, 1)
   um1 = masknan.ffill(dim='dim_0',limit=1).fillna(0.) + masknan.bfill(dim='dim_0',limit=1).fillna(0.)- 2*masknan.fillna(0)
   um2 = masknan.ffill(dim='dim_1',limit=1).fillna(0.) + masknan.bfill(dim='dim_1',limit=1).fillna(0.)- 2*masknan.fillna(0)
   bord = (um1+um2)/2
   um = bord + masknan.fillna(0)
   um = xr.where(um!=0, 1, np.nan)
   mb = xr.where(bord !=0, 1, np.nan)

   # fill_sea_over_land
   r = in_a.rolling(dim_0=window, dim_1=window, min_periods=period, center=True) 
   r1 = r.mean()
   border = ~np.isnan(mb).stack(ind=mb.dims)
   aa = in_a.stack(ind=in_a.dims)
   rs = r1.stack(ind=r1.dims)
   aa[border]=rs[border]
   aaa = aa.unstack()

   return aaa


def Romain_SoL(var_in,
               nloop = 1,
               xdim='x', ydim='y',
               ismax = False):
   var_out = var_in.copy()
   for loop in range(nloop):
      # initialize the tuple to store the shifts
      var_shift = ()
      # shift in all directions
      for x in range(-1, 2):
         for y in range(-1, 2):
            if ((x != 0) | (y != 0)): # skip the no-shifting
               # store the shifting in the tuple
               var_shift = var_shift + (var_out.shift({xdim:x, ydim:y}),)
      # take either the mean or the max over 'shift'
      if ismax:
         var_mean = xr.concat(var_shift, dim='shift').max(dim='shift')
      else:
         var_mean = xr.concat(var_shift, dim='shift').mean(dim='shift')
      # Replace input masked points (nan values) with new ones
      var_out = var_out.where(~np.isnan(var_out), other=var_mean)
      if np.sum(np.isnan(var_out)) == 0:  # nothing more to flood
          print('WARNING. Field does not have anymore land points,', str(loop + 1),
                'steps were sufficient to flood it completely.', file=sys.stderr)
          break
   return var_out
   

def Girardi_SoL(data, iterations=1, copy=False):
    if copy:
        data = np.ma.copy(data)

    if not np.ma.is_masked(data):
        return data

    for _ in range(iterations):
        shifted = []
        ni, nj = data.shape
        for i in range(-1,2):
            for j in range(-1,2):
                if i != 0 or j != 0:
                    shifted.append(data[1+i:ni-1+i,1+j:nj-1+j])

        approx = np.ma.mean(shifted, axis=0)
        
        view = data[1:-1,1:-1]
        np.copyto(view, approx, where=(view.mask & ~approx.mask))

        view.mask &= approx.mask

    return data


def Zapata_SoL(yy, mm, ver):
   """ The interpolator is initialized by default with window=3, period=1, method='linear'
   """
   # define the interpolator
   w = zint.Ocean_Interpolator("L75_025_TRP_GLO", "L75_025_REG_GLO", level=ver)

   udata, vdata = Zapata_inputs(yy, mm, ver)

   ### from interp_UV
   # Insert NaN
   udata = xr.where(udata < 200, udata, np.nan)
   vdata = xr.where(vdata < 200, vdata, np.nan)
   # Fill U, V values over land
   udata = w.fill_sea_over_land(udata, w.masku)
   vdata = w.fill_sea_over_land(vdata, w.maskv)

   return udata, vdata


def Zapata_inputs(yy, mm, ver):
   """ Read the inputs in the way Zapata does it. Used for both Z and Romain's
   """
   # read U, V inputs
   filename = "/data/products/GLOBAL_REANALYSES/C-GLORSv7/MONTHLY/NEMO_1m_"+str(yy)+"_grid_U.nc"
   ini_u = xr.open_dataset(filename)
   filename = "/data/products/GLOBAL_REANALYSES/C-GLORSv7/MONTHLY/NEMO_1m_"+str(yy)+"_grid_V.nc"
   ini_v = xr.open_dataset(filename)
   variables3DUV = {"vozocrtx":ini_u.vozocrtx, "vomecrty":ini_v.vomecrty}

   # Input variables
   udata = variables3DUV["vozocrtx"][mm,ver,:,:]
   vdata = variables3DUV["vomecrty"][mm,ver,:,:]

   return udata, vdata


if  __name__ == "__main__":
 
   print('--------------------------------------------')
   print('    Comparison of SoL on a small matrix     ')
   print('--------------------------------------------')

   a = np.empty((11,11))
   a[:] = np.nan
   a[5,3:8] = [1,2,3,4,5]
   a[4,4:7] = [2,3,4]
   a[3,5] = 3
   a[6,4:7] = 1
   a[7,5] = 1
    
   data = np.ma.masked_array(a.copy(), np.isnan(a))
   print('Input \n', data.data)
   gi = Girardi_SoL(data)
   print('Girardi \n', gi.data.round(1))
   za = test_zapata_SoL(a.copy())
   print('Zapata \n', za.round(1).values)
   ro = Romain_SoL(xr.DataArray(a), xdim='dim_0', ydim='dim_1')
   print('Romain \n', ro.round(1).values, '\n')
 
   print('--------------------------------------------')
   print('       Comparison of SoL on U, V data       ')
   print('--------------------------------------------')
   # Zapata SoL
   Z_udata, Z_vdata = Zapata_SoL(yy = 2016, mm = 3, ver = 0)

   # Romain SoL
   u_in, v_in = Zapata_inputs(yy = 2016, mm = 3, ver = 0)
   u_in = xr.where(u_in < 200, u_in, np.nan)
   v_in = xr.where(v_in < 200, v_in, np.nan)
   R_udata = Romain_SoL(u_in)
   R_vdata = Romain_SoL(v_in)

   # Girardi SoL
   u_in, v_in = Zapata_inputs(yy = 2016, mm = 3, ver = 0)
   u_in = xr.where(u_in < 200, u_in, np.nan)
   v_in = xr.where(v_in < 200, v_in, np.nan)
   G_u = Girardi_SoL(np.ma.masked_array(u_in, np.isnan(u_in))).data
   G_v = Girardi_SoL(np.ma.masked_array(v_in, np.isnan(v_in))).data

   # Results
   GvsZ = ((G_u == Z_udata) | (np.isnan(G_u) & np.isnan(R_udata))).sum().values
   GvsR = ((G_u == R_udata) | (np.isnan(G_u) & np.isnan(R_udata))).sum().values
   tot = G_u.shape[0] * G_u.shape[1]
   print(f"Zapata and Girardi differ in {tot-GvsZ} values out of {tot}.")
   print(f"Romain and Girardi differ in {tot-GvsR} values out of {tot}.")

   comp=(G_u == R_udata) | (np.isnan(G_u) & np.isnan(R_udata))
   rows = np.where(comp == False)[0]
   cols = np.where(comp == False)[1]
   print('Romain and Girardi differ in these points:')
   for i,j in zip(rows, cols):
      print(f'Girardi: {G_u[i,j]}, Romain: {R_udata[i,j].values}, in coordinates ({i},{j})')
