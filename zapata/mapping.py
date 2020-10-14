'''
Contouring, stremlines and vector plots
=======================================

Latitude-Longitude Plotting 
---------------------------
A set of routines for horizontal plotting

**xmap, xsmap, xstmap, vecmap**

Zonal Plotting
--------------
These routine split zonal sections in pressures and latitude.

**zonal_plot, zonal_stream_plot, ocean_zonal_plot**

Utilities 
---------
**choose_contour, add_ticks, add_ticklabels, add_colorbar, make_ticks, add_sttick, set_title_and_labels**

Detailed Description:
---------------------

'''
import math as mp
import cartopy.crs as car
import cartopy.util as utl
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt 
import matplotlib.path as mpath
import numpy as np
import xarray as xr

import cartopy.mpl.ticker as ctick
import cartopy.feature as cfeature
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter)
import zapata.data as era
import zapata.computation as zcom
import zapata.lib as zlib

import mpl_toolkits.axes_grid1 as tl

import geocat.datafiles as gdf
from geocat.viz import cmaps as gvcmaps
import geocat.viz.util as gvutil

import docrep
# Uses DOCREP for avoiding copying docstring, contrary to the docs delete works
# only on one param at the time.

d = docrep.DocstringProcessor()


def init_figure(rows,cols,proview,constrained_layout=True,figsize=(16,8)):
    """
    Initialization for the entire figure, choose projection and number of panels.

    Parameters
    ----------
    rows :  
        Number of rows in multiple picture

    cols :       
        Number of columns in multiple picture

    figsize :    
        Size of figure in inches (height, width)

    proview :    
        Projection for the map  

        - *"Pacific"*,   cartopy PlateCarree, central longitude 180   
        - *"Atlantic"*,  cartopy PlateCarree, central longitude 0      
        - *"NHStereoEurope"*,  NH Stereo, central longitude 0     
        - *"NHStereoAmerica"*,  NH Stereo, central longitude 90   
        - *"SHStereoAfrica"*,  NH Stereo, central longitude 0     
     
    constrained_layout :    
        True/False 

    Returns
    -------
    fig :   handle   
        Dictionary with matplotlib-like info on the plot    
    ax :     axes   
        Axes of the panels  
    pro :    projection
        Projection chosen   
    """
    
    # This fixes the projection for the mapping: 
    #      Data Projection is then fixed in mapping routine
    if   proview == 'Pacific':
        projection = car.PlateCarree(central_longitude=180.)
    elif proview == 'Atlantic':
        projection = car.PlateCarree(central_longitude=0.)
    elif proview == 'NHStereoEurope':
        projection = car.NorthPolarStereo(central_longitude=0.)
    elif proview == 'NHStereoAmerica':
        projection = car.NorthPolarStereo(central_longitude=-90.)
    elif proview == 'SHStereoAfrica':
        projection = car.SouthPolarStereo(central_longitude=90.)
    else:
        print(' Error in init_figure projection {}'.format(proview))
        raise SystemExit
    #Store View in projection
    projection.view = proview

    fig, ax = plt.subplots(rows, cols,figsize=figsize, constrained_layout=constrained_layout, \
                           subplot_kw={ "projection": projection})
    
    print(' Opening figure , %i rows and %i cols \n' % (rows,cols))
    
    return fig,ax,projection

@d.get_sections(base='xmap')
@d.dedent
def xmap(field, cont, pro, ax=None, fill=True,contour=True, \
                    data_cent_lon=180, \
                    clabel=True, c_format = ' {:6.0f} ', \
                    refline=None, Special_Value = 9999.,\
                    lefttitle='',righttitle='',maintitle='',\
                    dxtick=30,dytick=30,\
                    xlimit=None,ylimit=None,\
                    colorbar=False,cmap='coolwarm',\
                    coasts=True,color_land='lightgray'):
    """
    Lat-Lon mapping function based on cartopy and NCAR GEOCAT.

    The data is supposed to be on a Plate (lat-lon) projection and the central longitude can be defined 
    via the paramater `data_cent_lon`. The defualt is `data_cent_lon=180`, meaning that the central longitude is over the pacific. 
    Coordinates that go from 0 to 360 implicitly assume such a Pacific central longitude.

    For the `Pacific` view it is assumed that the central longitude is at the daetline
    whereas for the `Atlantic` view the central longitude is at Greenwich.


    Parameters
    ----------
    field :
        xarray  --  cyclic point added in the routine
    cont :
        * [cmin,cmax,cinc]       fixed increment from cmin to cmax step cinc 
        * [ c1,c2, ..., cn]      fixed contours at [ c1,c2, ..., cn] 
        * n                      n contours 
        * []                     automatic choice  
    pro :
        Map Projection as initialized by init
    data_cent_lon :
        Central longitude for the data projection
    ax :
        Plot axis to be used   
    fill :
        True/False flag to have filled contours or not 
    contour :  
        True/False flag to have  contours or not
    refline :
        If a numeric value a single enhanced contour is plotted here
    clabel :
        True/False flag to have labelled contours or not
    c_format :
        Format for the contour labels
    dxtick : 
        Tick interval for longitude
    dytick :
        Tick interval for latitude
    Special_Value :
        Values to be ignore
    lefttitle :
        Title string on the lef
    righttitle :
        Title string on the right
    maintitle :
        Title string at the center
    cmap :
        Colormap
    coasts:
        False/True   Plotting or not empty coastlines 
    color_land:
        if coasts=False, use color_land for land
    xlimit:
        Longitude limits of the map, if not set they are derived from the data.
        The projection used is the geographical projection `pro`
    ylimit:
        Latitude limits of the map, if not set they are derived from the data.
        The projection used is the geographical projection `pro`
    
    
    Returns
    -------

    handle :    
        Dictionary with matplotlib-like info on the plot
    
    """
    

    #Check right projection
    this = pro.__class__.__name__
    if not this  in  ['PlateCarree']:
        print(' Wrong Projection in `xmap` {}'.format(this))
        raise SystemExit

    # Set data projection
    data_proj = car.PlateCarree(central_longitude=data_cent_lon)

    #Special Values
    data=field.where(field != 9999.)
    
    #Add Cyclic point
    data=gvutil.xr_add_cyclic_longitudes(data, 'lon')

    #Eliminate extra dimensions
    if len(data.shape) > 2:
        data = data.squeeze()
        
    # Add coastlines
    if coasts:
        ax.coastlines(linewidth=0.5)
    else:
        ax.coastlines(linewidths=0.5)
        ax.add_feature(cfeature.LAND, facecolor=color_land)
    
    if ylimit is  None:
        ylim = ax.projection.y_limits
    else:
        ylim=ylimit
    ax.set_ylim(ylim)

    if xlimit is  None:
        xlim = ax.projection.x_limits
    else:
        xlim=xlimit
    ax.set_xlim(xlim)

    print(' Plotting with x limits {}  '.format(xlim)  ) 
    print(' Plotting with y limits {}  '.format(ylim) )
    
    

    add_ticks(ax)
    add_ticklabels(ax)

    #Choose Contours
    vmin = data.min()
    vmax = data.max()
    cc = choose_contour(vmin,vmax,cont)
    print(f'Contouring from {vmin.data} to {vmax.data} with {cc} contours')
    


    handles = dict()
    if fill:
        handles["filled"] = data.plot.contourf(
            ax=ax,                            # this is the axes we want to plot to
            cmap=cmap,                        # our special colormap
            levels=cc,                        # contour levels specified outside this function
            vmax=vmax,                        # Maximum value for colormap  
            vmin=vmin,                        # Minimum value for colormap 
            xticks=make_ticks(xlim,dt=dxtick),    # nice x ticks
            yticks=make_ticks(ylim,dt=dytick),    # nice y ticks
            transform=data_proj,              # data projection,default is cent_lat=180
            add_colorbar=False,               # don't add individual colorbars for each plot call
            add_labels=False,                 # turn off xarray's automatic Lat, lon labels
        )
    if contour:
        handles['contours'] = data.plot.contour(
            ax=ax,                            # this is the axes we want to plot to
            colors='black',                   # our special colormap
            linestyles="-",
            linewidths=0.8,
            levels=cc,                        # contour levels specified outside this function
            vmax=vmax,                        # Maximum value for colormap  
            vmin=vmin,                        # Minimum value for colormap 
            transform=data_proj,              # data projection,default is cent_lat=180
            add_labels = False
        )
    if refline:
        handles["refline"] = data.plot.contour(
        ax=ax,
        levels=refline,
        colors="k",  # note plurals in this and following kwargs
        linestyles="-",
        linewidths=1.0,
        transform=data_proj,               # data projection,default is cent_lat=180
        add_labels=False  # again turn off automatic labels
        )
    # Label the contours
    if clabel and contour:
        ax.clabel(handles['contours'], colors='black', inline= True, use_clabeltext=True, inline_spacing=5,
                fontsize=8, fmt=c_format.format )
        if refline:
            ax.clabel(handles['refline'], colors='black', inline= True, use_clabeltext=True, inline_spacing=5,
                fontsize=8, fmt=c_format.format )
    
    # Use geocat.viz.util convenience function to add main title as well as titles to left and right of the plot axes.
    set_titles_and_labels(ax, lefttitle=lefttitle, lefttitlefontsize=14,
                                maintitle=maintitle, maintitlefontsize=16,
                                righttitle=righttitle, righttitlefontsize=14)
    # Add colorbar
    if colorbar:
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = tl.make_axes_locatable(ax)
        cax = divider.append_axes('right',size="2.5%", pad=0.2, axes_class=plt.Axes)
        ax.get_figure().colorbar(handles['filled'], cax=cax,orientation='vertical')
    
    return handles    

d.delete_params('xmap.parameters','dxtick')
d.delete_params('xmap.parameters.no_dxtick','dytick')
@d.dedent
def xsmap(field, cont, pro, ax=None, fill=True, contour=True, clabel=True,\
                      refline=None, Special_Value = 9999.,\
                      lefttitle='',righttitle='',maintitle='',\
                      xlimit=None,ylimit=None,\
                      colorbar=False,cmap='coolwarm',
                      coasts=True,color_land='lightgray',
                      Top_label=False,Lat_labels=True,
                      c_format = ' {:6.0f} ',
                      ):
    """
    Stereo mapping function for cartopy.

    Data must on a (0,360) longitude coordinate referred to Greenwich, but the `xlimit` are
    referred to the projection chosen in `init`.

    For the `Pacific` view it is assumed that the central longitude is at the daetline
    whereas for the `Atlantic` view the central longitude is at Greenwich.

    Parameters 
    ----------
    %(xmap.parameters.no_dxtick.no_dytick)s    
    Top_label :    
        True/False  Turn off the top labels
    Lat_labels :    
        True/False  Turn off the latitude labels
    
    Returns
    -------

    handle :    
        Dictionary with matplotlib-like info on the plot
    
    """
    #Check right projection
    this = pro.__class__.__name__
    if not this  in  ['SouthPolarStereo' ,'NorthPolarStereo' ]:
        print(' Wrong Projection in `xsmap` {}'.format(this))
        raise SystemExit

    #Special Values
    data=field.where(field != 9999.)

    # Fix the artifact of not-shown-data around 0 and 360-degree longitudes
    data = gvutil.xr_add_cyclic_longitudes(field, 'lon')
    
    #Eliminate extra dimensions
    if len(data.shape) > 2:
        data = data.squeeze()
        
    # Add coastlines
    if coasts:
        ax.coastlines(linewidth=0.5,color='grey')
    else:
        ax.coastlines(linewidths=0.5)
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
    
    # Choose Contours
    vmin = data.min()
    vmax = data.max()
    cc = choose_contour(vmin,vmax,cont)
    print(f'Contouring from {vmin.data} to {vmax.data} with {cc} contours')
   

    # Set axes limits
    if xlimit is  None:
        xlim = (np.amin(data.lon.values)-180,np.amax(data.lon.values)-180+0.001)
    else:
        xlim=xlimit
        
    if ylimit is  None:
        ylim = (np.amin(data.lat.values),np.amax(data.lat.values))
    else:
        ylim=ylimit
    print(' Plotting with x limits {}  '.format(xlim)  ) 
    print(' Plotting with y limits {}  '.format(ylim) )
    
    ax.set_extent(list(xlim+ylim),car.PlateCarree())  # 
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.spines['geo'].set_linewidth(2.0)
  
    handles = dict()
    if fill:
        handles["filled"] = data.plot.contourf(
            ax=ax,                            # this is the axes we want to plot to
            cmap=cmap,                        # our special colormap
            levels=cc,      # contour levels specified outside this function
            transform=car.PlateCarree(),      # data projection, for usual maps is assumed PlaceCarree
            #xticks=np.arange(xlim[0],xlim[1], 30),  # nice x ticks
            #yticks=np.arange(ylim[0], ylim[1], 15),    # nice y ticks
            add_colorbar=False,               # don't add individual colorbars for each plot call
            add_labels=False,                 # turn off xarray's automatic Lat, lon labels
        )
    if contour:
        handles['contours'] = data.plot.contour(
            ax=ax,                            # this is the axes we want to plot to
            colors='black',                       # our special colormap
            linestyles="-",
            linewidths=0.8,
            levels=cc,      # contour levels specified outside this function
            transform=car.PlateCarree(),      # data projection, for usual maps is assumed PlaceCarree
            add_labels = False
        )
    if refline:
        handles["refline"] = data.plot.contour(
        ax=ax,
        levels=[refline],
        colors="k",  # note plurals in this and following kwargs
        linestyles="-",
        linewidths=1.0,
        transform=car.PlateCarree(),      # data projection, for usual maps is assumed PlaceCarree
        add_labels=False  # again turn off autransform=car.Geodetic()tomatic labels
        )
    # Label the contours
    if clabel:
        ax.clabel(handles['contours'], colors='black', inline= True, use_clabeltext=True, inline_spacing=5,
                fontsize=8, fmt=c_format.format )
        if refline:
            ax.clabel(handles['refline'], colors='black', inline= True, use_clabeltext=True, inline_spacing=5,
                fontsize=8, fmt=c_format.format )
    
    xt = np.arange(-180,180,45)
    yt = np.arange(np.min(ylim),np.min([np.max(ylim),75]),20)
    add_sttick(ax,xt,yt[1:],xlim,ylim,pro,Top_label=Top_label,Lat_labels=Lat_labels,verbose=False)
    
    gl=ax.gridlines(color='grey', linestyle='--', draw_labels=True)
    gl.xlocator = mticker.FixedLocator(xt)
    gl.ylocator = mticker.FixedLocator(yt)
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    gl.xlabel_style = {'size': 16, 'color': 'black'}
    
    # Use geocat.viz.util convenience function to add main title as well as titles to left and right of the plot axes.
    set_titles_and_labels(ax, lefttitle=lefttitle, lefttitlefontsize=18,
                                maintitle=maintitle, maintitlefontsize=18,
                                righttitle=righttitle, righttitlefontsize=18,ytitle=1.00)
    # Add colorbar
    if colorbar:
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = tl.make_axes_locatable(ax)
        cax = divider.append_axes('right',size="2.5%", pad=0.2, axes_class=plt.Axes)
        ax.get_figure().colorbar(handles['filled'], cax=cax,orientation='vertical')

    return handles
def xstmap(U, V,  color='black', proj=car.PlateCarree(),ax=None, \
                      data_cent_lon=0, \
                      density=2, Special_Value = 9999.,\
                      lefttitle='',righttitle='',maintitle='',\
                      dxtick=30,dytick=30,\
                      xlimit=None,ylimit=None, \
                      colorbar=False,
                      cscale=(0,1),cmap='coolwarm',norm=None):
    """
    Plot streamline for field U and V.

    This routine draws streamlines for the fields U,V, optionally colored with the field ``color``. Internally assume data is always on a latlon projection with central longitude at Greenwich.

    The data is supposed to be on a Plate (lat-lon) projection and the central longitude can be defined 
    via the paramater `data_cent_lon`. The defualt is `data_cent_lon=180`, meaning that the central longitude is over the pacific. 
    Coordinates that go from 0 to 360 implicitly assume such a Pacific central longitude.

    For the `Pacific` view it is assumed that the central longitude is at the dateline
    whereas for the `Atlantic` view the central longitude is at Greenwich.


    Parameters      
    ----------
    U : xarray
        X component of the streamlines   
    
    V : xarray
        Y component of the streamlines
    
    color :   
        Color of the stremalines ('black'). If it is xarray color the streamlines with the colormap ``cmap``       
    
    cscale :
        If an array is used in `color`, it normalize to cscale = ( min, max)

    data_cent_lon : 
        Central longitude for the data projection
    
    density :    
        Density of the streamlines      

    ax :          
        Plot axis to be used        
    
    Special_Value :         
        Values to be ignored       
    
    lefttitle :         
        Title string on the left       
    
    righttitle :       
        Title string on the right      
    
    maintitle :      
        Title string at the center

    dxtick : 
        Tick interval for longitudes
        
    dytick :
        Tick interval for latitudes   
    
    cmap :  
        Colormap    
    
    colorbar :   
        False/True  
    
    xlimit :       
        Limit of the map (lon)  
    
    ylimit :        
        Limit of the map (lat)  

    Returns
    -------

    handle :    
        Dictionary with matplotlib-like info on the plot
    
    """
    #Special Values
    #U=U.where(U != 9999.)

    # Set data projection
    data_proj = car.PlateCarree(central_longitude=data_cent_lon)
    
    #select color scale, if `color` is an array
    # colors are normalized accordin to `cscale`
    this = type(color).__name__
    normuv = None

    #create local copies
    this_U = U
    this_V = V
    
    
    #check colorscale
    if this == 'str':
        color_scale = color
    elif this == 'DataArray':
        if not cscale:
            cscale= (color.min(), color.max() )
        normuv = plt.cm.colors.Normalize(cscale[0],cscale[1])
        print(f'Color set an xarray -> cscale {cscale}')
        color_scale = color.data      
    elif this == 'ndarray':  
        if not cscale:
            cscale= (color.max(), color.min() )
        normuv = plt.cm.colors.Normalize(cscale[0],cscale[1])
        print(f'Color set an array -> cscale {cscale}')
        color_scale=color
    else:
        color_scale='black'

    # Add coastlines
    ax.coastlines(linewidth=0.5)
    # Draw filled polygons for land
    ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black', color='lightgray')

    #Set axes limits
    if ylimit is  None:
        ylim = ax.projection.y_limits
    else:
        ylim=ylimit
    ax.set_ylim(ylim)

    if xlimit is  None:
        xlim = ax.projection.x_limits
    else:
        xlim=xlimit
    ax.set_xlim(xlim)
    
    # Stream-plot the data
    # There is no Xarray streamplot function, yet. So need to call matplotlib.streamplot directly. Not sure why, but can't
    # pass xarray.DataArray objects directly: fetch NumPy arrays via 'data' attribute'
    sp=ax.streamplot(this_U.lon.data, this_U.lat.data, this_U.data, this_V.data, \
        linewidth=1, density=density, color=color_scale, transform=proj,\
                      cmap=cmap,  zorder=1, norm = normuv)
    

    # Use geocat.viz.util convenience function to add main title as well as titles to left and right of the plot axes.
    set_titles_and_labels(ax, lefttitle=lefttitle, lefttitlefontsize=14,
                                 maintitle=maintitle, maintitlefontsize=16,
                                 righttitle=righttitle, righttitlefontsize=14)
    # Add colorbar
    if colorbar:
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = tl.make_axes_locatable(ax)
        cax = divider.append_axes('right',size="2.5%", pad=0.2, axes_class=plt.Axes)
        ax.get_figure().colorbar(sp.lines, cax=cax,orientation='vertical')
    # Use geocat.viz.util convenience function to set axes tick values
    tx=make_ticks(xlim,dt=dxtick)
    ty=make_ticks(ylim,dt=dytick)
    gvutil.set_axes_limits_and_ticks(ax, xlim=xlim, ylim=ylim, xticks=tx, yticks=ty)

    #  add minor and major tick lines
    add_ticks(ax)
    # Use geocat.viz.util convenience function to make plots look like NCL plots by using latitude, longitude tick labels
    gvutil.add_lat_lon_ticklabels(ax)

    return sp
def vecmap(U, V,  C, proj=car.PlateCarree(),ax=None, color='black',\
                      data_cent_lon=0,stride=10, Special_Value = 9999.,\
                      veckey=None,vec_min = 0.0, \
                      dxtick=30,dytick=30,\
                      lefttitle='',righttitle='',maintitle='',\
                      xlimit=None,ylimit=None, \
                      colorbar=False,cmap='coolwarm',cscale=None,**kw):
    """
    Vector plot for field U and V.

    This routine draws vectors for the fields U,V, optionally colored with the field ``color``. Internally assume data is always on a latlon projection with central longitude at Greenwich.

    Data must be on a (0,360) longitude coordinate referred to Greenwich, but the `xlimit` are
    referred to the projection chosen in `init`.

    For the `Pacific` view it is assumed that the central longitude is at the dateline
    whereas for the `Atlantic` view the central longitude is at Greenwich.

    It also takes all the other arguments for `quiver`


    Parameters      
    ----------
    U : xarray
        X component of the streamlines   
    
    V : xarray
        Y component of the streamlines
    
    C : xarray, (optional)
         xarray that color the streamlines with the colormap ``cmap``
    
    stride :
        Stride in the vector drawing
    
    colvec :   
        Color of the stremalines ('black').        
    
    cscale :
        If an array is used in `C`, it normalize the color to cscale = ( min, max)
    
    veckey : 
        It is a dictionary for the reference vector, the values are
            * X, Y -- Location in figure coordinates
            * U -- Reference Unit
            * label -- Reference string, it can a latex string, i.e  r'$1 \\frac{m}{s}$'   

    vec_min :   
        Minimum value to represent with the vectors

    ax :          
        Plot axis to be used        
    
    Special_Value :         
        Values to be ignored       
    
    lefttitle :         
        Title string on the left       
    
    righttitle :       
        Title string on the right      
    
    maintitle :      
        Title string at the center   
    
    dxtick : 
        Tick interval for longitudes
        
    dytick :
        Tick interval for latitudes 
    
    cmap :  
        Colormap    
    
    colorbar :   
        False/True  
    
    xlimit :       
        Limit of the map (lon)  
    
    ylimit :        
        Limit of the map (lat)

    Returns
    -------

    handle :    
        Dictionary with matplotlib-like info on the plot
      
    """
    
    #Special Values
    U=U.where(U != 9999.)
    
    #select color scale, color are normalized accordin to `cscale`
    this = type(C).__name__
    normuv = None

    #Eliminate small values
    mag = np.sqrt(U*U + V*V)
    this_U = U.where(mag > vec_min)
    this_V = V.where(mag > vec_min)
    this_C = C.where(mag > vec_min)
    
    # Check longitude coordinates
    # Because we are not using the integrated matplotlib with xarray, the coordinate must be 
    # transformed to (-180,180) if necessary
    if proj.view == 'Pacific':
        this_U = adjust_data_centlon(this_U)
        this_V = adjust_data_centlon(this_V)
        this_C = adjust_data_centlon(this_C)

    # Color array
    if this == 'DataArray':
        print(f'Color set an xarray {C.name}')
        if not cscale:
            cscale= (this_C.min(), this_C.max() )
        normuv = plt.cm.colors.Normalize(cscale[0],cscale[1])
        col_vec = this_C.data[::stride,::stride]      
    elif this == 'ndarray':
        print(f'Color set an array {this_C.name}')
        if not cscale:
            cscale= (this_C.max(), this_C.min() )
        normuv = plt.cm.colors.Normalize(cscale[0],cscale[1])
        col_vec=this_C[::stride,::stride]
    else:
        col_vec = None

    
    # Add coastlines
    ax.coastlines(linewidth=0.5)
    # Draw filled polygons for land
    ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black', color='lightgray')

    if ylimit is  None:
        ylim = ax.projection.y_limits
    else:
        ylim=ylimit
    
    if xlimit is  None:
        xlim = ax.projection.x_limits
    else:
        xlim=xlimit

    print(' Plotting with x limits {}  '.format(xlim)  ) 
    print(' Plotting with y limits {}  '.format(ylim) )
    


    # Vector the data
    # There is no Xarray quiver function, yet. So need to call matplotlib.streamplot directly. Not sure why, but can't
    # pass xarray.DataArray objects directly: fetch NumPy arrays via 'data' attribute'
    sp=ax.quiver(this_U.lon.data[::stride], this_U.lat.data[::stride], this_U.data[::stride,::stride], this_V.data[::stride,::stride],col_vec, \
                        linewidth=1, color=color,\
                        cmap=cmap,  zorder=1,norm = normuv,**kw)

    if veckey:
        ax.quiverkey( sp, veckey['X'], veckey['Y'], veckey['Unit'], veckey['label'],coordinates='figure')

    # Use geocat.viz.util convenience function to add main title as well as titles to left and right of the plot axes.
    set_titles_and_labels(ax, lefttitle=lefttitle, lefttitlefontsize=14,
                                 maintitle=maintitle, maintitlefontsize=16,
                                 righttitle=righttitle, righttitlefontsize=14)
    # Add colorbar
    if colorbar:
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = tl.make_axes_locatable(ax)
        cax = divider.append_axes('right',size="2.5%", pad=0.2, axes_class=plt.Axes)
        ax.get_figure().colorbar(sp, cax=cax,orientation='vertical')
    # Use geocat.viz.util convenience function to set axes tick values
    tx=make_ticks(xlim,dt=dxtick)
    ty=make_ticks(ylim,dt=dytick)
    gvutil.set_axes_limits_and_ticks(ax, xlim=xlim, ylim=ylim, xticks=tx, yticks=ty)

    #  add minor and major tick lines
    add_ticks(ax)
    # Use geocat.viz.util convenience function to make plots look like NCL plots by using latitude, longitude tick labels
    gvutil.add_lat_lon_ticklabels(ax)

    return sp

@d.get_sections(base='zonal_plot')
@d.dedent
def zonal_plot(data,ax,cont,cmap,colorbar=True, maintitle=None, lefttitle=None, righttitle=None,refline=None):
    """
    Zonal mapping function for xarray (lat,pressure). 
    
    Parameters
    ----------
    data :    
            xarray  --  cyclic point added in the routine (latitude, pressure)      
    cont :
        * [cmin,cmax,cinc]       fixed increment from cmin to cmax step cinc 
        * [ c1,c2, ..., cn]      fixed contours at [ c1,c2, ..., cn] 
        * n                      n contours 
        * []                     automatic choice  
    ax :            
            Plot axis to be used      
    Special_Value:     
            Values to be ignored      
    lefttitle:     
            Title string on the left     
    righttitle:   
            Title string on the right     
    maintitle:  
            Title string at the center     
    cmap:  
            Colormap     
    refline:   
            False/True if a zero line is desired     
    colorbar:   
            False/True if a colorbar is desired
    
    Returns
    --------
    
    handle: 
        Dict with plot parameters
    
    Examples
    --------
    
    >>> zonal_plot(data,ax,[],'BYR',colorbar=True, maintitle=None, lefttitle=None, righttitle=None,refline=None)
    """
    #Choose Contours
    vmin = data.min()
    vmax = data.max()
    cc = choose_contour(vmin,vmax,cont)
    print(f'Contouring from {vmin.data} to {vmax.data} with {cc} contours')
   

    handle = data.plot.contourf(
        ax=ax,                            # this is the axes we want to plot to
        cmap=cmap,                        # our special colormap
        levels=cc(cont),      # contour levels specified outside this function
        xticks=np.arange(-90, 90, 15),  # nice x ticks
        yticks=[1000,850,700,500,300,200,100],    # nice y ticks
        add_colorbar=colorbar,               # don't add individual colorbars for each plot call
        add_labels=False                 # turn off xarray's automatic Lat, lon labels
    )
    if refline:
        hc = data.plot.contour(
        ax=ax,
        levels=refline,
        colors="k",  # note plurals in this and following kwargs
        linestyles="-",
        linewidths=1.25,
        add_labels=False  # again turn off automatic labels
        )
    lev=data.pressure.values
    nlev=len(lev)
    ax.set_ylim(lev[nlev-1], lev[0])  # Invert y axis
    ax.set_xlim(90,-90)  # Invert x axis
    #
    set_titles_and_labels(ax, maintitle=maintitle, maintitlefontsize=18, \
                          lefttitle=lefttitle, lefttitlefontsize=18, \
                          righttitle=righttitle, righttitlefontsize=18,
                          xlabel='Latitude', ylabel='Pressure', labelfontsize=16)
    return handle
def zonal_stream_plot(datau,datav,ax,color='black',\
                     cmap='bwr', density=2,\
                     maintitle=None, lefttitle=None, righttitle=None,\
                     colorbar=True, smooth=True, special_value=9999):
    """
    Zonal mapping streamfunction.

    Plot zonal streamline for fielddatu e datav.

    Parameters      
    ----------
    datau : xarray
        X component of the streamlines   
    
    datav : xarray
        Y component of the streamlines
    
    color :   
        Color of the stremalines ('black'). If it is xarray color the streamlines with the colormap ``cmap``       
    
    density :    
        Density of the streamlines      

    ax :          
        Plot axis to be used        
    
    Special_Value :         
        Values to be ignored       
    
    lefttitle :         
        Title string on the left       
    
    righttitle :       
        Title string on the right

    maintitle:  
        Title string at the center
            
    cmap:  
        Colormap
            
    smooth:   
        False/True if smoothing is desired
            
    colorbar:   
        False/True if a colorbar is desired      
    

    """
    #Special Values
    datau=datau.where(datau != 9999.)
    datav=datav.where(datav != 9999.)
    
    #select color scale
    this = type(color).__name__
    if this == 'str':
        color_scale=color
    elif this == 'DataArray':
        print(this)
        color_scale = color.interp(pressure=np.arange(100,1000,50)).data
    elif this == 'ndarray':
        color_scale=color
    else:
        color_scale='black'
        colorbar=False

    #Eliminate extra dimensions
    if len(datau.shape) > 2:
        datau = datau.squeeze()
        datav = datav.squeeze()
    #Interpolate on a pressure regular grid
    U=datau.interp(pressure=np.arange(100,1000,50))
    V=datav.interp(pressure=np.arange(100,1000,50))
    
    # Stream-plot the data
    # There is no Xarray streamplot function, yet. So need to call matplotlib.streamplot directly. Not sure why, but can't
    # pass xarray.DataArray objects directly: fetch NumPy arrays via 'data' attribute'
    hc=ax.streamplot(U.lat.data, V.pressure.data, U.data, V.data, linewidth=1, density=density, color=color_scale, \
                     zorder=1,cmap=cmap   )
    #  Label the contours
    #     ax.clabel
    #         handles["contour"], fontsize=8, fmt="%.0f",  # Turn off decimal points
    #    )

    lev=U.pressure.values
    nlev=len(lev)
    ax.set_ylim(lev[nlev-1], lev[0])  # Invert y axis
    ax.set_xlim(90,-90)  # Invert x axis
    #
    set_titles_and_labels(ax, maintitle=maintitle, maintitlefontsize=18, \
                          lefttitle=lefttitle, lefttitlefontsize=18, \
                          righttitle=righttitle, righttitlefontsize=18,
                          xlabel='Latitude', ylabel='Pressure', labelfontsize=16)
    # Add colorbar
    if colorbar:
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = tl.make_axes_locatable(ax)
        cax = divider.append_axes('right',size="2.5%", pad=0.2, axes_class=plt.Axes)
        ax.get_figure().colorbar(hc.lines, cax=cax,orientation='vertical')

    return hc

def choose_contour(vmin,vmax,cont):
    """
    Choose contours according to length of `cont`.

    Parameters
    -----------
    vmin :
        Max Value of the field
    vmax :
        Min Value of the field
    cont :  
        * [cmin,cmax,cinc]       fixed increment from cmin to cmax step cinc 
        * [ c1,c2, ..., cn]      fixed contours at [ c1,c2, ..., cn]  
        * n                      n contours 
        * []                     automatic choice   

    Returns
    --------
    
    contour_levels :    
        Levels of selected Contours

    Examples
    --------
    >>> [0,1000,10] # Contours from 0 1000
    >>> [0.01, 0.05, 1.0, 2.0, 5.0, 10.0] # Set Contours Leves   
    """
    if len(cont) == 3:
        print("Setting Fixed Contours")
        cc=np.arange(cont[0]-cont[2],cont[1]+2*cont[2],cont[2])
        print(' Contouring from ', cont[0], '  to', cont[1],' with interval  ',cont[2])  
    elif len(cont)> 3:
        cc=cont
        print('Fixed Contours to..',cont)
    elif len(cont) == 1:
        cc=cont[0]
        print('Number of Contours ',cc)
    else:
        cc=10
        delta = abs(vmax.data-vmin.data)
        if delta < 0.0001:
            cc = [vmax]
            print(f'Almost constant field {vmax.data}, {vmin.data}, only one contour')
        else:
            print(f'Ten Contours automatic')
    return cc
def add_ticks(ax, x_minor_per_major=3, y_minor_per_major=3, labelsize="large",length=6,width=0.9):
    """
    Utility function to make plots look like NCL plots by adding minor and major tick lines
    
    Parameters
    -----------
    ax :       
        Current axes to the current figure

    x_minor_per_major :  
        Number of minor ticks between adjacent major ticks on x-axis

    y_minor_per_major :   
        Number of minor ticks between adjacent major ticks on y-axis
    """
    import matplotlib.ticker as tic

    ax.tick_params(labelsize=labelsize)
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(tic.AutoMinorLocator(n=x_minor_per_major))
    ax.yaxis.set_minor_locator(tic.AutoMinorLocator(n=y_minor_per_major))

    # length and width are in points and may need to change depending on figure size etc.
    ax.tick_params(
        "both",
        length=length,
        width=width,
        which="major",
        bottom=True,
        top=False,
        left=True,
        right=True,
    )
    ax.tick_params(
        "both",
        length=length/2,
        width=width/2,
        which="minor",
        bottom=True,
        top=False,
        left=True,
        right=True,
    )   
def add_ticklabels(ax, zero_direction_label=False, dateline_direction_label=False):
    """
    Utility function to make plots look like NCL plots by using latitude, longitude tick labels
    
    Parameters
    ----------
    ax :    
        Current axes to the current figure

    zero_direction_label :    
        Set True to get 0 E / O W or False to get 0 only.

    dateline_direction_label :      
        Set True to get 180 E / 180 W or False to get 180 only.
    """
    from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter)
    
    lon_formatter = LongitudeFormatter(zero_direction_label=zero_direction_label,
                        dateline_direction_label=dateline_direction_label,degree_symbol='')
    lat_formatter = LatitudeFormatter(degree_symbol='')
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    return
def add_colorbar(fig, handle, ax, colorbar_size=0.05,label_size=10,edges=True):
    """
    Add colorbar to plot.
    
    Parameters
    -----------
    
    handle :   
        Handle to plot
    ax :   
        Axis to which to add the colorbar
    colorbar_size:    
        Size of colorbar as a fraction of the axis
    label_size:   
        Size of labels
    edges:   
        Draw edges of the color bar
    """    
    #Eliminate overlapping labels
    stride = 1
    if len(handle.levels) > 10:
        stride = int(len(handle.levels)/10)
    divider = tl.make_axes_locatable(ax)
    cax = divider.append_axes('bottom',size="2.5%", pad=0.4, axes_class=plt.Axes)
    ax.get_figure().colorbar(handle, cax=cax, orientation='horizontal',\
                        ticks=handle.levels[::stride],fraction=colorbar_size,drawedges=edges)
    return
def make_ticks(xt,dt=20):
    '''
    Calculates nice tickes for the axes.
    Selecting 20 or 10 intervals

    Parameters
    ----------

    xt : 
        Axes limit [west, east]
    
    dt:
        Tick spacing

    Returns
    -------

    ticks:
        Nice ticks position
    '''
    n = (xt[1]-xt[0])/dt
    print(f' {n} Ticks set at {dt}   intervals') 
    return np.linspace(xt[0], xt[1], int(n+1))
def add_sttick(ax,xt,yt,xlim,ylim, promap,Top_label=True,Lat_labels=True,verbose=False):
    """
    Utility function add ticks and labels to stereo plots
    
    Parameters
    -----------
    ax :       
        Current axes to the current figure
    xt :    
        Locations of desired Longitude labels
    yt :     
        Locations of desired Latitude labels
    xlim :  
        Longitudinal extent of the stereo map
    ylim :  
        Latitudinal extent of the stereo map
    promap :    
        Projection 
    Top_label :    
        True/False  Turn off the top labels

    Lat_labels :    
        True/False  Turn off the latitude labels

    """
    # Hardwire ticks for grid lines
    xloc = xt
    yloc = yt
    xlab = []
    ylab = []
    # use Geodetic
    pro=car.Geodetic()

    for i in range(len(xloc)):
        if xloc[i] < 0: 
            xlab.append(str(abs(xloc[i])) + 'W')
        elif xloc[i] > 0: 
            xlab.append(str(xloc[i]) + 'E')
        else:
           xlab.append(str(xloc[i]))
    for i in range(len(yloc)):
        if yloc[i] < 0: 
            ylab.append(str(int(abs(yloc[i]))) + 'S')
        elif yloc[i] > 0: 
            ylab.append(str(int(yloc[i])) + 'N')
        else:
            ylab.append(str(yloc[i]))
    if verbose : print(xlab,ylab)

    # Center plot is always at (0,0) in projected coordinate
    # Choose border and label shift according to hemisphere
    if np.min(ylim) >= 0 :
        border = np.min(ylim) - 2.5
        shift = 0
    else:
        border = np.max(ylim) + 2.5
        shift = 1

    for i in range(len(xloc)):
        tx=ax.text(xloc[i], border, xlab[i], transform=pro,\
            horizontalalignment='center', verticalalignment='center',fontsize=16)
        tx=_set_label_location(tx,xloc[i], border ,promap,ylim)
        if shift > 0 and tx.__getattribute__('Location') == 'Right':
            tx.set_position([xloc[i],border+shift])
        if not Top_label and tx.__getattribute__('Location') == 'Top':
            tx._visible = False
    if Lat_labels:
        for i in range(len(yloc)):
            tx=ax.text(xloc[0], yloc[i], ylab[i], transform=pro,\
                horizontalalignment='center', verticalalignment='center',fontsize=16) 
        

    return
def _set_label_location(tx,x, y, pro,ylim):
    """ utility to find the location of labels on stereo projection """

    xu,yu = pro.transform_point(x,y,car.PlateCarree())
    xmin,xmax = pro.x_limits
    ymin,ymax = pro.y_limits
    ymax=ymax*abs(np.max(ylim)-np.min(ylim))/180.
    #print(xmin,xmax,ymin,ymax,xu,yu,x,y)
    if xu < 0 and abs(xu) > 0.001:
        tx.__setattr__('Location','Left')
    elif xu > 0 and abs(xu) > 0.001:
        tx.__setattr__('Location','Right')
    elif  yu > 0 and abs(xu) < 0.001:
        tx.__setattr__('Location','Top')
    elif  yu < 0 and abs(xu) < 0.001:
        tx.__setattr__('Location','Bottom')
    #print(xu,yu,ymax, ymin,tx.__getattribute__('Location'))

    if tx.__getattribute__('Location') == 'Left':
        tx.set_horizontalalignment('right')
    return tx
def set_titles_and_labels(ax, maintitle=None, maintitlefontsize=18, \
                          lefttitle=None, lefttitlefontsize=18, \
                          righttitle=None, righttitlefontsize=18,
                          xlabel=None, ylabel=None, labelfontsize=16,
                          ytitle=0.98):
    """
    Utility function to handle axis titles, left/right aligned titles.

    Parameters  
    ----------

    ax :    
        Current axes to the current figure  
    
    maintitle :   
        Text to use for the maintitle   

    maintitlefontsize :   
        Text font size for maintitle. A default value of 18 is used if nothing is set
    
    lefttitle :  
        Text to use for an optional left-aligned title, if any. For most plots, only a maintitle is enough,
        but for some plot types, a lefttitle likely with a right-aligned title, righttitle, can be used together.
    
    lefttitlefontsize :     
        Text font size for lefttitle. A default value of 18 is used if nothing is set
    
    righttitle :        
        Text to use for an optional right-aligned title, if any. For most plots, only a maintitle is enough,
        but for some plot types, a righttitle likely with a left-aligned title, lefttitle, can be used together.
    
    righttitlefontsize :    
        Text font size for righttitle. A default value of 18 is used if nothing is set

    xlabel :    
        Text for the x-axis label

    ylabel :    
        Text for the y-axis label

    labelfontsize :     
        Text font size for x- and y-axes. A default value of 16 is used if nothing is set
    
    ytitle :    
        Y position of the main title
    """
    

    if maintitle is not None:
        ax.set_title(maintitle, fontsize=maintitlefontsize, loc='center',pad=20)

    if lefttitle is not None:
        ax.set_title(lefttitle, fontsize=lefttitlefontsize,  loc='left')

    if righttitle is not None:
        ax.set_title(righttitle, fontsize=righttitlefontsize,  loc='right')

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=labelfontsize)

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=labelfontsize)

    return

d.delete_params('zonal_plot.parameters','data')
@d.dedent
def ocean_zonal_plot(data,ax,cont,cmap,colorbar=True, \
    fill=True,contour=True, \
    clabel=True, c_format = ' {:4.0f} ', \
    maintitle=None, lefttitle=None, righttitle=None,\
          xlimit=None,ylimit=None,refline=None,view='Atlantic'):
    """
    Ocean Zonal mapping function for xarray (lat,depth). 
    
    Parameters
    ----------
    data :    
            xarray  --  cyclic point added in the routine (latitude, depth)   
    %(zonal_plot.parameters.no_data)s
    fill :
        True/False flag to have filled contours or not 
    contour :  
        True/False flag to have  contours or not
    refline :
        If a numeric value a single enhanced contour is pltted here
    clabel :
        True/False flag to have labelled contours or not
    c_format :
        Format for the contour labels
    xlimit:
        x axis limits
        A Pacific view is obtained by giving limits in coordinae with
        the central longitude at Greenwich, e.g.  (120,-120) will produce a
        shifted Pacific section 
    ylimit:
        y axis limits
    view:
        Shift between `Atlantic` and `Pacific` views

    Returns
    -------
    
    handle: 
        Dict with plot parameters
    
    Examples
    --------
    
    >>> ocean_zonal_plot(data,ax,[],'BYR',colorbar=True, maintitle=None, lefttitle=None, righttitle=None,refline=None)
    """
    #Set a working view
    work = data
    
    #Choose Contours
    vmin = data.min()
    vmax = data.max()
    cc = choose_contour(vmin,vmax,cont)
    print(f'Contouring from {vmin.data} to {vmax.data} with {cc} contours')
        
    #Vertical levels
    lev=data.deptht.data
    nlev=len(lev)

    # Vertical ticks
    if lev.max() > 1500:
        levtick =[i for i in range(0,1000,200)] + [i for i in range(1000,5500,500)]
    elif lev.max() > 100:
        levtick =[i for i in range(0,1000,100)] 
    else:
        levtick = [i for i in range(0,1000,100)] 
    
    # Choose labels
    if 'lat' in data.dims:
        xlab = 'Latitude'
        xtick_loc = np.arange(-90,90,10)
        xtick_lab = [zlib.lat_string(i) for i in xtick_loc]
    elif 'lon' in data.dims:
        #Check longitude zero line
        if view == 'Pacific':
            #Pacific View
            print(f'Shifting Coordinates fro Pacific View')
            work=work.roll(lon=-720,roll_coords=False)
            xlab = 'Longitude'
            xtick_loc = np.arange(-180,181,20)
            dateline = int(len(xtick_loc)/2)
            xtick_lab = [zlib.long_string((i+180)%360) for i in xtick_loc]
            xtick_lab[dateline:] = [zlib.long_string(i -180) for i in xtick_loc[dateline:]]
        else:
            xlab = 'Longitude'
            xtick_loc = np.arange(-180,181,20)
            xtick_lab = [zlib.long_string(i) for i in xtick_loc]
    else:
        xlab = 'Missing'
    
    
    handles = dict()
    if fill:
        handles['filled'] = work.plot.contourf(
            ax=ax,                            # this is the axes we want to plot to
            cmap=cmap,                        # our special colormap
            levels=cc,      # contour levels specified outside this function
            xticks=xtick_loc,  # nice x ticks
            yticks = levtick,    # nice y ticks
            add_colorbar=colorbar,               # don't add individual colorbars for each plot call
            add_labels=False                 # turn off xarray's automatic Lat, lon labels
        )
    if contour:
        handles['contours'] = work.plot.contour(
            ax=ax,                            # this is the axes we want to plot to
            colors='black',                       # our special colormap
            linestyles="-",
            linewidths=0.8,
            levels=cc,      # contour levels specified outside this function
            add_labels = False
        )
    if refline:
        handles["refline"] = work.plot.contour(
        ax=ax,
        levels=refline,
        colors="k",  # note plurals in this and following kwargs
        linestyles="-",
        linewidths=1.0,
        add_labels=False  # again turn off autransform=car.Geodetic()tomatic labels
        )
    
    ax.set_xticks(xtick_loc)
    ax.set_xticklabels(xtick_lab)

    # Y limits
    if ylimit == None :
        ax.set_ylim(lev[nlev-1], lev[0])  # Invert y axis
    else:
        ax.set_ylim(ylimit)
    #X limits
    if (xlimit == None) and (xlab == 'Latitude'):
        ax.set_xlim(90,-90)  # Invert x axis
    elif (xlimit == None) and (xlab == 'Longitude'):
        ax.set_xlim(-180,180)  
    else:
        ax.set_xlim(xlimit)

    # Label the contours
    if clabel and contour:
        ax.clabel(handles['contours'], colors='black', inline= True, use_clabeltext=True, inline_spacing=5,
                fontsize=8,rightside_up=True,fmt=c_format.format )
        if refline:
            ax.clabel(handles['refline'], colors='black', inline= True, use_clabeltext=True, inline_spacing=5,
                fontsize=8, rightside_up=True, fmt=c_format.format )
    
    # set bathymetry to black
    ax.set_facecolor('k')

    set_titles_and_labels(ax, maintitle=maintitle, maintitlefontsize=18, \
                          lefttitle=lefttitle, lefttitlefontsize=18, \
                          righttitle=righttitle, righttitlefontsize=18,
                          xlabel=xlab, ylabel='Depth', labelfontsize=18)
    return handles
def small_map(fig,ax,cent_lon=0,lat=None, lon=None):
    '''
    Add small map for sections. The line is drawn at one
    `lat` or `lon`.

    Parameters
    ==========

    fig:
        `fig` to add the small map
    ax:
        `ax` to used
    cent_lon:
        central longitude for longitude sections
    lat: (None)
        Latitude of the longitude section
    lon: (None)
        Longitude of the latitude section
    '''
    pro = car.PlateCarree(central_longitude=cent_lon)
    X,Y = ax.get_position().get_points()
    r_ax = fig.add_axes([0.125,0.125,Y[0]/8,Y[1]/8], anchor='SW', facecolor='w',projection=pro)
   
    r_ax.set_global()
    if lat is not None:
        r_ax.plot([-180,180],[lat,lat],transform=pro,ls='-',color='r')
    if lon is not None:
        r_ax.plot([lon,lon],[-90,90],transform=pro,ls='-',color='r')
    r_ax.coastlines(resolution='110m')
    r_ax.gridlines()
    return

def adjust_data_centlon(data,centlon_old=0, centlon_new=180.):
    '''
    Adjust xarray to a different central longitude.
    The data are rolled without rolling the coordinates.

    Parameters
    ==========
    data: xarray
        xarray data to be shifed. it is assumed to have dimension `lat` and `lon`
    centlon_old:
        Original Central longitude of the data
    centlon_new:
        New central longitude of the data
    
    Returns
    =======
        Rolled xarray
    '''

    # Check longitude dimension

    if 'lon' not in data.dims:
        SystemError(f'Wrong dimension in `adjust_data_centlon` --> {data.dims}')
    #Check shift
    if abs(centlon_new-centlon_old) < 180:
        SystemError(f'Wrong dimension in `adjust_data_centlon` /n  \
        Actual implementation only for 180 shifts  --> {centlon_new} -- {centlon_old}')
    
    print(f'Shifting from {centlon_old} to {centlon_new}')

    #Get shift value
    shift = int(data.sizes['lon']/2+1)
    
    #Roll data to new cent_lon
    new = data.roll(lon=720,roll_coords=False)

    return new
