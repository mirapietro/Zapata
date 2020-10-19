"""

This is the assembled library of codes jointly developed at CMCC that contains computation, plotting, and dataset reading modules.

The library core is composed by the following **modules**: 

**computation**: Routines for averaging and various computations 


The mapping is based on cartopy and GEOCAT libraries from NCAR.
It uses `xarray` as a basic data structure. 

The module `data` contains the information on the data banks. The routine `DataGrid` in `data` must be modified to the location of the basic data for each installation.

*Zapata* contains computation modules and plotting modules. Examples of working Jupyterlab Notebooks are in `docs.examples_notebook`

Modules
-------
    
**computation** :   
    Routines for averaging and various computations
    
**data** :  
    Information on data sets and routines to get data from the data sets

**mapping** :   
    Mapping routines based on *Cartopy*
    
**lib** :   
    Utilities for the rest of the modules.

**colormap** :
    Routines to use colormap in xml format

"""
