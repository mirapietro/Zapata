# Zapata
## Computational and Mapping Library   

A revolutionary library for analysis and plotting of meteorological data. The mapping is based on cartopy and GEOCAT libraries from NCAR.
It uses `xarray` as a basic data structure. The folder `docs` contains documentation for the modules.

The directory 

The module `data` contains the information on the data banks. The routine `data_grid` in `data` must be modified to the location of the basic data for each installation.

**ZAPATA** contains computation modules and plotting modules. Examples of working Jupyterlab Notebooks are in `examples_notebook`

Modules
-----------
    
**computation** :   
    Routines for averaging and various computations
    
**data** :  
    Information on data sets and routine to get data from the data sets

**mapping** :   
    Mapping routines based on *Cartopy*
    
**lib** :   
    Utilties for the rest of the modules.
