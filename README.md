# Zapata
## Computational and Mapping Library   

A revolutionary library for analysis and plotting of meteorological data. The mapping is based on cartopy and GEOCAT libraries from NCAR.
It uses `xarray` as a basic data structure. 

The folder `docs` contains documentation for the modules. The subdirectory `htnl` contains the html file sfor the documentation produced by Sphinx. 

The directory 

The module `data` contains the information on the data banks. The routine `data_grid` in `data` must be modified to the location of the basic data for each installation.

# Packages

## zapata
contains computation modules and plotting modules. Examples of working Jupyterlab Notebooks are in `examples_notebook`

SubModules
-------
    
**computation** :   
    Routines for averaging and various computations
    
**data** :  
    Information on data sets and routine to get data from the data sets

**mapping** :   
    Mapping routines based on *Cartopy*
    
**lib** :   
    Utilties for the rest of the modules.


##  klus
contains algorithms for data analysis contributed by Stefan Klus

SubModules
-------
    
**algorithms** :   
    Routines for averaging and various computations
    
**kernel** :  
    Information on data sets and routine to get data from the data sets

**osbervables** :   
    Mapping routines based on *Cartopy*
    
Modules
-------

## interp
subroutine for interpolation

## zeus
Routine for remote working on Zeus
