Zapata package
==============


This is the assembled library of codes jointly developed at CMCC that contains computation, plotting, and dataset reading modules to analyse and produce graphics from meteorological and oceanographic data.

The library core is composed by the following **modules**:

- **computation**: routines for averaging and various computations

- **data**: information on data sets and routines to retrieve data from built-in data banks list and/or user defined input data.

- **mapping**: mapping routines based on Cartopy (https://scitools.org.uk/cartopy) and GEOCAT (https://geocat.ucar.edu/) libraries.

- **colormap**: routines to use colormap in xml format

- **lib**: utilities for the rest of the modules.


The whole infrastructure uses `xarray` as a basic data structure.


Submodules
----------

.. toctree::
   :maxdepth: 4

   zapata.colormap
   zapata.computation
   zapata.data
   zapata.lib
   zapata.mapping
   zapata.things_to_do
   zapata.work_in_progress

