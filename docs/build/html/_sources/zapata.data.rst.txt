zapata.data module
==================

This module contains the two main functions of the data interface (see below for usage details):

- **inquire_catalogue** : Retrieve requested dataset informative structure from catalogue 
- **read_data** : Load into a DataArray the requested variable from specified dataset

The data extraction from each dataset is performed by the function **load_dataarray** that contains the `default` driver for extraction operations
and also handles the call to specific data drivers, which are contained in module `data_drivers.py` (see `data_drivers`_).

The dataset catalogue is located within the zapata library, named `catalogue.yml` (YAML format).

.. automodule:: zapata.data
   :members:
   :undoc-members:
   :show-inheritance:

.. _data_drivers:
.. automodule:: zapata.data_drivers
   :members:
   :undoc-members:
   :show-inheritance:
