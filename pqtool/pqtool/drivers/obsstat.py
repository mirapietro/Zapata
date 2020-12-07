import xarray as xr
import numpy as np
import pandas as pd
from netCDF4 import num2date
from .base import DataSource


class ObsstatSource(DataSource):
    """Intake driver class to handles obsstat files

    Attributes:
        -----------
    name: name of the drive
    version: version string for the driver
    container: container type of data sources created by this object
    partition_access: data sources returned by this driver have multiple partitions
    """
    container = 'xarray'
    name = 'obsstat'
    version = '0.0.1'
    partition_access = True

    def _get_partition(self, i):
        """Return all of the data from partition id i

        :param i: partition number
        :return data: data from partition id i
        """
        data = xr.open_dataset(self.files[i]).chunk()

        time = self.metadata.get('coords', {}).get('time', 'TIME')

        # Create dimension model
        data['OMG'] = xr.concat([data['OMG']], dim='model')
        data.coords['model'] = [self.name]

        variables = []
        for name, param in self.metadata.get('variables', {}).items():
            variable = data.where(data['PARAM'] == param).dropna('OBS')
            variable['OMG'] = variable['VALUE'] - variable['OMG']

            if len(variable[time]):
                variable[time].data = num2date(variable[time].data,
                                               'days since 1950-01-01 00:00',
                                               calendar='standard').astype('datetime64[s]')

            try:
                variable['INST'] = variable['INST'].astype(int)
            except ValueError:
                pass

            variable = variable.rename({v: k for k, v in self.metadata.get('coords', {}).items()})

            variable = variable.rename({'OBS': 'obs',
                                        'VALUE': name,
                                        'OMG': 'model_%s' % name})

            coords = self.metadata.get('coords', {}).keys()
            index = pd.MultiIndex.from_arrays([variable[c] for c in coords])
            variable = variable.reindex(obs=index).drop_vars(coords)

            variable = variable.get([name, 'model_%s' % name])

            duplicated = index.duplicated(keep='first')
            if duplicated.any():
                variable = variable.isel(obs=~duplicated)

            variables.append(variable)

        return xr.merge(variables, join='outer')

    def read(self):
        """Return all the data in memory in one in-memory container.

        :return: data in memory in an xarray container
        """
        self._load_metadata()
        return xr.concat([self.read_partition(i) for i in range(self.npartitions)], dim='obs')
