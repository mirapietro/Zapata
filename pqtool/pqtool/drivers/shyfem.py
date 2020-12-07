import xarray as xr
import numpy as np
from .base import DataSource


class ShyfemSource(DataSource):
    """Intake driver class to handles obsstat files

    Attributes:
    -----------
    name: name of the drive
    version: version string for the driver
    container: container type of data sources created by this object
    partition_access: data sources returned by this driver have multiple partitions
    """
    container = 'xarray'
    name = 'shyfem'
    version = '0.0.1'
    partition_access = True

    def _get_partition(self, i):
        """Return all of the data from partition id i

        :param i: partition number
        :return data: data from partition id i
        """
        data = xr.open_dataset(self.files[i]).chunk()

        #coords = self.metadata.get('coords', None)
        #if coords:
        #    data = data.get([v for v in coords.values() if v in data])
        #    data = data.rename({v: k for k, v in coords.items() if v in data})

        #data = data.set_coords(coords)

        if not np.issubdtype(data['time'].dtype, np.datetime64):
            try:
                from xarray.coding.times import cftime_to_nptime
                data['time'] = cftime_to_nptime(data['time'])
            except ValueError:
                pass

        variables = self.metadata.get('variables', None)
        if variables:
            data = data.get([v for v in variables.values() if v in data])
            data = data.rename({v: k for k, v in variables.items() if v in data})

        coords = self.metadata.get('coords', None)

        if coords:
            data = data.set_coords(coords)

        return data

    def read(self):
        """Return all the data in memory in one in-memory container.

        :return: data in memory in an xarray container
         """
        self._load_metadata()
        return xr.combine_by_coords([self.read_partition(i) for i in range(self.npartitions)])
