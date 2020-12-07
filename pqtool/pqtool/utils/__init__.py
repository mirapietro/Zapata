import xarray as xr
import numpy as np


def metrics(data):
    result = xr.Dataset()

    for name, var in data.variables.items():
        if not name.startswith('model_') and 'model_%s' % name in data.variables:
            bias = data['model_%s' % name] - var
            result['%s_bias' % name] = bias.mean(dim='obs')
            result['%s_rmse' % name] = np.sqrt((bias ** 2.).mean(dim='obs'))
            result['%s_nobs' % name] = bias.count(dim='obs')

    return result

