#!/usr/bin/env python
import xarray as xr
import numpy as np
from pqtool.common import common


def interp(source, coords, **kwargs):
    """
    This is a more efficient version of DataArray.interp() for arrays that are
    not in memory. Where DataArray.interp() loads all the data, this loads only
    the coordinates and the nearest points.
    """
    index = {}
    kwargs['method'] = 'nearest'
    mask = np.zeros_like(len(coords['time']), dtype='bool')
    for dim, values in coords.items():

        tmp = xr.DataArray(np.arange(len(source[dim])), dims=dim)
        tmp.coords[dim] = source[dim]

        ind = tmp.interp({dim: values}, **kwargs)
        index[dim] = ind.fillna(0).astype(int)
        mask = np.logical_or(mask, np.isnan(ind))

    return source.isel(**index).where(~mask).load()


parser = common.create_parser()
args = parser.parse_args()

intermediate, model = common.preprocess(args.input, args.catalog, args.name,
                                        args.start_date, args.end_date)

# Select nearest in all coordinates except depth
nearest_coords = {k: v for k, v in intermediate.coords.items() if k not in ['depth', 'model', 'dc_reference']}
model = interp(model.rename({'depth': 'model_depth'}), nearest_coords, method='nearest')

model = common.interpolate_over_depth(intermediate, model)

common.postprocess(intermediate, model, args.name, args.output)

