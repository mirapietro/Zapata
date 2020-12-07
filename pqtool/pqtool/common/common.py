from argparse import ArgumentParser
import xarray as xr
import numpy as np
import intake
import logging

logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.INFO)
logger = logging.getLogger('model')


def create_parser():
    """A function to anable an arguments parser.

    This function creates an a parser of input parameters usefull to design script for model-obs interpolations

    :return: the arguments parser
    """
    parser = ArgumentParser(description='Interpolate model results')
    parser.add_argument('-c', '--catalog', default='catalog.yaml', help='catalog file')
    parser.add_argument('-n', '--name', required=True, help='dataset name (in catalog)')
    parser.add_argument('-s', '--start-date', help='start date')
    parser.add_argument('-e', '--end-date', help='end date')
    parser.add_argument('-i', '--input', required=True, help='input file')
    parser.add_argument('-o', '--output', required=True, help='output file')
    return parser


def preprocess(input, catalog, name, start_date, end_date):
    """Pre-process model and obs input data

    :param input: intermediate file
    :param catalog: path to the catalog file
    :param name: dataset name
    :param start_date: start date
    :param end_date: end date
    :return: model and obs in xarray containers
    """
    logger.info('Opening input file {}'.format(input))
    intermediate = xr.open_dataset(input)
    intermediate['model'] = intermediate['model'].astype(str)  # Workaround for bug in xarray

    logger.info('Opening catalog {}'.format(catalog))
    cat = intake.open_catalog(catalog)

    dataset = cat[name]
    logger.info('Dataset "{}" contains {} files'.format(name, len(dataset.files)))

    if start_date:
        dataset = dataset.subset(date=slice(start_date, None))
    if end_date:
        dataset = dataset.subset(date=slice(None, end_date))
    logger.info('Using subset of {} files'.format(len(dataset.files)))

    return intermediate, dataset.read()


def interpolate_over_depth(intermediate, model):
    """Interpolate the model in the domain of the observations

    :param intermediate: intermediate file
    :param model: model dataset
    :return: model interpolated on the obs depths
    """
    # Linearly interpolate in depth
    linear_coords = {'model_depth': intermediate.coords['depth'], 'obs': intermediate.coords['obs']}
    model = model.where(model['salinity'] != 0)  # Drop salinity == 0 values (outside grid)
    model = model.interp(linear_coords, method='linear')

    del model.coords['model_depth']
    del model['obs']

    # Rounding errors could prevent merging of the datasets, make sure depth is exact
    np.testing.assert_allclose(model.coords['depth'], intermediate.coords['depth'])
    model.coords['depth'] = intermediate.coords['depth']

    return model


def postprocess(intermediate, model, name, output):
    """Post-process the model dataset to properly save in the intermediate file

    :param intermediate: intermediate file
    :param model: model dataset
    :param name: variable name
    :param output: output filename
    :return: final netcdf file
    """
    model = model.rename({'temperature': 'model_temperature',
                          'salinity': 'model_salinity'})
    model.coords['model'] = xr.DataArray(np.array([name]), dims='model')
    model = xr.concat([model], dim='model')

    intermediate = intermediate.merge(model)

    if output:
        logger.info('Writing output dataset to {}'.format(output))
        intermediate.to_netcdf(output)
