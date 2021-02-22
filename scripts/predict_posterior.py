import sys
import os
sys.path.insert(0, os.path.abspath(__file__).rsplit('/', 2)[0])
from argtools import command, argument
import logging
import lh_calib
from lh_calib import data_models
from lh_calib import predictions
import numpy as np


@command
@argument('-i', '--input', required=True)   # new labels
@argument('-pr', '--prior', required=True)   # prior prediction
@argument('-o', '--output', default='/dev/stdout')
@argument('-p', '--partition')
def predict_posterior(args):
    """
    """
    table = data_models.LHTable.load(args.input)
    logging.info('partition: %s', args.partition)
    if args.partition:
        table = table.get_partition(args.partition)
    hists = table.get_hists()   # Series
    hists = np.asarray(list(hists)).astype('float32')   # Series to 2d ndarray
    logging.info('hist shape: %s', hists.shape)

    hist_ids = table.id

    prior_pred = predictions.load(args.prior)
    logging.info('prior type: %s', prior_pred.__class__)

    assert (prior_pred.ids == hist_ids).all() # validation for dataset matching
    posterior = prior_pred.get_posterior(hists)
    posterior.save(args.output)


if __name__ == '__main__':
    command.run()
