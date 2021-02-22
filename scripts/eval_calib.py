import sys
import os
sys.path.insert(0, os.path.abspath(__file__).rsplit('/', 2)[0])
from argtools import command, argument
import logging
import numpy as np
import pandas as pd
from lh_calib import predictions
from lh_calib import evaluation
import tqdm


@command
@argument('-i', '--input', required=True)
@argument.exclusive(
    argument('-p', '--paths', nargs='+'),   # assuming a part of input table
    argument('-f', '--path-file'),
    argument('-pt', '--path-table'),
)
@argument('-wo2', '--without-order2',  dest='use_order2', action='store_false', default=True)
@argument('-ue', '--use_exact', action='store_true')
@argument('-o', '--outfile', default='/dev/stdout')
def eval_calibration(args):
    """
    - input table must include following columns:
        id, counts, total, partition
    - scores table must include following columns:
        id, one of (prob, mc_prob, alpha, mc_alpha)
    """
    input_tab = pd.read_csv(args.input, sep='\t')
    input_tab = input_tab[input_tab['partition'] == 'test']
    results = []
    if args.path_file:
        with open(args.path_file) as fp:
            paths = [line.strip() for line in fp]
    elif args.path_table:
        path_table = pd.read_csv(args.path_table, sep='\t')
        paths = path_table['path'].values
    else:
        paths = list(args.paths)
    logging.info('Evaluating %s files', len(paths))
    if args.use_exact:   # for simulation
        K = len(input_tab['hist'].iloc[0].split(','))
        K_ident = np.identity(K)
        def get_probs(row):
            ys = list(map(int, row['_org_y'].split(',')))
            ratio = row['_ratio']
            if len(ys) == 1:
                return K_ident[ys[0]]
            if len(ys) == 2:
                return K_ident[ys[0]] * ratio + K_ident[ys[1]] * (1 - ratio)
            raise NotImplementedError

        org_hists = input_tab.apply(get_probs, axis=1)
        org_hists.index = input_tab['id'].values
    else:
        org_hists = pd.Series(input_tab['hist'].values, index=input_tab['id'].values)
        org_hists = pd.Series([np.asarray(list(map(int, v.split(',')))) for v in org_hists], index=org_hists.index)

    for path in tqdm.tqdm(paths):
        try:
            logging.info(path)
            pred = predictions.load(path)
            hists = np.asarray(list(org_hists.loc[pred.ids].values))
            if args.use_exact:
                lh = evaluation.InfLabelHist(hists)
            else:
                lh = evaluation.LabelHist(hists)
            lh_eval = evaluation.LHEval(lh, pred)
            result = lh_eval.summarize(use_order2=args.use_order2)
        except Exception as e:
            logging.error('%s', e)
            #raise
            result = pd.DataFrame({'path': [path]})  # empty result
        logging.info(result)
        results.append(result)
    tab = pd.concat(results, axis=0, sort=False)
    logging.info('saving %s records to %s from %s paths', len(tab), args.outfile, len(paths))
    tab['path'] = paths
    tab.to_csv(args.outfile, index=False, float_format='%.6g', sep='\t')


if __name__ == '__main__':
    command.run()
