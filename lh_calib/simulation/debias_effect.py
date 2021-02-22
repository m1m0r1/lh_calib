import logging
import pandas as pd
import numpy as np
import scipy
from argtools import command, argument
from tqdm import tqdm
import sys
sys.path.insert(0, __file__.rsplit('/', 3)[0])
from lh_calib import evaluation
from lh_calib import predictions
import itertools

def create_probs(N, alpha=1., K=2):
    p_mat = np.random.dirichlet(np.ones(K) * alpha, N)
    return p_mat

def create_counts(p_mat, n=2):
    count_mat = np.asarray([np.random.multinomial(n, pvals) for pvals in p_mat])   # (N, K)
    return count_mat

def scale_logit(p_mat, t=1.):   # (N, K) -> (N, K)
    p_logit = np.log(p_mat)    # (N, K)
    p_unnorm = np.exp(p_logit / t)   # (N, K)
    p_mat = p_unnorm / p_unnorm.sum(axis=1, keepdims=True)   # (N, K)
    return p_mat

def apply_blur(p_mat, rho=0.):  # (N, K) -> (N, K)
    # rho: 0 - 1
    alpha0 = 1 / (1 - rho)
    return np.asarray([np.random.dirichlet(p_vec) for p_vec in p_mat * alpha0])  # (N, K)  <- precision parameter


def _gen_and_eval_basic(N, n, K=2, alpha=1.):
    probs = create_probs(N, alpha=alpha, K=K)
    counts = create_counts(probs, n=n)
    ids = np.arange(len(probs))
    pred = predictions.ProbPrediction(probs, ids)
    label_hist = evaluation.LabelHist(counts)
    lh_eval = evaluation.LHEval(label_hist, pred)
    return lh_eval


@command.add_sub
@argument('-N', '--ninstance', nargs='+', required=True, type=int)
@argument('-n', '--nrater', nargs='+', type=int, default=[2])
@argument('-r', '--nrep', type=int, default=5)
@argument('-s', '--start-seed', type=int, default=0)
@argument('-o', '--output', default='/dev/stdout')
def gen_and_eval_basic(args):
    binning_scheme = evaluation.EquallySpacedBinningScheme(15)
    loop = list(itertools.product(range(args.nrep), args.nrater, args.ninstance))

    seed = args.start_seed
    K = 2
    recs = [] 
    for rep, n, N in tqdm(loop):
        np.random.seed(seed)
        lh_eval = _gen_and_eval_basic(N=N, n=n, K=K)
        result_tab, binnings = lh_eval.get_cpe_decomp(binning_scheme)
        ps = lh_eval.get_prob_score()
        # Series.sum ignores nan by default
        recs.append({
            'seed': seed, 'rep': rep, 'n': n, 'N': N, 'K': K,
            'ps': ps,
            'el_plugin': result_tab['el_plugin'].sum(),
            'el_bias': result_tab['el_bias'].sum(),
            'cl_plugin': result_tab['cl_plugin'].sum(),
            'cl_bias': result_tab['cl_bias'].sum(),
            'dl_plugin': result_tab['dl_plugin'].sum(),
            'dl_bias': result_tab['dl_bias'].sum(),
        })
        seed += 1
    columns = 'seed rep n N K ps el_plugin el_bias cl_plugin cl_bias dl_plugin dl_bias'.split(' ')
    tab = pd.DataFrame.from_records(recs, columns=columns)
    tab.to_csv(args.output, index=False)

if __name__ == '__main__':
    command.run()
