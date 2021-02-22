import logging
import numpy as np
import numpy.ma as ma
import pandas as pd
import scipy.special as ssp
from . import predictions


class LabelHist:
    def __init__(self, counts):
        assert len(counts.shape) == 2
        self._counts = np.asarray(counts)   # (N, K)
        self._totals = counts.sum(axis=1)   # (N,)
        self._probs = self._counts / self._totals[:, None]

    def __len__(self):
        return len(self._totals)

    @property
    def nclass(self):
        return self._probs.shape[1]

    def get_counts(self):
        return self._counts   # (N, K)

    def get_totals(self):
        return self._totals   # (N,)

    def estimate_probs(self):   # (N, K)
        return self._probs

    def estimate_gini_indexes(self):   # (N,)
        return 1 - self.estimate_agreement_probs()

    def estimate_agreement_probs(self):
        total = self._totals   # (N,)
        pair_total = total * (total - 1) / 2   # (N,)
        denom = ma.array(pair_total, mask=pair_total==0)   # masked pair_total
        class_pair_counts = self._counts * (self._counts - 1) / 2   # (N, K)
        agreed_ratios = class_pair_counts.sum(axis=1) / denom   # (N,)
        return agreed_ratios


class InfLabelHist(LabelHist):
    def __init__(self, probs):
        assert len(probs.shape) == 2
        self._probs = np.asarray(probs)   # (N, K)
        self._totals = np.inf * np.ones(probs.shape[0])

    def get_counts(self):
        raise NotImplementedError

    def estimate_agreement_probs(self):
        return (self._probs ** 2).sum(axis=1)

    def estimate_class_rank_probs(self, rank):   # (N, K, rank+1)
        raise NotImplementedError


def get_lh_brier_scores(lh_probs, probs):  # (N,), (N,) -> (N,)
    return (lh_probs * (1 - lh_probs) + (lh_probs - probs)**2)

def get_lh_prob_scores(lh_probs, probs):  # (N, K), (N, K) -> (N,)
    return (lh_probs * (1 - lh_probs) + (lh_probs - probs)**2).sum(axis=1)


class Binning:   # binning information
    def __init__(self, closed_anchors):
        self._closed_anchors = np.asarray(closed_anchors)

    def __len__(self):
        return len(self._closed_anchors) - 1

    @property
    def starts(self):
        return self._closed_anchors[:-1]

    @property
    def ends(self):
        return self._closed_anchors[1:]

    def get_labels(self, format='.1f'):
        starts = self._closed_anchors[:-1]
        ends = self._closed_anchors[1:]
        labels = ['[{:{fmt}}, {:{fmt}})'.format(s, e, fmt=format) for s, e in zip(starts, ends)]
        labels[-1] = '[{:{fmt}}, {:{fmt}}]'.format(starts[-1], ends[-1], fmt=format)  # modify last
        return labels


class EquallySpacedBinningScheme:
    """
    >>> esb = EquallySpacedBinningScheme(10)
    >>> index, binning = esb([0. - 1e-15, 0., 0.05, 0.1, 0.15, 0.2, 0.5, 0.9, 0.99, 1.00001])
    >>> len(binning)
    10
    >>> list(index)
    [0, 0, 0, 1, 1, 2, 5, 9, 9, 9]
    >>> binning.get_labels()[0]
    '[0.0, 0.1)'
    >>> binning.get_labels()[1]
    '[0.1, 0.2)'
    >>> binning.get_labels()[-2]
    '[0.8, 0.9)'
    >>> binning.get_labels()[-1]
    '[0.9, 1.0]'
    """
    def __init__(self, n_partition, start=0., end=1.):  # left included
        self._outer_anchors = np.linspace(start, end, n_partition + 1)
        self._start = 0
        self._end = end
        self._mid_anchors = self._outer_anchors[1:-1]
        self._n_partition = n_partition

    def __call__(self, val):  # (N,), returns index of bins
        indexes = np.digitize(val, self._mid_anchors)
        return indexes, Binning(self._outer_anchors)


# probs should be unbiased
def estimate_calibration_error(probs, preds, binning_scheme):
    N = len(probs)
    assert len(probs) == len(preds) == N
    bin_index, binning = binning_scheme(preds)
    bin_indexes = group_array(np.arange(N), bin_index)

    bins = list(range(len(binning)))
    bin_cls = []
    bin_cl_biases = []

    for idx in bins:
        idxs = bin_indexes.get(idx, [])
        if len(idxs):
            cl = np.square(probs[idxs].mean() - preds[idxs].mean()) * len(idxs) / N
            cl_bias = probs[idxs].var() * (len(idxs) / (len(idxs) - 1)) / N \
                    if len(idxs) > 1 else np.nan  # not available
            bin_cls.append(cl)
            bin_cl_biases.append(cl_bias)
        else:
            bin_cls.append(0.)
            bin_cl_biases.append(0.)

    bin_cls = np.asarray(bin_cls)
    bin_cl_biases = ma.array(bin_cl_biases, mask=np.isnan(bin_cl_biases))
    cl = bin_cls.sum()
    cl_bias = bin_cl_biases.sum()
    cl_debias = cl - cl_bias
    return {
            'cl_plugin': cl,
            'cl_bias': cl_bias,
            'cl_debias': cl_debias,
            'bin_cl_plugins': bin_cls,
            'bin_cl_bias': bin_cl_biases,
            'binning': binning,
    }


class LHEval:
    def __init__(self, label_hist, pred):
        self._lh = label_hist
        self._pred = pred
        #assert self._lh.nclass == self.pred.nclass

    def get_accuracy(self):
        ref = self._lh.estimate_probs().argmax(axis=1)
        pred = self._pred.get_probs().argmax(axis=1)
        #return (ref == pred).mean()
        #np.random.shuffle(pred)
        return (ref == pred).mean()

    def get_prob_score(self):  # unbiased
        lh_probs = self._lh.estimate_probs()   # (N, K)
        probs = self._pred.get_probs()   # (N, K)
        scores = get_lh_prob_scores(lh_probs, probs)   # (N,)
        return scores.mean()   # instance-wise mean

    def get_cpe_decomp(self, binning_scheme):
        tabs = []
        binnings = []
        for k in range(self._lh.nclass):
            tab, binning = self.get_cpe_decomp1(binning_scheme, k)
            tab.insert(0, 'k', k)
            tabs.append(tab)
            binnings.append(binning)
        tab = pd.concat(tabs, axis=0, sort=False, ignore_index=True)
        return tab, binnings

    def get_disagreement_calib_loss(self, binning_scheme):
        mu = self._lh.estimate_agreement_probs()
        phi = self._pred.get_agreement_probs()
        return estimate_calibration_error(mu, phi, binning_scheme)

    def get_cpe_decomp1(self, binning_scheme, k):
        lh_probs = self._lh.estimate_probs()[:, k]
        lh_totals = self._lh.get_totals()
        probs = self._pred.get_probs()[:, k]
        N = len(probs)

        bin_index, binning = binning_scheme(probs)
        bin_indexes = group_array(np.arange(N), bin_index)

        # EL terms can be calculate instance-wise
        el_terms = (lh_probs - probs) ** 2 / N
        el_bias_denom = lh_totals - 1
        el_bias_terms = lh_probs * (1 - lh_probs) / ma.array(el_bias_denom, mask=el_bias_denom == 0) / N

        bins = list(range(len(binning)))
        bin_counts = []
        bin_els = []
        bin_el_biases = []
        bin_cls = []
        bin_cl_biases = []

        for idx in bins:
            idxs = bin_indexes.get(idx, [])
            bin_counts.append(len(idxs))
            bin_els.append(el_terms[idxs].sum())
            bin_el_biases.append(el_bias_terms[idxs].sum())
            if len(idxs):
                cl = np.square(lh_probs[idxs].mean() - probs[idxs].mean()) * len(idxs) / N
                cl_bias = lh_probs[idxs].var() * (len(idxs) / (len(idxs) - 1)) / N \
                        if len(idxs) > 1 else np.nan  # not available
                bin_cls.append(cl)
                bin_cl_biases.append(cl_bias)
            else:
                bin_cls.append(0.)
                bin_cl_biases.append(0.)

        bin_dls = np.asarray(bin_els) - np.asarray(bin_cls)
        bin_dl_biases = np.asarray(bin_el_biases) - np.asarray(bin_cl_biases)
        tab = pd.DataFrame({
            'bin': bins,
            'bin_start': binning.starts,
            'bin_end': binning.ends,
            'bin_count': bin_counts,
            'el_plugin': np.asarray(bin_els),
            'el_bias': np.asarray(bin_el_biases),
            'cl_plugin': np.asarray(bin_cls),
            'cl_bias': np.asarray(bin_cl_biases),
            'dl_plugin': np.asarray(bin_dls),
            'dl_bias': np.asarray(bin_dl_biases),
        })
        return tab, binning

    def get_disagreement_brier_score(self):   #unbiased
        mu = self._lh.estimate_agreement_probs()
        phi = self._pred.get_agreement_probs()
        return get_lh_brier_scores(mu, phi).mean()   # instance-wise mean

    def summarize(self, use_order2=True):
        bin_count = 15
        binning_scheme = EquallySpacedBinningScheme(bin_count)

        data = {
            'accuracy': [self.get_accuracy()],
            'ps':  [self.get_prob_score()],
            'bin_count': [bin_count],
            'nclass': [self._lh.nclass],
        }
        # CPE decomposition
        cpe_decomp, binnings = self.get_cpe_decomp(binning_scheme)
        el_plugin = cpe_decomp['el_plugin'].sum()
        el_debias = el_plugin - cpe_decomp['el_bias'].sum()
        cl_plugin = cpe_decomp['cl_plugin'].sum()
        cl_debias = cl_plugin - cpe_decomp['cl_bias'].sum()
        dl_plugin = cpe_decomp['dl_plugin'].sum()
        dl_debias = dl_plugin - cpe_decomp['dl_bias'].sum()
        data.update(
            el_plugin=[el_plugin],
            cl_plugin=[cl_plugin],
            dl_plugin=[dl_plugin],
            el_debias=[el_debias],
            cl_debias=[cl_debias],
            dl_debias=[dl_debias],
        )
        if not use_order2:
            return pd.DataFrame(data)

        # BS for disagree
        data.update({
            'd_bs': [self.get_disagreement_brier_score()],  # agreement
        })
        # CL for disagree
        rec = self.get_disagreement_calib_loss(binning_scheme)
        d_cl_plugin = rec['cl_plugin']
        d_cl_debias = rec['cl_debias']
        data.update({
            'd_cl_plugin': [d_cl_plugin],
            'd_cl_debias': [d_cl_debias],
        })
        return pd.DataFrame(data)


def group_array(values, indexes):
    """
    >>> a = np.asarray([0, 1, 2, 3, 4])
    >>> b = np.asarray([0, 2, 0, 2, 0])
    >>> groups = group_array(a, b)
    >>> list(groups[0])
    [0, 2, 4]
    >>> list(groups[2])
    [1, 3]
    """
    assert len(values) == len(indexes)
    return {index: values[tab.index] for index, tab in 
            pd.DataFrame({'a': indexes}).groupby('a')}   # order preserved within group


def plot_cd_diagram(lh_eval, binning_scheme, k, plugin=True, debias=True):
    import matplotlib.pyplot as plt
    decomp_tab, binning = lh_eval.get_cpe_decomp1(binning_scheme, k)
    bin_labels = np.asarray(binning.get_labels(format='.2f'))
    if plugin:
        plt.plot(bin_labels, decomp_tab['cl_plugin'],
                linestyle='dotted', color='C1', label='CL (plugin)')
        plt.plot(bin_labels, decomp_tab['dl_plugin'],
                linestyle='dotted', color='C2', label='DL (plugin)')
    if debias:
        plt.plot(bin_labels, decomp_tab['cl_plugin'] - decomp_tab['cl_bias'], 
                color='C1', label='CL (debias)')
        plt.plot(bin_labels, decomp_tab['dl_plugin'] - decomp_tab['dl_bias'], 
                color='C2', label='DL (debias)')
    plt.xticks(rotation=90)
    plt.legend(loc='upper right', fontsize=12)
    plt.xlabel('Predictive Probability')
    plt.ylabel('Loss')
