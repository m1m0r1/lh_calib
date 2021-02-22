import logging
import numpy as np
import pandas as pd
import scipy.special
import scipy.stats


def encode_array(vals, sep=',', fmt='{:.6g}'):
    return sep.join(map(fmt.format, vals))

def decode_array(vals, sep=','):
    return np.asarray(list(map(float, vals.split(','))))

def encode_matrix(vals, sep1=',', sep2=';', fmt='{:.6g}'):
    return sep2.join(encode_array(vals1, sep=sep1, fmt=fmt) for vals1 in vals)

def decode_matrix(vals, sep1=',', sep2=';'):
    return np.asarray([decode_array(vals1, sep=sep1) for vals1 in vals.split(';')])


def load(path):
    cands = [
        MCAlphaPrediction,
        AlphaPrediction,
        WMCProbPrediction,
        MCProbPrediction,
        ProbPrediction,
    ]
    errors = []
    for cls in cands:
        try:
            return cls.load(path)
        except KeyError as e:
            errors.append(e)
    for e in errors:
        logging.error(e)
    raise NotImplementedError


class Prediction:
    @property
    def ids(self):
        return self._ids

    def get_probs(self):   # (N, K)
        return self._probs

    @classmethod
    def load(cls, path):
        raise NotImplementedError

    def save(self, path, ids):
        raise NotImplementedError

    def get_posterior(self, hists):
        raise NotImplementedError


def hist_likelihood(hists, probs):   # (..., K), (..., K) -> (...,)
    return (probs ** hists).sum(axis=-1)


def get_posterior_dirichlet0(hists, alpha0=1.):
    K = hists.shape[1] # (N, K)
    alpha = alpha0 * np.ones(K) / K
    post_alpha = hists + alpha[:, None]
    return AlphaPrediction(post_alpha, pred.ids)

def get_posterior_dirichlet(pred, hists, alpha0=1.):
    probs = pred.get_probs()
    alpha = alpha0 * probs
    assert hists.shape == probs.shape # (N, K)
    post_alpha = hists + alpha
    return AlphaPrediction(post_alpha, pred.ids)


class ProbPrediction(Prediction):
    def __init__(self, probs, ids):
        self._probs = np.asarray(probs)    # (N, K)
        assert len(self._probs.shape) == 2
        self._ids = ids

    def get_agreement_probs(self):  # (N,)
        return (self._probs ** 2).sum(axis=1)

    @classmethod
    def load(cls, path):
        tab = pd.read_csv(path, sep='\t')
        probs = np.asarray(list(map(decode_array, tab['prob'])))
        return cls(probs, tab['id'])

    def save(self, path):
        columns = ['id', 'prob']
        tab = pd.DataFrame({
            'id': self._ids,
            'prob': list(map(encode_array, self._probs)),
        }, columns=columns)
        tab.to_csv(path, sep='\t', index=False)


class MCProbPrediction(Prediction):
    def __init__(self, mc_probs, ids):
        self._mc_probs = np.asarray(mc_probs)  # (N, S, K)
        assert len(self._mc_probs.shape) == 3
        self._probs = self._mc_probs.mean(axis=1)  # (N, K)
        self._ids = ids

    def get_agreement_probs(self):  # (N,)
        mc_agree_probs = (self._mc_probs ** 2).sum(axis=2)  # (N, S)
        return mc_agree_probs.mean(axis=1)

    @classmethod
    def load(cls, path):
        tab = pd.read_csv(path, sep='\t')
        mc_probs = np.asarray(list(map(decode_matrix, tab['mc_prob'])))
        return cls(mc_probs, tab['id'])

    def save(self, path):
        columns = ['id', 'mc_prob']
        tab = pd.DataFrame({
            'id': self._ids,
            'mc_prob': list(map(encode_matrix, self._mc_probs)),
        }, columns=columns)
        tab.to_csv(path, sep='\t', index=False)

    def get_posterior(self, hists):
        hl = hist_likelihood(hists[:, None, :], self._mc_probs)   # (N, S, K) -> (N, S)
        weights = hl / hl.sum(axis=-1, keepdims=True) # normalized -> (N, S)
        logging.info(weights)

        wmc_pred = WMCProbPrediction(self._mc_probs, weights, ids=self.ids)   # (N, S, K), (N, S)
        return wmc_pred


class WMCProbPrediction(Prediction):
    def __init__(self, mc_probs, mc_weights, ids):
        self._mc_probs = np.asarray(mc_probs)  # (N, S, K)
        self._mc_weights = np.asarray(mc_weights)   # (N, S) or (1, S)
        assert len(self._mc_probs.shape) == 3
        assert self._mc_weights.shape == self._mc_probs.shape[:2]
        self._probs = (self._mc_probs * self._mc_weights[:, :, None]).sum(axis=1)  # (N, K)
        self._ids = ids

    @classmethod
    def load(cls, path):
        tab = pd.read_csv(path, sep='\t')
        mc_probs = np.asarray(list(map(decode_matrix, tab['mc_prob'])))
        mc_weights = np.asarray(list(map(decode_array, tab['mc_weight'])))
        return cls(mc_probs, mc_weights, tab['id'])

    def save(self, path):
        columns = ['id', 'mc_prob', 'mc_weight']
        tab = pd.DataFrame({
            'id': self._ids,
            'mc_prob': list(map(encode_matrix, self._mc_probs)),
            'mc_weight': list(map(encode_array, self._mc_weights)),
        }, columns=columns)
        tab.to_csv(path, sep='\t', index=False)


class AlphaPrediction(Prediction):
    eps = clip_min = np.finfo(float).eps
    clip_max = 1./np.finfo(float).eps

    def __init__(self, alphas, ids):
        self._alphas = np.asarray(alphas)    # (N, K)
        self._alphas[np.isnan(self._alphas)] = self.clip_min   # Repair underflowed values
        self._alphas = np.clip(self._alphas, self.clip_min, self.clip_max)
        assert len(self._alphas.shape) == 2
        self._alpha0s = self._alphas.sum(axis=1)
        self._probs = self._alphas / self._alpha0s[:,None]
        self._ids = ids

    def get_alphas(self):
        return self._alphas

    def get_agreement_probs(self):  # (N,)
        denom = self._alpha0s * (self._alpha0s + 1)
        square_moments = self._alphas * (self._alphas + 1) / denom[:, None]  # (N, K)
        agree_probs = square_moments.sum(axis=1)   # (N,)
        return agree_probs

    @classmethod
    def load(cls, path):
        tab = pd.read_csv(path, sep='\t')
        alphas = np.asarray(list(map(decode_array, tab['alpha'])))
        return cls(alphas, tab['id'])

    def save(self, path):
        columns = ['id', 'alpha']
        tab = pd.DataFrame({
            'id': self._ids,
            'alpha': list(map(encode_array, self._alphas)),
        }, columns=columns)
        tab.to_csv(path, sep='\t', index=False)

    def get_posterior(self, hists):
        alpha = self._alphas
        assert hists.shape == alpha.shape # (N, K)
        post_alpha = hists + alpha
        return AlphaPrediction(post_alpha, self.ids)


class MCAlphaPrediction(Prediction):
    eps = clip_min = np.finfo(float).eps
    clip_max = 1./np.finfo(float).eps

    def __init__(self, mc_alphas, ids):
        self._mc_alphas = np.asarray(mc_alphas)  # (N, S, K)
        self._mc_alphas[np.isnan(self._mc_alphas)] = self.clip_min   # repair underflowed values
        self._mc_alphas = np.clip(self._mc_alphas, self.clip_min, self.clip_max)
        assert len(self._mc_alphas.shape) == 3
        self._alphas = self._mc_alphas.mean(axis=1)  # (N, K)
        self._mc_alpha0s = self._mc_alphas.sum(axis=2)  # (N, S)
        self._mc_mean_probs = self._mc_alphas / self._mc_alpha0s[:, :, None]  #(N, S, K)
        self._probs = self._mc_mean_probs.mean(axis=1) #(N, K)
        self._ids = ids

    def get_alphas(self):
        return self._alphas

    def get_agreement_probs(self):  # (N,)
        mc_square_moments = self._mc_alphas * (self._mc_alphas + 1) / (self._mc_alpha0s * (self._mc_alpha0s + 1))[:, :, None]  # (N, S, K)
        mc_agree_probs = mc_square_moments.sum(axis=2)   # (N, S)
        return mc_agree_probs.mean(axis=1)

    @classmethod
    def load(cls, path):
        tab = pd.read_csv(path, sep='\t')
        mc_alphas = np.asarray(list(map(decode_matrix, tab['mc_alpha'])))
        return cls(mc_alphas, tab['id'])

    def save(self, path):
        columns = ['id', 'mc_alpha']
        tab = pd.DataFrame({
            'id': self._ids,
            'mc_alpha': list(map(encode_matrix, self._mc_alphas)),
        }, columns=columns)
        tab.to_csv(path, sep='\t', index=False)
