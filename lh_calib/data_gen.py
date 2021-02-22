import itertools
import keras
import numpy as np
import pandas as pd
import skimage
import logging
import time


def crop_image(image, size):
    if image.shape[:2] == (size, size):
        return image
    offset0 = (image.shape[0] - size[0]) // 2
    offset1 = (image.shape[1] - size[1]) // 2
    cropped = image[ offset0:offset0 + size[0], offset1:offset1 + size[1], : ]
    return cropped


class BaseProcessor:
    def __call__(self, xs):
        """ Returns generator (or iterable) with same length
        """
        return xs

class ListProcessor(BaseProcessor):
    def __init__(self, procs=None):
        if procs is None:
            procs = []
        self._procs = procs

    def append(self, proc):
        self._procs.append(proc)

    def __call__(self, xs):
        for proc in self._procs:
            xs = proc(xs)
        return xs

class LambdaProcessor(BaseProcessor):
    def __init__(self, fn):
        self._fn = fn

    def __call__(self, xs):
        return map(self._fn, xs)


def get_image_cropper(size):
    assert len(size) == 2
    return LambdaProcessor(lambda x: crop_image(x, size))


class KerasImageAugmentor(BaseProcessor):
    def __init__(self, augment_opts):
        self._image_data_gen = keras.preprocessing.image.ImageDataGenerator(**augment_opts)   # assuming keras.ImageDataGenerator

    def __call__(self, imgs):
        imgs = np.asarray(list(imgs))   # required to be ndarray
        size = len(imgs)
        imgs = next(self._image_data_gen.flow(imgs, batch_size=size, shuffle=False))
        return imgs


class LHGenerator(keras.utils.Sequence):
    def __init__(self, x, hists=None, processor=None, batch_size=128, shuffle=False, use_prob=False, use_weight=False, weight_norm_mode=None):
        """
        x: (N, ...) dim input
        hists: (N, K) dim integer arrray
        """
        self.use_prob = use_prob
        self.use_weight = use_weight
        self.batch_size = batch_size
        self.processor = processor   # assuming data agumentation applied on x
        self._shuffle = shuffle
        if isinstance(x, pd.Series):
            x = x.values
        self.x = x
        self._indexes = np.asarray(list(range(len(x))))   # saving instance order
        self.hists = None
        self.probs = None
        if hists is not None:
            if isinstance(hists, pd.Series):
                hists = hists.values
            assert len(x) == len(hists), '#x: {} != #hists: {}'.format(len(x), len(hists))
            self.hists = np.asarray(list(hists)).astype('float32')
            assert len(self.hists.shape) == 2, 'histogram should have dimension 2: {}'.format(hists)
            self._n = self.hists.sum(axis=1)
            self.probs = self.hists / self._n[:, None]
        if use_weight:
            assert hists is not None
            self._weights = self._n.astype('float32')
            self._mean_weight = self._weights.mean()
        self._weight_norm_mode = weight_norm_mode
        self._update_samples()

    def dump_all(self):
        """
        Returns: x, y or x, y, w
        when bool(self.use_weight) == True, returns weight or 
        """
        if self._weight_norm_mode == 'batch_mean':
            raise Exception('Cannot call when weight norm mode is batch_mean')
        all_data = [self[i] for i in range(len(self))]
        x = np.concatenate([batch[0] for batch in all_data])
        if self.hists is None:
            return x
        y = np.concatenate([batch[1] for batch in all_data])
        if self.use_weight:
            w = np.concatenate([batch[2] for batch in all_data])
            return x, y, w
        else:
            return x, y

    def _normalize_weights(self, weights):
        if self._weight_norm_mode is None:
            return weights
        if self._weight_norm_mode == 'batch_mean':
            return weights / weights.sum() * weights.size
        if self._weight_norm_mode == 'mean':
            return weights / self._mean_weight
        raise NotImplementedError

    def __len__(self):
        'The number of batches per epoch'
        l = int(np.ceil(len(self._indexes) / self.batch_size))
        return l

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self._indexes[index*self.batch_size:(index+1)*self.batch_size]
        x = self.x[indexes]
        if self.processor:
            x = list(self.processor(x))

        x = np.asarray(x)
        if self.hists is None:
           return x
        if self.use_prob:
            y = self.probs[indexes]
        else:
            y = self.hists[indexes]
        if self.use_weight:
            w = self._weights[indexes]
            w = self._normalize_weights(w)
            return x, y, w
        else:
            return x, y

    def _update_samples(self):
        if self._shuffle:
            np.random.shuffle(self._indexes)

    def on_epoch_end(self):
        'Updates indexes at each epoch'
        self._update_samples()
