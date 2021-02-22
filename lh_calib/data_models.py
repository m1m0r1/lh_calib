import logging
import numpy as np
import pandas as pd
import skimage

def get_image_from_table(table, data_source=None, pool=None):
    raise NotImplementedError


def _task(arg):
    self, idx = arg
    return self._get_image(idx)


class ImageArray:
    def __init__(self, paths, pool=None, memory=True):
        self.paths = list(paths)  # do not use as pd.Series
        self._pool = pool

        self._image_cache = None
        if memory:
            self._image_cache = skimage.io.imread_collection(self.paths, conserve_memory=False)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self._get_image(index)
        if isinstance(index, (list, np.ndarray)):
            return self._get_images(index)
        raise NotImplementedError

    def _get_images(self, indexes):
        #X = np.asarray([self._get_image(self._tab.iloc[idx]['path']) for idx in indexes])
        if self._pool is not None:
            X = self._pool.map(_task, ((self, idx) for idx in indexes), chunksize=50)
        else:
            X = (self._get_image(idx) for idx in indexes)
        return list(X)

    def __len__(self):
        return len(self.paths)

    def _get_image(self, idx):
        if self._image_cache:
            for _ in range(10):
                try:
                    img = self._image_cache[idx]
                    break
                except IndexError as e:
                    raise e
                except Exception as e:
                    logging.warning('image_cache_error: %s (%s)', e, e.__class__)
                    time.sleep(0.1 * _)
        else:
            path = self.paths[idx] #self._tab.iloc[idx]['path']
            img = io.imread(path)

        #logging.info(img.shape)
        return img


class DataSource:
    def __init__(self, table):
        self._table = table

class MixedDataSource(DataSource):
    def _generate_images(self, x): # (N, H, W, C)
        new_xs = []
        for i, row in self._table.iterrows():  # TODO optimize
            if row._mixed:
                id1, id2 = row._org_id.split(',')
                new_xs.append(x[int(id1)] * row._ratio + x[int(id2)] * (1 - row._ratio))
            else:
                new_xs.append(x[int(row._org_id)])
        return np.asarray(new_xs)

class MixedMNIST(MixedDataSource):
    def get_images(self):
        from keras.datasets import mnist
        (x_train, _), (x_test, _) = mnist.load_data()
        logging.info('Loaded mnist %s + %s entries', len(x_train), len(x_test))
        x = np.concatenate([x_train, x_test]).reshape(-1, 28, 28, 1)
        x = self._generate_images(x)
        logging.info('Selected %s entries with shape=%s', len(x), x.shape[1:])
        return x

class MixedCIFAR10(MixedDataSource):
    def get_images(self):
        from keras.datasets import cifar10
        (x_train, _), (x_test, _) = cifar10.load_data()
        logging.info('Loaded cifar10 %s + %s entries', len(x_train), len(x_test))
        x = np.concatenate([x_train, x_test]).reshape(-1, 32, 32, 3)
        x = self._generate_images(x)
        logging.info('Selected %s entries with shape=%s', len(x), x.shape[1:])
        return x

def _get_data_sources(lh_table):
    return {
        'mixed_mnist': MixedMNIST(lh_table),
        'mixed_cifar10': MixedCIFAR10(lh_table),
    }


class LHTable:
    """
    Assumed columsn
    - id
    - partition
    - hist (comma separated integers)
    - (path)  for image data
    """
    def __init__(self, table):
        self._table = table
        self._data_sources = _get_data_sources(table)

    @property
    def table(self):
        return self._table

    def __add__(self, other):
        table = pd.concatenate([self._table, other.table], axis=0, sort=False, ignore_index=True)
        return self.__class__(table)

    def split(self, ratio):
        assert 0 <= ratio <= 1
        len1 = int(np.round(len(self.table) * ratio))
        tab1 = self.__class__(self.table.iloc[:len1])
        tab2 = self.__class__(self.table.iloc[len1:])
        return tab1, tab2

    def get_partition(self, partition, max_entry=None):
        tab = self._table[self._table['partition'] == partition]
        if max_entry:
            tab = tab.iloc[:max_entry]
        return self.__class__(tab)

    @classmethod
    def load(cls, path):
        tab = pd.read_csv(path, sep='\t')
        return cls(tab)

    def __len__(self):
        return len(self.table)

    @property
    def id(self):
        return self.table['id']

    ids = id

    def get_hists(self):
        return self.table['hist'].map(lambda x: np.asarray(list(map(float, x.split(',')))))

    def get_images(self, data_source=None, pool=None):
        if data_source is None:
            return ImageArray(self._table['path'], pool=pool)

        return self._data_sources[data_source].get_images()
