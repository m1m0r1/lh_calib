import tempfile
import os
import glob
import re
import logging
import numpy as np
import pandas as pd
import keras
from keras.models import load_model
from datetime import datetime, timezone, timedelta

def get_now():
    tz = timezone(timedelta(hours=+9), 'JST')
    return datetime.now(tz)


def _find_weights(ckpt_path, paths=None, start=0):
    """ ckpt_path should be format string containing '{epoch}'

    >>> paths = ['abc.01.ckpt', 'abc.05.ckpt', 'abc.best.ckpt']
    >>> _find_weights('abc.{epoch}.ckpt', paths=paths)
    [(0, 'abc.01.ckpt'), (4, 'abc.05.ckpt')
    """
    if paths is None:
        paths = glob.glob(ckpt_path.format(epoch='*'))
    pat = ckpt_path.format(epoch='([0-9]+)')

    def get_epoch_path(path):
        m = re.match(pat, path)
        if m:
            return (int(m.groups()[0]) - 1 + start, path)

    epoch_paths = list(filter(lambda x: x, (get_epoch_path(path) for path in paths)))
    return epoch_paths

def find_last_model(ckpt_path, paths=None):
    """ ckpt_path should be format string containing '{epoch}'

    >>> paths = ['abc.01.ckpt', 'abc.05.ckpt', 'abc.best.ckpt']
    >>> find_last_model('abc.{epoch}.ckpt', paths=paths)
    (5, 'abc.05.ckpt')
    """
    epoch_paths = _find_weights(ckpt_path, paths=paths, start=1)
    if epoch_paths:
        last_one = list(sorted(epoch_paths, key=lambda x: x[0]))[-1]
        return last_one



# Ref: https://github.com/raghakot/keras-vis
def apply_modifications(model, custom_objects=None):
    """Applies modifications to the model layers to create a new Graph. For example, simply changing
    `model.layers[idx].activation = new activation` does not change the graph. The entire graph needs to be updated
    with modified inbound and outbound tensors because of change in layer building function.
    Args:
        model: The `keras.models.Model` instance.
    Returns:
        The modified model with changes applied. Does not mutate the original `model`.
    """
    # The strategy is to save the modified model and load it back. This is done because setting the activation
    # in a Keras layer doesnt actually change the graph. We have to iterate the entire graph and change the
    # layer inbound and outbound nodes with modified tensors. This is doubly complicated in Keras 2.x since
    # multiple inbound and outbound nodes are allowed with the Graph API.
    model_path = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()) + '.h5')
    model.save(model_path) #, include_optimizer=False)
    try:
        new_model = load_model(model_path, custom_objects=custom_objects, compile=False)
        return new_model
    finally:
        os.remove(model_path)


def _get_last_input(layer):
    #return layer.input
    try:
        return layer.input
    except AttributeError as e:
        logging.warning(e)
        index = len(layer._inbound_nodes) - 1   # hack
        return layer.get_input_at(index)

def _get_last_output(layer):
    #return layer.output
    try:
        return layer.output
    except AttributeError as e:
        logging.warning(e)
        index = len(layer._inbound_nodes) - 1   # hack
        return layer.get_output_at(index)

def apply_mc_dropout(model, dropout_rate=None):
    layer = model.layers[0]
    x = _get_last_input(layer)  #layer.input
    z = _get_last_output(layer)
    org_key_outputs = {}
    org_key_outputs[z.name] = z

    #print ([(k, out.name) for k, out in org_key_outputs.items()])
    for i, layer in enumerate(model.layers[1:], 1):
        # workaround for merge type layer
        if isinstance(_get_last_input(layer), list):
            z = [org_key_outputs[org_tensor.name] for org_tensor in _get_last_input(layer)]   # hack
        else:
            z = org_key_outputs[_get_last_input(layer).name]
        org_name = _get_last_output(layer).name

        if 'dropout' in layer.name:
            rate = layer.rate if dropout_rate is None else dropout_rate
            z = keras.layers.Dropout(rate=rate)(z, training=True)
        else:
            z = layer(z)   # only sequentail layer is possible

        # workaround for merge type layer
        org_key_outputs[org_name] = z
    return keras.models.Model(inputs=x, outputs=z)


class AppendMetric(keras.callbacks.Callback):
    """ Set this callback before History callbacks
    """
    def __init__(self, name, getter):
        super().__init__()
        assert callable(getter)
        self.name = name
        self._getter = getter

    def on_epoch_end(self, epoch, logs=None):
        logs[self.name] = self._getter(self)


class HistoryRecorder(keras.callbacks.History):
    def __init__(self, out_prefix, make_plot=True, initial_epoch=0):
        super().__init__()
        self._out_prefix = out_prefix
        self._table_name = '{}.txt'.format(out_prefix)
        opt_keys = ['epoch', 'ts', 'elapsed']
        if os.path.exists(self._table_name) and initial_epoch > 0:
            tab = pd.read_csv(self._table_name, sep='\t')
            tab = tab[tab['epoch'] < initial_epoch]  # Discard history after the initial_epoch
            self._opts = {k: list(tab[k]) for k in opt_keys}
            self._history = {k: list(tab[k]) for k in tab.drop(columns=opt_keys).columns}
        else:
            self._opts = {k: [] for k in opt_keys}
            self._history = {}

        self.start_ts = get_now().timestamp()
        self.table = None
        self._make_plot = make_plot

    def get(self, metric):
        return self.history.get(metric)

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs=logs)
        self.epoch.extend(self._opts['epoch'])
        self.history.update(self._history)

    def plot_metrics(self):
        keys = [key for key in self.history.keys() if not key.startswith('val_')]
        for key in keys:
            self.plot_metric(key)

    def plot_metric(self, key):
        import matplotlib.pyplot as plt
        def plot(key):
            plt.plot(self.epoch, self.history[key], label=key)

        output = '{}.{}.pdf'.format(self._out_prefix, key)
        plt.figure()
        plt.title(output)
        plot(key)
        if 'val_' + key in self.history:
            plot('val_' + key)
        plt.legend()
        plt.savefig(output)
        plt.clf()
        plt.close()

    def save(self):
        self.table.to_csv(self._table_name, sep='\t', float_format='%.6g') #'%.4e')
        if self._make_plot:
            self.plot_metrics()

    def on_epoch_end(self, epoch, logs=None):
        now = get_now().timestamp()
        self._opts['epoch'].append(epoch)
        self._opts['ts'].append(now)
        self._opts['elapsed'].append(now - self.start_ts)
        super().on_epoch_end(epoch, logs=logs)

        d = {}
        d.update(self._opts)
        d.update(self.history)
        table = pd.DataFrame(d).set_index('epoch')
        self.table = table
        self.save()


class MetricTester:
    def __init__(self, history, metric, cond):
        self.history = history
        self.metric = metric
        assert cond in ('min', 'max')
        self.cond = cond

    def is_best(self, logs):
        metric = self.metric
        hist_values = self.history.get(metric)
        value = logs and logs.get(metric)

        if value is None or hist_values is None:
            logging.warning('Metric %s was not found in (%s, %s)', metric, self.history, logs)
            return False
        if len(hist_values) == 0:
            return True
        if self.cond == 'min' and value <= np.min(hist_values):
            logging.info('Metric: %s reached minimum value: %s', metric, value)
            return True
        elif self.cond == 'max' and value >= np.max(hist_values):
            logging.info('Metric: %s reached minimum value: %s', metric, value)
            return True
        return False


class CustomModelCheckpoint(keras.callbacks.Callback):
    def __init__(self, model, path, period=1, test_fn=None, epoch_path=None):
        """ Simular but not same as keras.ModelCheckpoint
        Save weights if either of conditions is satisfied.
        - epoch is at period of save
        - test_fn(epoch, logs) is set and it returns true
        """
        super().__init__()
        self._model = model
        self.path = path
        self.epoch_path = epoch_path
        self.period = period
        self.test_fn = test_fn

    def is_save_period(self, epoch):
        if self.period < 1:
            return False
        return (epoch + 1) % self.period == 0

    def should_save(self, epoch, logs):
        if self.is_save_period(epoch):
            return True
        if self.test_fn and self.test_fn(logs):
            return True
        return False

    def on_epoch_end(self, epoch, logs=None):
        if not self.should_save(epoch, logs=logs):
            return
        epoch1 = epoch + 1   # 1-start definition
        logging.info("Saving weights to : {}".format(self.path.format(epoch=epoch1)))
        self._model.save_weights(self.path.format(epoch=epoch1), overwrite=True)
        if self.epoch_path:  # Note that epoch is saved as 0-started value here
            logging.info("Saving epoch to : {}".format(self.epoch_path.format(epoch=epoch1)))
            with open(self.epoch_path, 'w+') as fp:
                fp.write(str(epoch) + '\n')


def device_aware_call(fn, use_cpu=None):
    """ Execute function in specified device scope if required

    This is useful for building or resuming model when possibly use multiple gpus.
    See
     - https://keras.io/ja/utils/
     - https://github.com/keras-team/keras/issues/11313#issuecomment-427768441
    """
    if use_cpu:
        with tf.device('/cpu:0'):
            return fn()
    else:
        return fn()  # use default environment


class WeightInfo:
    def __init__(self, prefix):
        self.weight_path = '{}.h5'.format(prefix)
        self.epoch_path = '{}.epochs'.format(prefix)

    def find(self):
        with open(self.epoch_path) as fp:
            epoch = int(fp.read().strip())
        return epoch, self.weight_path

    def get_callback(self, model, period=1, test_fn=None):
        model_weights_cb = CustomModelCheckpoint(model, path=self.weight_path, period=period, test_fn=test_fn, epoch_path=self.epoch_path)
        return model_weights_cb


class WeightInfoEpochs:
    def __init__(self, prefix):
        self.weight_path = '{}.{{epoch}}.h5'.format(prefix)

    def find(self):
        epoch_weights = _find_weights(model_weights_path)
        if epoch_weights:
            epoch, weight = list(sorted(epoch_weights, key=lambda x: x[0]))[-1]
            return epoch, weight

    def get_callback(self, model, period=1, test_fn=None):
        model_weights_cb = CustomModelCheckpoint(model, path=self.weight_path.format(epoch='{epoch:03d}'), period=period, test_fn=test_fn)
        return model_weights_cb


class ResumableModelBuilder:
    """
    Setup model using build_fn or resume model from file.
    Use use_cpu=True in parallel gpu mode.

    weight_info: WeightInfo
    build_fn: returns Model
    """

    def __init__(self, prefix, weight_info=None, build_fn=None, custom_objects=None, use_cpu=False):
        self._build_fn = build_fn
        self.model_path = '{}.model.h5'.format(prefix)    # model after compile
        self.custom_objects = custom_objects
        self._use_cpu = use_cpu
        self._model = None
        self._initial_epoch = None
        self._weight_info = weight_info

    @property
    def model(self):
        return self._model

    @property
    def initial_epoch(self):
        return self._initial_epoch

    def model_exists(self):
        return os.path.exists(self.model_path)

    def _resume(self):
        model = load_model(self.model_path, custom_objects=self.custom_objects)  # TODO separating models and weights
        try:
            if self._weight_info is not None:
                epoch, path = self._weight_info.find()
                next_epoch = epoch + 1   # because last_epoch is 1-start and initial_epoch is 0-start index
            logging.info('Loading weights from epoch: %s (%s)', epoch, path)
            model.load_weights(path)
        except Exception as e:
            logging.warning(e)
            next_epoch = 0
        self._model = model
        self._initial_epoch = next_epoch

    def build(self):
        model = device_aware_call(self._build_fn, use_cpu=self._use_cpu)
        assert model._is_compiled, 'Model must be compiled'
        model.save(self.model_path)   # save initial model
        self._model = model
        self._initial_epoch = 0

    def resume(self):
        device_aware_call(lambda : self._resume(), use_cpu=self._use_cpu)

    def build_or_resume(self):
        if self.model_exists():
            logging.info('Model %s found', self.model_path)
            return self.resume()

        return self.build()


def get_multi_step_lr_scheduler(init_lr, warmup_epochs=None, decay_epochs=None):
    if warmup_epochs is None:
        warmup_epochs = 0
    if decay_epochs is None:
        decay_epochs = [] #[50, 100]

    def lr_scheduler(epoch):  # epoch is 0-start
        epoch1 = epoch + 1
        if epoch1 < warmup_epochs:
            offset = 1e-2
            return init_lr * (offset + (1 - offset) * epoch1 / warmup_epochs)

        lr = init_lr
        for decay_epoch in decay_epochs:
            if epoch1 >= decay_epoch:
                lr *= 0.1
        return lr

    return lr_scheduler
