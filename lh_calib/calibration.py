import logging
import numpy as np
import numpy.ma as ma
import pandas as pd
import tensorflow as tf
import keras
from keras import backend as K
from . import keras_utils
from .keras_utils import AppendMetric


def get_unique_name(name):
    return name + '_' + str(K.get_uid(name))

def strip_last_activation(model, custom_objects=None):
    """ Strip from last layer and rebuild model
    """
    last_layer = model.layers[-1]
    last_layer.activation = keras.activations.get(None)

    # This is required to refresh the model
    model = keras_utils.apply_modifications(model, custom_objects=custom_objects)
    #model.compile(loss=model.loss, optimizer=model.optimizer, metrics=model.metrics)
    return model


def make_train_only_regularizer(regularizer, default=0.):
    #return TrainOnlyRegularizer(regularizer, default=default)
    if regularizer is None:
        return lambda x: default
    return lambda x: K.in_train_phase(regularizer(x), default)


class TemperatureScaling(keras.layers.Layer):
    def __init__(self, init_log_t=0., activation='softmax', **kwargs):
        self.init_log_t = init_log_t
        self.activation = keras.activations.get(activation)
        super().__init__(**kwargs)

    def get_config(self):
        d = dict(super().get_config())
        d.update({
            'init_log_t': self.init_log_t,
            'activation': self.activation,
        })
        return d

    def build(self, input_shape):   # (., n)
        self.log_t = self.add_weight('log_t', shape=(1,), trainable=True, initializer=keras.initializers.Constant(self.init_log_t)) # (anchors, d)
        #self.trainable_weights.append(self.log_t)   # 
        super().build(input_shape)

    def call(self, inputs):  # input as logits
        z = inputs / tf.exp(self.log_t)    # (., n) -> (., n)
        z = self.activation(z)
        return z

    def compute_output_shape(self, input_shape):
        return input_shape


class PlattScaling(keras.layers.Layer):
    def __init__(self, init_a=1., init_b=0., activation='softmax', **kwargs):
        self.init_a = init_a
        self.init_b = init_b
        self.activation = keras.activations.get(activation)
        super().__init__(**kwargs)

    def get_config(self):
        d = dict(super().get_config())
        d.update({
            'init_a': self.init_a,
            'init_b': self.init_b,
            'activation': self.activation,
        })
        return d

    def build(self, input_shape):   # (., n)
        self.a = self.add_weight('a', shape=(1,), trainable=True, initializer=keras.initializers.Constant(self.init_a)) # (., n)
        self.b = self.add_weight('b', shape=(1,), trainable=True, initializer=keras.initializers.Constant(self.init_b)) # (., n)
        super().build(input_shape)

    def call(self, inputs):  # input as logits
        z = self.a * inputs + self.b    # (., n) -> (., n)
        #z = _activations[self.activation](z)
        z = self.activation(z)
        return z

    def compute_output_shape(self, input_shape):
        return input_shape


class VectorScaling(keras.layers.Layer):
    def __init__(self, init_w=1., init_b=0., w_regularizer=None, b_regularizer=None, activation='softmax', **kwargs):
        self.init_w = init_w
        self.init_b = init_b
        self.w_regularizer = keras.regularizers.get(w_regularizer)
        self.b_regularizer = keras.regularizers.get(b_regularizer)
        self.activation = keras.activations.get(activation)
        super().__init__(**kwargs)

    def get_config(self):
        d = dict(super().get_config())
        d.update({
            'init_w': self.init_w,
            'init_b': self.init_b,
            'activation': self.activation,
            'w_regularizer': keras.regularizers.serialize(self.w_regularizer),
            'b_regularizer': keras.regularizers.serialize(self.b_regularizer),
        })
        return d

    def build(self, input_shape):   # (., n)
        self.w = self.add_weight('w', shape=(input_shape[1],), trainable=True, initializer=keras.initializers.Constant(self.init_w),
                regularizer=make_train_only_regularizer(self.w_regularizer)) # (., n)
        self.b = self.add_weight('b', shape=(input_shape[1],), trainable=True, initializer=keras.initializers.Constant(self.init_b),
                regularizer=make_train_only_regularizer(self.b_regularizer)) # (., n)
        super().build(input_shape)

    def call(self, inputs):  # input as logits
        z = self.w * inputs + self.b    # (., n) -> (., n)
        #z = _activations[self.activation](z)
        z = self.activation(z)
        return z

    def compute_output_shape(self, input_shape):
        return input_shape


def get_logit_calibration_builder(model, method, reg_opts):
    if method == 'ts':
        return TemperatureScalingBuilder(model)
    if method == 'ps':
        return PlattScalingBuilder(model)
    if method == 'vs':
        return VectorScalingBuilder(model, b_l2=reg_opts['b_l2'])
    if method == 'ms':
        return MatrixScalingBuilder(model, b_l2=reg_opts['b_l2'], w_off_diag_l2=reg_opts['w_off_diag_l2'])
    raise NotImplementedError


class LogitCalibrationBuilder:
    def _wrap_model(self, model):
        raise NotImplementedError

    def build(self, optimizer='adam', metrics=None, custom_objects=None):
        model = strip_last_activation(self._model)
        # make all layers non-trainable
        for layer in model.layers:
            layer.trainable = False

        calib_model, callbacks = self._wrap_model(model)
        if metrics is None:
            metrics = ['categorical_accuracy', 'mse']
        loss = 'categorical_crossentropy'
        calib_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        return calib_model, callbacks


class TemperatureScalingBuilder(LogitCalibrationBuilder):
    def __init__(self, model):
        self._model = model

    def _wrap_model(self, model):
        layer = TemperatureScaling(activation='softmax')
        z = layer(model.output)
        model = keras.models.Model(model.input, z)
        callbacks = [  # accessing self.model is invalid for multi-gpu case
            AppendMetric('t', lambda self: np.exp(K.eval(model.get_layer(layer.name).log_t)[0]))
        ]
        return model, callbacks


class PlattScalingBuilder(LogitCalibrationBuilder):
    def __init__(self, model):
        self._model = model

    def _wrap_model(self, model):
        layer = PlattScaling(activation='softmax')
        z = layer(model.output)
        model = keras.models.Model(model.input, z)
        callbacks = [
            AppendMetric('a', lambda self: K.eval(model.get_layer(layer.name).a)[0]),
            AppendMetric('b', lambda self: K.eval(model.get_layer(layer.name).b)[0]),
        ]
        return model, callbacks


class VectorScalingBuilder(LogitCalibrationBuilder):
    def __init__(self, model, b_l2=None):
        self._model = model
        self.b_l2 = b_l2

    def _wrap_model(self, model):
        w_reg = b_reg = None
        k = float(model.output[0].get_shape().as_list()[0])
        if self.b_l2:
            b_reg = keras.regularizers.l2(self.b_l2 / k)
        layer = VectorScaling(w_regularizer=w_reg, b_regularizer=b_reg, activation='softmax')
        z = layer(model.output)
        model = keras.models.Model(model.input, z)
        callbacks = [
            AppendMetric('w_mean', lambda self: K.eval(model.get_layer(layer.name).w).mean()),
            AppendMetric('b_mean', lambda self: K.eval(model.get_layer(layer.name).b).mean()),
        ]
        return model, callbacks


class OffDiagonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, l2=0.0):
        self.l2 = K.cast_to_floatx(l2)

    def __call__(self, mat):
        reg = 0.
        assert mat.shape[0] == mat.shape[1]
        if self.l2:
            reg += self.l2 * K.sum(K.square(mat - tf.diag(tf.diag_part(mat))))
        return reg

    def get_config(self):
        return {'l2': float(self.l2)}


def np_mask_diag(mat):
    """
    >>> np_mask_diag(np.ones(shape=(3, 3))).sum()
    6.0
    """
    assert mat.shape[0] == mat.shape[1]
    diag_mask = np.identity(mat.shape[0]).astype('bool')
    return ma.array(mat, mask=diag_mask)


class MatrixScalingBuilder(LogitCalibrationBuilder):
    def __init__(self, model, b_l2=None, w_off_diag_l2=None):
        self._model = model
        self.b_l2 = b_l2
        self.w_off_diag_l2 = w_off_diag_l2

    def _wrap_model(self, model):
        w_reg = b_reg = None
        k = float(model.output[0].get_shape().as_list()[0])
        if self.w_off_diag_l2:
            w_reg = OffDiagonalRegularizer(self.w_off_diag_l2 / (k * (k-1)))
        if self.b_l2:
            b_reg = keras.regularizers.l2(self.b_l2 / k)
        layer = keras.layers.Dense(k, name=get_unique_name('matrix_scaling'),
                kernel_initializer=keras.initializers.Identity(),
                bias_initializer=keras.initializers.Zeros(),
                kernel_regularizer=make_train_only_regularizer(w_reg),
                bias_regularizer=make_train_only_regularizer(b_reg),
                activation='softmax')
        z = layer(model.output)
        model = keras.models.Model(model.input, z)
        callbacks = [
            AppendMetric('w_diag_mean', lambda self: K.eval(tf.diag_part(model.get_layer(layer.name).kernel)).mean()),
            AppendMetric('w_off_diag_mean', lambda self: np_mask_diag(K.eval(model.get_layer(layer.name).kernel)).mean()),
            AppendMetric('b_mean', lambda self: K.eval(model.get_layer(layer.name).bias).mean()),
        ]
        return model, callbacks


def dirichlet_multinomial_nll(y_true, y_pred, from_logits=False, clip_min=1e-12, clip_max=1e+12):
    """
    y_true: count data with shape (N, K)
    y_pred: parameters alpha of DirMult or their logits with shape (N, K)
    """
    y_true = K.cast(y_true, y_pred.dtype)
    if from_logits:
        alpha = tf.exp(y_pred)   # (N, K)
    else:
        alpha = y_pred
    #alpha = tf.clip_by_value(alpha, epsilon, alpha)
    alpha = tf.clip_by_value(alpha, clip_min, clip_max)

    nll = 0.  # (N,)
    nll += tf.lgamma(tf.reduce_sum(alpha + y_true, axis=1))
    nll -= tf.reduce_sum(tf.lgamma(alpha + y_true), axis=1)
    nll -= tf.lgamma(tf.reduce_sum(alpha, axis=1))
    nll += tf.reduce_sum(tf.lgamma(alpha), axis=1)
    return nll


class AlphaCalibrationBuilder:
    clip_min = 1e-12
    clip_max = 1e+12

    def __init__(self, model, feature_layer=-2, activation='exponential', alpha0=1., alpha0_l1=0., log_alpha0_l2=0., always_regularize=False):
        self._model = model
        self._feature_layer = feature_layer   # default is penultimate layer
        self.activation = activation
        self.alpha0_l1 = alpha0_l1
        self.log_alpha0_l2 = log_alpha0_l2
        self.always_regularize = always_regularize
        assert self.activation in ('exponential', 'softplus')
        if self.activation == 'exponential':
            self.bias = np.log(alpha0)
        elif self.activation == 'softplus':
            self.bias = np.log(np.exp(alpha0) - 1)

    def _wrap_model(self, model):
        try:
            layer_num = int(self._feature_layer)
            layer_name = model.layers[layer_num].name
        except ValueError:
            layer_name = self._feature_layer
        logging.info('Feature layer: %s', layer_name)
        feature_layer = model.get_layer(layer_name)   # get feature layer
        alpha0_layer = keras.layers.Dense(1, name=get_unique_name('alpha0'),
                kernel_initializer=keras.initializers.Zeros(),
                bias_initializer=keras.initializers.Constant(self.bias),
                activation = self.activation)
        if len(feature_layer.output.shape) > 2:
            feature_output = keras.layers.Flatten()(feature_layer.output)
        else:
            feature_output = feature_layer.output
        alpha0 = alpha0_layer(feature_output) # (., 1)
        #alpha0 = keras.layers.Lambda(lambda x: tf.clip_by_value(x, self.clip_min, self.clip_max))(alpha0)
        alpha = keras.layers.multiply([model.output, alpha0])   # (., 1), (., K) -> (., K)
        calib_model = keras.models.Model(model.input, alpha)
        if self.always_regularize:
            f = lambda x: x
        else:
            f = lambda x: K.in_train_phase(x, 0.)
        calib_model.add_loss(f(K.mean(alpha0) * self.alpha0_l1))
        calib_model.add_loss(f(K.mean(K.square(K.log(alpha0))) * self.log_alpha0_l2))
        #calib_model.add_loss(K.mean(alpha0) * self.alpha0_l1 + 100)
        callbacks = [
            AppendMetric('alpha0_w_mean', lambda self: K.eval(calib_model.get_layer(alpha0_layer.name).kernel).mean()),
            AppendMetric('alpha0_b_mean', lambda self: K.eval(calib_model.get_layer(alpha0_layer.name).bias).mean()),
        ]
        return calib_model, callbacks

    def build(self, optimizer='adam', metrics=None, custom_objects=None):
        calib_model, callbacks = self._wrap_model(self._model)
        if metrics is None:
            metrics = ['categorical_accuracy']
        loss = dirichlet_multinomial_nll
        calib_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        return calib_model, callbacks
