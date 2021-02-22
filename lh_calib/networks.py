import keras
import keras.layers
import keras.models


def add_cnn_layer(model, filters, kernel_size, pooling_size=0, padding='same', dropout_rate=0, dilation_rate=1, batch_norm=True, **kwargs):
    model.add(keras.layers.Conv2D(filters, kernel_size, dilation_rate=dilation_rate, padding=padding, **kwargs))
    if batch_norm:
        model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    if pooling_size > 0:
        model.add(keras.layers.MaxPooling2D(pooling_size))
    if dropout_rate > 0:
        model.add(keras.layers.Dropout(rate=dropout_rate))
    return model


def build_simple_cnn(input_shape, output_dim, dropout_rate=0.05, fc_dropout_rate=0.5, fc_width=128, fc_depth=1, batch_norm=False, last_activation='softmax'):
    """
    """
    model = keras.models.Sequential()
    padding = 'same'

    model = add_cnn_layer(model, 32, 5, dropout_rate=dropout_rate, pooling_size=2, input_shape=input_shape, batch_norm=batch_norm)   # (., X, Y, C) -> (., X, Y, C)
    model = add_cnn_layer(model, 64, 7, dropout_rate=dropout_rate, pooling_size=4, batch_norm=batch_norm)
    model = add_cnn_layer(model, 128, 11, dropout_rate=dropout_rate, batch_norm=batch_norm)
    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.Dropout(rate=fc_dropout_rate))

    fc_dims = [fc_width] * fc_depth
    for dim in fc_dims:
        model.add(keras.layers.Dense(dim, activation='relu'))
        model.add(keras.layers.Dropout(rate=fc_dropout_rate))

    # last fc
    model.add(keras.layers.Dense(output_dim, activation=last_activation))
    return model


def build_vgg16(input_shape, output_dim, fc_dropout_rate=0.5, fc_width=128, fc_depth=1, init_weights=None, last_activation='softmax'):
    """
    """
    from keras.applications.vgg16 import VGG16
    base_model = VGG16(include_top=False, weights=init_weights, pooling='max', input_shape=input_shape)    # min input is 48
    model = keras.models.Sequential()
    model.add(base_model)
    model.add(keras.layers.Dropout(rate=fc_dropout_rate))

    fc_dims = [fc_width] * fc_depth
    for dim in fc_dims:
        model.add(keras.layers.Dense(dim, activation='relu'))
        model.add(keras.layers.Dropout(rate=fc_dropout_rate))

    # last fc
    model.add(keras.layers.Dense(output_dim, activation=last_activation))
    return model


def get_preprocess_input(model):
    if model == 'vgg16':
        from keras.applications.vgg16 import preprocess_input
        return preprocess_input   # [0-255] -> (1, -1)
    if model in ('simple_cnn',):
        from keras.applications.inception_resnet_v2 import preprocess_input   # [0-255] -> (1, -1)
        return preprocess_input
    raise NotImplementedError
