def get_data_setting_mds1():
    return {
        'input_shape': (224, 224, 3),
        'augment_opts': {
            'horizontal_flip': True,
            'width_shift_range': 10,
            'height_shift_range': 10,
            'rotation_range': 180,
        },
    }

def get_data_setting_mixed_mnist():
    return {
        'input_shape': (28, 28, 1),
        'data_source': 'mixed_mnist',
    }

def get_data_setting_mixed_cifar10():
    return {
        'input_shape': (32, 32, 3),
        'data_source': 'mixed_cifar10',
        'augment_opts': {
            'horizontal_flip': True,
            'height_shift_range': 5,
            'width_shift_range': 5,
            'zoom_range': 0.2,
            'rotation_range': 20,
        },
    }

def get_data_setting(name):
    return globals()['get_data_setting_' + name]()
