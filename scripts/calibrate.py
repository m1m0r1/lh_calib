import sys
import os
sys.path.insert(0, os.path.abspath(__file__).rsplit('/', 2)[0])
from argtools import command, argument
import numpy as np
import keras
import tensorflow as tf
import logging
import lh_calib
from lh_calib import data_gen
from lh_calib import data_models
from lh_calib import calibration
from lh_calib import networks
from lh_calib import keras_utils
import settings
from concurrent.futures import ThreadPoolExecutor


@command
@argument('-i', '--input', required=True)
@argument('-s', '--setting', required=True)
@argument('-m', '--model', required=True)
@argument('-mp', '--model-path', required=True)
@argument('-mw', '--model-weights')
@argument('-o', '--out-prefix', default='lh_calib_out', help='Output {out_prefix}.{model.h5,record.txt,best.weights.h5}')
@argument('-lc', '--logit-calib', choices=['ts', 'ps', 'vs', 'ms'])
@argument('--w-off-diag-l2', type=float, default=0.)
@argument('--b-l2', type=float, default=0.)
@argument('-la', '--learn-alpha', action='store_true')
@argument('-aa', '--alpha-activation', default='exponential', choices=['exponential', 'softplus'])
@argument('-a0', '--alpha0', default=1., type=float)
@argument('--alpha0-l1', default=0., type=float)
@argument('--log-alpha0-l2', default=0., type=float)
@argument('-a0ar', '--alpha0-always-regularize', action='store_true')
@argument('-fl', '--feature-layer', default='-2')  # used for alpha
@argument('-ut', '--use-train', action='store_true', help='reuse training data (mainly for alpha calib)')
@argument('-vs', '--valid-subsample', type=int)   # subsample (take first n lines) from original validation partition
@argument('-ua', '--use-augment', action='store_true')
@argument('-b', '--batch-size', type=int, default=128)
@argument('-cvr', '--calib-valid-ratio', type=float, default=0.2)  # data used for validation of calibration
@argument('-e', '--max-epochs', type=int, default=30)
@argument('-es', '--early-stopping', type=int)
@argument('--testing', action='store_true')
@argument('--seed', type=int)
def calibrate(args):
    if args.seed is not None:
        np.random.seed(args.seed)
        tf.set_random_seed(args.seed)

    batch_size = args.batch_size
    max_entry = None
    valid_max_entry = args.valid_subsample
    if args.testing:
        logging.info('* test run *')
        max_entry = int(batch_size * 4)
        valid_max_entry = max_entry or valid_max_entry

    logging.info('max_entry: %s', max_entry)
    max_epochs = args.max_epochs
    logging.info('Batch size: %s', batch_size)
    logging.info('Max epochs: %s', max_epochs)

    table = data_models.LHTable.load(args.input)
    valid_table = table.get_partition('valid', max_entry=valid_max_entry)
    calib_table, calib_valid_table = valid_table.split(ratio=1 - args.calib_valid_ratio)

    if args.use_train:
        logging.info('Adding training data')
        train_table = table.get_partition('train')
        calib_table = train_table + calib_table

    logging.info('#record for calibration: %s', len(calib_table))
    logging.info('#record for calib-validation: %s', len(calib_valid_table))

    # Building model
    custom_objects = lh_calib.get_custom_objects()
    model = keras.models.load_model(args.model_path, custom_objects=custom_objects)
    if args.model_weights:
        model.load_weights(args.model_weights)

    callbacks = []
    if args.logit_calib:
        logging.info('Building logit calibration model: %s', args.logit_calib)
        builder = calibration.get_logit_calibration_builder(model,
                method=args.logit_calib,
                reg_opts={'b_l2': args.b_l2, 'w_off_diag_l2': args.w_off_diag_l2})
        model, cbs = builder.build(custom_objects=custom_objects)
        callbacks.extend(cbs)
    else:
        # make all layers non-trainable if no logit calibration is applied
        for layer in model.layers:
            layer.trainable = False
    if args.learn_alpha:
        logging.info('Building alpha calibration model')
        builder = calibration.AlphaCalibrationBuilder(model, feature_layer=args.feature_layer, activation=args.alpha_activation,
                alpha0=args.alpha0, alpha0_l1=args.alpha0_l1, log_alpha0_l2=args.log_alpha0_l2,
                always_regularize=args.alpha0_always_regularize)
        model, cbs = builder.build(custom_objects=custom_objects)
        callbacks.extend(cbs)

    model.summary(print_fn=lambda x: logging.info(x))   # Show model summary info
    with open(args.out_prefix + '.summary.txt', 'w+') as fp:
        model.summary(print_fn=lambda x: fp.write(x + '\n'))   # Show model summary info
    model.save(args.out_prefix + '.model.h5')

    recorder = keras_utils.HistoryRecorder(out_prefix=args.out_prefix + '.record')
    callbacks.append(recorder)

    tester = keras_utils.MetricTester(recorder, metric='val_loss', cond='min')
    weight_info = keras_utils.WeightInfo(args.out_prefix + '.best.weights')
    weight_cb = weight_info.get_callback(model, period=0, test_fn=tester.is_best)
    callbacks.append(weight_cb)

    if args.early_stopping:
        callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=args.early_stopping, verbose=1, mode='min'))

    # setup image processor
    setting = settings.get_data_setting(args.setting)
    logging.info('setting for %s: %s', args.setting, setting)
    input_shape = setting['input_shape']

    calib_processor = data_gen.ListProcessor()
    from skimage.color import gray2rgb
    if setting.get('gray2rgb'):
        calib_processor.append(data_gen.LambdaProcessor(gray2rgb))
    if args.use_augment:
        logging.info('Using data augmentation: %s', setting['augment_opts'])
        calib_processor.append(data_gen.KerasImageAugmentor(setting['augment_opts']))
    calib_processor.append(data_gen.get_image_cropper(input_shape[:2]))
    preprocess_input = networks.get_preprocess_input(args.model)
    calib_processor.append(data_gen.LambdaProcessor(preprocess_input))

    calib_valid_processor = data_gen.ListProcessor()
    if setting.get('gray2rgb'):
        calib_valid_processor.append(data_gen.LambdaProcessor(gray2rgb))
    calib_valid_processor.append(data_gen.get_image_cropper(input_shape[:2]))
    calib_valid_processor.append(data_gen.LambdaProcessor(preprocess_input))

    with ThreadPoolExecutor() as pool:
        calib_x = calib_table.get_images(data_source=setting.get('data_source'), pool=pool)
        calib_hists = calib_table.get_hists()
        calib_valid_x = calib_valid_table.get_images(data_source=setting.get('data_source'), pool=pool)
        calib_valid_hists = calib_valid_table.get_hists()

        use_prob = not bool(args.learn_alpha)
        calib_gen = data_gen.LHGenerator(calib_x, calib_hists, processor=calib_processor, batch_size=batch_size, use_prob=use_prob, use_weight=True, weight_norm_mode='mean', shuffle=True)
        calib_valid_gen = data_gen.LHGenerator(calib_valid_x, calib_valid_hists, processor=calib_valid_processor, batch_size=batch_size, use_prob=use_prob, use_weight=True, weight_norm_mode='mean')

        model.fit_generator(generator=calib_gen,
                            validation_data=calib_valid_gen,
                            epochs=max_epochs,
                            shuffle=False, # shuffle is applied inside of LHGenerator
                            callbacks=callbacks,
                            workers=4,
                            max_queue_size=2)

if __name__ == '__main__':
    command.run()
