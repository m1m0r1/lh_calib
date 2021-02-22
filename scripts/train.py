import sys
import os
sys.path.insert(0, os.path.abspath(__file__).rsplit('/', 2)[0])
from argtools import command, argument
import logging
import subprocess
from lh_calib import keras_utils
from lh_calib import data_gen
from lh_calib import data_models
from lh_calib import networks
import settings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np
import pandas as pd
import keras
import tensorflow as tf


def build_model(args, input_shape, nclass, last_activation='softmax'):
    logging.info('Building model: %s', args.model)
    logging.info('input shape: %s', input_shape)
    fc_dropout_rate = args.fc_dropout_rate
    logging.info('Using fc dropout rate: %s', fc_dropout_rate)
    fc_depth = 1 if args.fc_depth is None else args.fc_depth
    logging.info('Using fc depth: %s', fc_depth)

    if args.model == 'simple_cnn':
        dropout_rate = 0.05 if args.dropout_rate is None else args.dropout_rate
        logging.info('Using dropout rate: %s', dropout_rate)
        model = networks.build_simple_cnn(input_shape, output_dim=nclass, batch_norm=False, dropout_rate=dropout_rate, fc_dropout_rate=fc_dropout_rate, fc_depth=fc_depth, last_activation=last_activation)
        return model

    if args.model == 'vgg16':
        model = networks.build_vgg16(input_shape, output_dim=nclass, fc_dropout_rate=fc_dropout_rate, fc_depth=fc_depth, last_activation=last_activation)
        return model

    raise NotImplementedError


@command.add_sub
@argument('-i', '--input', required=True)
@argument('-m', '--model', required=True)
@argument('-s', '--setting', required=True)   # load default settings
@argument('--run-name', default='run')
@argument('-o', '--outdir', default='lh-train_out')
@argument('--resume', action='store_true')
@argument('--testing', action='store_true')
@argument('-e', '--max-epochs', type=int, default=16)
@argument('-es', '--early-stopping', type=int)
@argument('-b', '--batch-size', type=int, default=128)
#@argument('-k', '--nclass', type=int)
@argument('-ml', '--model-loss', choices=['multinom_nll'], default='multinom_nll')
@argument('-lr', '--learning-rate', type=float, default=1e-3)
@argument('--use-warmup', action='store_true')
@argument('--ms-decay-epochs', nargs='+', type=int)
@argument('-do', '--dropout-rate', type=float, default=None)
@argument('-fcdo', '--fc-dropout-rate', type=float, default=0.5)
@argument('-fcdp', '--fc-depth', type=int, default=None)
@argument('--seed', type=int)
@argument('-vs', '--valid-subsample', type=int)
@argument('-sw', '--save-weight', choices=['epochs', 'last'], default='epochs')
@argument('-sp', '--save-period', type=int, default=5)
@argument('--save-best', action='store_true')
def train_lh(args):
    if args.seed is not None:
        np.random.seed(args.seed)
        tf.set_random_seed(args.seed)

    batch_size = args.batch_size

    train_max_entry = None
    valid_max_entry = args.valid_subsample
    if args.testing:
        logging.info('* test run *')
        train_max_entry = max_entry = int(batch_size * 4)
        valid_max_entry = max_entry or valid_max_entry

    logging.info('train_max_entry: %s', train_max_entry)
    logging.info('valid_max_entry: %s', valid_max_entry)
    max_epochs = args.max_epochs
    logging.info('Batch size: %s', batch_size)
    logging.info('Max epochs: %s', max_epochs)

    if args.model_loss == 'multinom_nll':
        loss = 'categorical_crossentropy'
        last_activation = 'softmax'
        use_prob = True
        use_weight = True
        weight_norm_mode = 'mean'
        metrics = ['mse', 'categorical_accuracy']   # these are calculated without weights
    else:
        raise NotImplementedError

    logging.info('Model loss: %s', args.model_loss)
    logging.info('Loss: %s', loss)
    logging.info('use_prob: %s', use_prob)

    # setup optimizers
    optimizer = keras.optimizers.Adam(lr=args.learning_rate)
    lr_scheduler = None
    if args.use_warmup or args.ms_decay_epochs:
        if args.use_warmup:
            warmup_epochs = 5
        else:
            warmup_epochs = 0
        init_lr = optimizer.get_config()['lr']
        logging.info('Using lr scheduler with wrmup_epochs: %s, decay_epochs: %s (init_lr: %s)',
                warmup_epochs, args.ms_decay_epochs, init_lr)
        lr_scheduler = keras_utils.get_multi_step_lr_scheduler(init_lr=init_lr, warmup_epochs=warmup_epochs, decay_epochs=args.ms_decay_epochs)

    # setup image processor
    setting = settings.get_data_setting(args.setting)
    logging.info('setting for %s: %s', args.setting, setting)
    input_shape = setting['input_shape']

    preprocess_input = networks.get_preprocess_input(args.model)

    input_tab = pd.read_csv(args.input, sep='\t')
    nclass = len(input_tab['hist'].iloc[0].split(','))
    logging.info('input_shape: %s', input_shape)
    logging.info('nclass: %s', nclass)

    subprocess.Popen('mkdir -p {}'.format(args.outdir), shell=True)

    logging.info('Results will be found in %s with run name: %s', args.outdir, args.run_name)
    custom_objects = {}
    custom_objects.update(globals())

    prefix = '{}/{}'.format(args.outdir, args.run_name)   # run name
    weight_prefix = '{}.weights'.format(prefix)
    model_summary_path = '{}.model.summary.txt'.format(prefix)
    record_prefix = '{}.record'.format(prefix)
    last_weight_info = None
    if args.save_weight == 'epochs':
        last_weight_info = keras_utils.WeightInfoEpochs(weight_prefix)
    elif args.save_weight== 'last':
        last_weight_info = keras_utils.WeightInfo(prefix + '.last.weights')
    else:
        raise NotImplementedError

    def build_fn():
        model = build_model(args, input_shape, nclass, last_activation=last_activation)
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        return model

    builder = keras_utils.ResumableModelBuilder(prefix=prefix, weight_info=last_weight_info, build_fn=build_fn, custom_objects=custom_objects)
    if args.resume:
        builder.build_or_resume()
    else:
        builder.build()
    run_model = model = builder.model

    assert run_model._is_compiled
    model.summary(print_fn=lambda x: logging.info(x))
    with open(model_summary_path, 'w+') as fp:
        model.summary(print_fn=lambda x: fp.write(x + '\n'))

    callbacks = []

    input_tab = data_models.LHTable(input_tab)
    train_tab = input_tab.get_partition('train', max_entry=train_max_entry)
    valid_tab = input_tab.get_partition('valid', max_entry=train_max_entry)
    with ThreadPoolExecutor() as pool:
        train_x = train_tab.get_images(data_source=setting.get('data_source'), pool=pool)
        train_hists = train_tab.get_hists()
        train_processor = data_gen.ListProcessor()
        from skimage.color import gray2rgb
        if setting.get('gray2rgb'):
            train_processor.append(data_gen.LambdaProcessor(gray2rgb))
        if setting.get('augment_opts'):
            train_processor.append(data_gen.KerasImageAugmentor(setting['augment_opts']))

        train_processor.append(data_gen.get_image_cropper(input_shape[:2]))
        train_processor.append(data_gen.LambdaProcessor(preprocess_input))
        valid_x = valid_tab.get_images(data_source=setting.get('data_source'), pool=pool)
        valid_hists = valid_tab.get_hists()
        valid_processor = data_gen.ListProcessor()
        if setting.get('gray2rgb'):
            valid_processor.append(data_gen.LambdaProcessor(gray2rgb))
        valid_processor.append(data_gen.get_image_cropper(input_shape[:2]))
        valid_processor.append(data_gen.LambdaProcessor(preprocess_input))
        logging.info('Training length: %s', len(train_x))
        logging.info('Validation length: %s', len(valid_hists))

        train_gen = data_gen.LHGenerator(train_x, train_hists, processor=train_processor, batch_size=batch_size, use_prob=use_prob, use_weight=use_weight, weight_norm_mode=weight_norm_mode, shuffle=True)
        valid_gen = data_gen.LHGenerator(valid_x, valid_hists, processor=valid_processor, batch_size=batch_size, use_prob=use_prob, use_weight=use_weight, weight_norm_mode=weight_norm_mode)

        if lr_scheduler:
            lr_scheduler_cb = keras.callbacks.LearningRateScheduler(lr_scheduler)
            callbacks.append(lr_scheduler_cb)
        recorder = keras_utils.HistoryRecorder(out_prefix=record_prefix, initial_epoch=builder.initial_epoch)

        callbacks.append(recorder)
        last_weights_cb = last_weight_info.get_callback(model, period=args.save_period)
        callbacks.append(last_weights_cb)
        if args.save_best:
            tester = keras_utils.MetricTester(recorder, metric='val_loss', cond='min')
            best_weight_info = keras_utils.WeightInfo(prefix + '.best.weights')
            best_weight_cb = best_weight_info.get_callback(model, period=0, test_fn=tester.is_best)
            callbacks.append(best_weight_cb)

        if args.early_stopping:
            callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=args.early_stopping, verbose=1, mode='min'))

        run_model.fit_generator(generator=train_gen,
                            validation_data=valid_gen,
                            initial_epoch=builder.initial_epoch,
                            epochs=max_epochs,
                            shuffle=False, # shuffle is applied inside of data_gen
                            callbacks=callbacks,
                            workers=4,
                            max_queue_size=2)

if __name__ == '__main__':
    command.run()
