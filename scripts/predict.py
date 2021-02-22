import sys
import os
sys.path.insert(0, os.path.abspath(__file__).rsplit('/', 2)[0])
from argtools import command, argument
import logging
import numpy as np
import pandas as pd
import keras
import gc
import lh_calib
from lh_calib import data_gen
from lh_calib import keras_utils
from lh_calib import data_models
from lh_calib import predictions
from lh_calib import networks
import settings


@command
@argument('-i', '--input', required=True)
@argument('-s', '--setting', required=True)
@argument('-m', '--model', required=True)
@argument('-mp', '--model-path', required=True)
@argument('-mw', '--model-weights')
@argument('-o', '--output', default='/dev/stdout')
@argument('-st', '--score-type', choices=['prob', 'alpha'], default='prob')
@argument('-b', '--batch-size', type=int, default=128)
@argument('-p', '--partition')
@argument('-ud', '--use-dropout', action='store_true')
@argument('-ua', '--use-augment', action='store_true')
@argument('-mcs', '--mc-samples', type=int, default=None)
@argument('-me', '--max-entry', type=int)
def predict(args):
    """
    """
    batch_size = args.batch_size
    logging.info('Batch size: %s', batch_size)
    logging.info('score_type: %s', args.score_type)
    if args.score_type == 'prob':
        use_prob = True
    elif args.score_type == 'alpha':
        use_prob = False
    else:
        raise NotImplementedError
    logging.info('use_dropout: %s', args.use_dropout)
    logging.info('use_augment: %s', args.use_augment)
    logging.info('mc_samples: %s', args.mc_samples)

    setting = settings.get_data_setting(args.setting)
    logging.info('setting for %s: %s', args.setting, setting)
    input_shape = setting['input_shape']
    preprocess_input = networks.get_preprocess_input(args.model)
    logging.info('input_shape: %s', input_shape)

    table = data_models.LHTable.load(args.input)
    logging.info('partition: %s', args.partition)
    if args.partition:
        max_entry = args.max_entry
        logging.info('Max entry: %s', max_entry)
        table = table.get_partition(args.partition, max_entry=max_entry)
    ids = table.id

    custom_objects = {}
    custom_objects.update(lh_calib.get_custom_objects())
    #custom_objects.update(globals())
    model = keras.models.load_model(args.model_path, custom_objects=custom_objects)
    if args.model_weights:
        model.load_weights(args.model_weights)

    if args.use_dropout:
        model = keras_utils.apply_mc_dropout(model)

    processor = data_gen.ListProcessor()
    from skimage.color import gray2rgb
    if setting.get('gray2rgb'):
        processor.append(data_gen.LambdaProcessor(gray2rgb))
    if args.use_augment:
        if setting.get('augment_opts'):
            processor.append(data_gen.KerasImageAugmentor(setting['augment_opts']))
    processor.append(data_gen.get_image_cropper(input_shape[:2]))
    processor.append(data_gen.LambdaProcessor(preprocess_input))

    pool = None
    x = table.get_images(data_source=setting.get('data_source'), pool=pool)
    lh_gen = data_gen.LHGenerator(x, processor=processor, batch_size=batch_size, use_prob=use_prob, shuffle=False)

    if args.mc_samples:
        score_list = []  # (nsample, N, nclass)
        for i in range(1, args.mc_samples + 1):
            logging.info('Iteration %s', i)
            scores = model.predict_generator(lh_gen, verbose=1, workers=4, max_queue_size=2)   # [(nclass,)]
            score_list.append(scores)
            gc.collect()

        score_samples = np.asarray(score_list).transpose((1, 0, 2))   # (N, nsample, nclass)
        if args.score_type == 'prob':
            pred = predictions.MCProbPrediction(score_samples, ids=ids)
        elif args.score_type == 'alpha':
            pred = predictions.MCAlphaPrediction(score_samples, ids=ids)
        pred.save(args.output)
    else:
        scores = model.predict_generator(lh_gen, verbose=1, workers=2, max_queue_size=2)   # [(nclass,)]
        if args.score_type == 'prob':
            pred = predictions.ProbPrediction(scores, ids=ids)
        elif args.score_type == 'alpha':
            pred = predictions.AlphaPrediction(scores, ids=ids)
        pred.save(args.output)


if __name__ == '__main__':
    command.run()
