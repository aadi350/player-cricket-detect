"""
    Adapted from Google/TF Keras for hyper-parameter tuning
"""
from datetime import datetime

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

import datagen.dataset_generator as dataset_generator
import models.mobilenet.mobilenet as mobilenet

EPOCHS = 1
MODEL = 'mobilenet'
LOGDIR = "logs/hparam_tuning/mobilenet"

(train_gen, val_gen), class_dict = dataset_generator.get_datagen()
# Specify discrete OR ranges of hyper-parameter intervals
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.3))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

METRIC_ACCURACY = 'accuracy'


# Harness for setting up model for each configuration
def train_test_model(hparams, train_data, val_data):
    model, _ = mobilenet.build_model(hparams)
    model.compile(
        optimizer=hparams['HP_OPTIMIZER'],
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    model.fit(
        train_data,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.TensorBoard(LOGDIR),  # log metrics
            hp.KerasCallback(LOGDIR, hparams),
        ]
    )
    _, accuracy = model.evaluate(val_data)
    return accuracy


# Harness for running model with logs
def run(hparams, train_data, val_data):
    hp.hparams(hparams)  # record the values used in this trial
    accuracy = train_test_model(hparams, train_data, val_data)
    tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)


# Grid search using every combination of discrete values
# TODO find a more efficient way of doing this
session_num = 0
for num_units in HP_NUM_UNITS.domain.values:
    for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
        for optimizer in HP_OPTIMIZER.domain.values:
            hparams = {
                'HP_NUM_UNITS': num_units,
                'HP_DROPOUT': dropout_rate,
                'HP_OPTIMIZER': optimizer,
            }
            run_name = "run-%d" % session_num
            print('--- Starting trial: %s' % run_name)
            run(hparams, train_gen, val_gen)
            print({h: hparams[h] for h in hparams})
            session_num += 1
