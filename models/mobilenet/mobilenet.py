import tensorflow as tf
import tensorflow_addons as tfa
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation, Resizing
from tensorflow.python.keras.layers import Dense

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# Parameter Inputs
DATA_DIR = "../../data/img/sahil_categories"
AUTOTUNE = tf.data.AUTOTUNE
NUM_CLASSES = 2
BATCH_SIZE = 32
image_height = 224
image_width = 224
IMAGE_SIZE = (image_width, image_height)
IMAGE_SIZE_DEPTH = (image_width, image_height, 3)
VALIDATION_SPLIT = 0.4
LABEL_MODE = 'categorical'
COLOR_MODE = 'rgb'
SEED = 42

# data augmentation layers
data_augmentation = tf.keras.Sequential([
    RandomFlip('horizontal'),
    RandomRotation(0.2),
])


def build_model(hparams=None):
    DROPOUT = 0.3
    NUM_UNITS = 32
    if hparams is not None:
        DROPOUT = hparams['HP_DROPOUT']
        NUM_UNITS = hparams['HP_NUM_UNITS']

    # Model configuration
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    # feature extraction into
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMAGE_SIZE_DEPTH,
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False
    # batch data
    resize_layer = Resizing(224, 224, name='resize')

    # convert to single 1280 element
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

    prediction_layer = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')

    inputs = tf.keras.Input(shape=(None, None, 3))
    x = resize_layer(inputs)
    x = data_augmentation(x)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = Dense(NUM_UNITS, activation='relu')(x)
    x = tf.keras.layers.Dropout(DROPOUT)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    print(model.summary())
    return model, base_model


def compile_model(model, optimizer=None, loss=None, metrics=None):
    base_learning_rate = 0.0001
    optimizer = optimizer if optimizer is not None else tf.keras.optimizers.Adam(lr=base_learning_rate)
    loss = loss if loss is not None else 'categorical_crossentropy'
    metrics = metrics if metrics is not None else ['categorical_accuracy']
    model.compile(optimizer='sgd',
                  loss=tfa.losses.SigmoidFocalCrossEntropy(),
                  metrics=metrics)
    print(model.summary())


def return_model():
    model, _ = build_model()
    compile_model(model)
    return model


if __name__ == '__main__':
    return_model()
