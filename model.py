import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation, Resizing
from tensorflow.keras.layers import Flatten
import numpy as np

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
DATA_DIR = "/media/aadi/Library1/_assets/video/sahil_categories"
AUTOTUNE = tf.data.AUTOTUNE
NUM_CLASSES = 2
BATCH_SIZE = 1
image_height = 224
image_width = 224
IMAGE_SIZE = (image_width, image_height)
IMAGE_SIZE_DEPTH = (image_width, image_height, 3)
VALIDATION_SPLIT = 0.4
LABEL_MODE = 'categorical'
COLOR_MODE = 'rgb'
SEED = 42
EPOCHS = 15

# Image Dataset Generation
train_ds = image_dataset_from_directory(
    DATA_DIR,
    validation_split=VALIDATION_SPLIT,
    subset='training',
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode=LABEL_MODE,
    color_mode=COLOR_MODE,
    seed=SEED
)

val_ds = image_dataset_from_directory(
    DATA_DIR,
    validation_split=VALIDATION_SPLIT,
    subset='validation',
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode=LABEL_MODE,
    color_mode=COLOR_MODE,
    seed=SEED
)

class_names = train_ds.class_names
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    RandomFlip('horizontal'),
    RandomRotation(0.2),
])

# Model configuration
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
base_model = tf.keras.applications.MobileNetV2(input_shape=IMAGE_SIZE_DEPTH,
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

image_batch, label_batch = next(iter(train_ds))
feature_batch = base_model(image_batch)
resize_layer = Resizing(224, 224, name='resize')

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
flatten_layer = Flatten()
prediction_layer = tf.keras.layers.Dense(NUM_CLASSES, activation='sigmoid')
prediction_batch = prediction_layer(feature_batch_average)

# Model definition
inputs = tf.keras.Input(shape=(None, None, 3))
x = resize_layer(inputs)
x = data_augmentation(x)

x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = prediction_layer(x)
outputs = tf.keras.layers.Dropout(0.3)(x)
model = tf.keras.Model(inputs, x)

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

print(model.summary())

hist = model.fit(
    train_ds,
    validation_data=val_ds,
    steps_per_epoch=2,
    epochs=EPOCHS
)
