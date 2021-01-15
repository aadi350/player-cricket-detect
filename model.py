from datetime import datetime
import matplotlib.pyplot as plt
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

# TODO: Refactor to use dataset_generator
# Parameter Inputs
DATA_DIR = "/media/aadi/Library1/_assets/video/sahil_categories"
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

# TODO: reset epochs
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 10

# # Image Dataset Generation
# train_ds = image_dataset_from_directory(
#     DATA_DIR,
#     validation_split=VALIDATION_SPLIT,
#     subset='training',
#     image_size=IMAGE_SIZE,
#     batch_size=BATCH_SIZE,
#     label_mode=LABEL_MODE,
#     color_mode=COLOR_MODE,
#     seed=SEED
# )
#
# val_ds = image_dataset_from_directory(
#     DATA_DIR,
#     validation_split=VALIDATION_SPLIT,
#     subset='validation',
#     image_size=IMAGE_SIZE,
#     batch_size=BATCH_SIZE,
#     label_mode=LABEL_MODE,
#     color_mode=COLOR_MODE,
#     seed=SEED
# )

# class_names = train_ds.class_names
# train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
# val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    RandomFlip('horizontal'),
    RandomRotation(0.2),
])


def get_callbacks():
    # Callback Definitions
    def lr_schedule(epoch):
        """
      Returns a custom learning rate that decreases as epochs progress.
      """
        learning_rate = 0.2
        if epoch > 10:
            learning_rate = 0.02
        if epoch > 20:
            learning_rate = 0.01
        if epoch > 50:
            learning_rate = 0.005

        tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
        return learning_rate

    checkpoint_filepath = '/tmp/checkpoint'
    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_categorical_accuracy',
        mode='max',
        save_best_only=True)

    return [lr_callback, tensorboard_callback, model_checkpoint_callback]


def config_model():
    # Model configuration
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    # feature extraction into
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMAGE_SIZE_DEPTH,
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False
    # batch data
    image_batch, label_batch = next(iter(train_ds))
    feature_batch = base_model(image_batch)
    resize_layer = Resizing(224, 224, name='resize')

    # convert to single 1280 element
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)
    flatten_layer = Flatten()
    prediction_layer = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    prediction_batch = prediction_layer(feature_batch_average)

    inputs = tf.keras.Input(shape=(None, None, 3))
    x = resize_layer(inputs)
    x = data_augmentation(x)

    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    return model, base_model


def compile_model(model):
    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    print(model.summary())


def lr_schedule(epoch):
    """
  Returns a custom learning rate that decreases as epochs progress.
  """
    learning_rate = 0.2
    if epoch > 10:
        learning_rate = 0.02
    if epoch > 20:
        learning_rate = 0.01
    if epoch > 50:
        learning_rate = 0.005

    tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
    return learning_rate


def train_model(model, train_ds, val_ds):
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=2,
        callbacks=get_callbacks(),
        epochs=INITIAL_EPOCHS
    )


def fine_tune(base_model, model):
    base_learning_rate = 0.0001
    """ For fine-tuning """
    base_model.trainable = True
    # Fine-tune from this layer onwards
    fine_tune_at = 100

    # freeze all BEFORE fine_tune_at
    # lower layers need not be fine-tuned
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate / 10),
                  metrics=['categorical_accuracy'])

    total_epochs = INITIAL_EPOCHS + FINE_TUNE_EPOCHS
    history_fine = model.fit(train_ds,
                             epochs=total_epochs,
                             initial_epoch=10,
                             validation_data=val_ds)

    acc = history_fine.history['categorical_accuracy']
    val_acc = history_fine.history['val_categorical_accuracy']

    loss = history_fine.history['loss']
    val_loss = history_fine.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.ylim([0.8, 1])
    plt.plot([INITIAL_EPOCHS - 1, INITIAL_EPOCHS - 1], plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.ylim([0, 1.0])
    plt.plot([INITIAL_EPOCHS - 1, INITIAL_EPOCHS - 1], plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()


from dataset_generator import get_data

if __name__ == '__main__':
    (train_ds, val_ds), class_names = get_data()
    # train_ds = train_ds.batch(BATCH_SIZE)
    # val_ds = val_ds.batch(BATCH_SIZE)
    model, base_model = config_model()
    compile_model(model)
    train_model(model, train_ds, val_ds)
