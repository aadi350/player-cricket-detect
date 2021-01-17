import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Parameter Inputs
DATA_DIR = "data/img/sahil_categories"
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


def get_data(data_dir=DATA_DIR):
    train_ds = image_dataset_from_directory(
        data_dir,
        color_mode=COLOR_MODE,
        validation_split=VALIDATION_SPLIT,
        subset='training',
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode=LABEL_MODE,
        seed=SEED
    )

    val_ds = image_dataset_from_directory(
        data_dir,
        color_mode=COLOR_MODE,
        validation_split=VALIDATION_SPLIT,
        subset='validation',
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode=LABEL_MODE,
        seed=SEED
    )

    class_names = train_ds.class_names
    return (train_ds, val_ds), class_names


if __name__ == '__main__':
    get_data()
