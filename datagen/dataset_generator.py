import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Parameter Inputs
DATA_DIR = "/home/aadi/PycharmProjects/player-cricket-detect/data/img/sahil_categories"
AUTOTUNE = tf.data.AUTOTUNE
NUM_CLASSES = 2
BATCH_SIZE = 32
image_height = 224
image_width = 224
IMAGE_SIZE = (image_width, image_height)
IMAGE_SIZE_DEPTH = (image_width, image_height, 3)
VALIDATION_SPLIT = 0.4
LABEL_MODE = CLASS_MODE = 'categorical'
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

def get_datagen(data_dir=DATA_DIR):
    datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       vertical_flip=True,
                                       validation_split=0.3)  # set validation split

    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=IMAGE_SIZE,
        color_mode=COLOR_MODE,
        class_mode=CLASS_MODE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
        save_to_dir=None,
        save_prefix='',
        save_format='jpg',
        follow_links=False,
        subset='training',
        interpolation='nearest'
    )

    val_gen = datagen.flow_from_directory(
        data_dir,  # same directory as training data
        target_size=IMAGE_SIZE,
        color_mode=COLOR_MODE,
        class_mode=CLASS_MODE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
        save_to_dir=None,
        save_prefix='',
        save_format='jpg',
        follow_links=False,
        subset='validation',
        interpolation='nearest'
    )

    return (train_gen, val_gen), train_gen.class_indices

if __name__ == '__main__':
    get_data()
