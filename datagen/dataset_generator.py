import os

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from skimage import transform
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Parameter Inputs
CAT_HOG_PATH = '/home/aadidev/projects/player-cricket-detect/data/img/categories_hog'
DATA_DIR = "/home/aadidev/projects/player-cricket-detect/data/img/sahil_categories"
DATA_CAT_BATSMAN = os.path.join(DATA_DIR, 'batsman')
DATA_CAT_OTHERS = os.path.join(DATA_DIR, 'others')
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


def get_data(data_dir=DATA_DIR, batch=BATCH_SIZE, labelmode=LABEL_MODE):
    """One-hot encodes categories of input images, placing them
    into training and validation classes, accessed via dataset objects
    :param data_dir: contains all files in n folders corresponding to n classes
    :param batch: batch size
    :param labelmode: 'categorical' or None
    :return: train and validation datasets with class names
    NB: No augmentation performed
    :param data_dir:
    :param batch:
    :param labelmode:
    :return:
    """

    train_ds = image_dataset_from_directory(
        data_dir,
        color_mode=COLOR_MODE,
        validation_split=VALIDATION_SPLIT,
        subset='training',
        image_size=IMAGE_SIZE,
        batch_size=batch,
        label_mode=labelmode,
        seed=SEED
    )

    val_ds = image_dataset_from_directory(
        data_dir,
        color_mode=COLOR_MODE,
        validation_split=VALIDATION_SPLIT,
        subset='validation',
        image_size=IMAGE_SIZE,
        batch_size=batch,
        label_mode=labelmode,
        seed=SEED
    )

    class_names = train_ds.class_names
    return (train_ds, val_ds), class_names


def get_datagen(data_dir=DATA_DIR):
    """Function augments and one-hot encodes categories of input images, placing them
    into training and validation classes, accessed via datagenerators
    :param data_dir: contains all files in n folders corresponding to n classes
    :return: train and validaiton Keras datagenerator objects with class indices
    """
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


def get_raw_img(datadir=DATA_DIR, num=1000, grey=True):
    labels = []
    img_names = []
    images = []
    for file in os.listdir(DATA_CAT_BATSMAN)[:num]:
        file_path = os.path.join(DATA_CAT_BATSMAN, file)
        image = cv2.imread(file_path)
        if grey: image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        image = transform.resize(image, (224, 224), anti_aliasing=True)
        images.append(image)
        labels.append(1)
        del image

    for file in os.listdir(DATA_CAT_OTHERS)[:num]:
        file_path = os.path.join(DATA_CAT_OTHERS, file)
        image = cv2.imread(file_path)
        if grey: image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        image = transform.resize(image, (224, 224), anti_aliasing=True)
        images.append(image)
        labels.append(0)
        del image

    data = dict({'images': images, 'labels': labels})
    images_df = pd.DataFrame(data)

    return images_df


def get_data_hog(datadir=CAT_HOG_PATH):
    return get_datagen(datadir)


if __name__ == '__main__':
    print(get_data().__doc__)