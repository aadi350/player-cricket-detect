import logging
import os
from PIL import Image
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from skimage.feature import hog
from datagen.dataset_generator import get_data
from read_input_frames import load_video_frames_batsman

IMG_DIRECTORY = '/home/aadi/PycharmProjects/player-cricket-detect/data/img'
CAT_PATH = '/home/aadi/PycharmProjects/player-cricket-detect/data/img/sahil_categories'
CAT_HOG_PATH = '/home/aadi/PycharmProjects/player-cricket-detect/data/img/categories_hog'
FRAME_DIRECTORY = IMG_DIRECTORY + '/sahil_frames'
SHOW_PLOTS = False

frames, _ = load_video_frames_batsman(num_frames=10)


def get_magnitude_spectrum(frame, channel=None):
    if channel:
        frame = frame[:, :, channel]
    else:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame = cv.resize(frame, (224, 224))
    f = np.fft.fft2(frame)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    return magnitude_spectrum


# function to return HoG
def get_hog(frame):
    fd, hog_image = hog(frame, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(3, 3), visualize=True, feature_vector=False, multichannel=True)
    return hog_image


def merge_hog(hog_img):
    return cv.merge((hog_img, hog_img, hog_img))


# FOR VISUALISATION ONLY
# Plots feature description for 10 frames
def show_hog(block=SHOW_PLOTS):
    for i, frame in enumerate(frames[:10]):
        if i == 6: i = 11
        if i > 20: break
        feature_plot = get_hog(frame)
        plt.subplot(4, 5, i + 1), plt.imshow(frame)
        plt.title('Input Image: ' + str(i)), plt.xticks([]), plt.yticks([])
        plt.subplot(4, 5, i + 5), plt.imshow(feature_plot, cmap='gray')
        plt.title('Feature Plot'), plt.xticks([]), plt.yticks([])
    plt.suptitle('HoG for Frames')
    plt.show(block=SHOW_PLOTS)
    plt.clf()


def show_mag_spectrum(block=SHOW_PLOTS):
    i = 1
    for frame in frames[:10]:
        if i == 6: i = 11
        if i > 20: break
        feature_plot = get_magnitude_spectrum(frame=frame, channel=None)
        plt.subplot(4, 5, i), plt.imshow(frame)
        plt.title('Input Image: ' + str(i)), plt.xticks([]), plt.yticks([])
        plt.subplot(4, 5, i + 5), plt.imshow(feature_plot, cmap='gray')
        plt.title('Feature Plot'), plt.xticks([]), plt.yticks([])
        i += 1
    plt.suptitle('Fourier Magnitude Spectrum for Frames')
    plt.show(block=block)
    plt.clf()


# Training
def lists_to_ds(train_features, train_labels):
    # Re-dataset data
    train_ds_hog = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
    train_ds_hog = train_ds_hog.batch(16)
    logging.info('len(train_labels): {}, len(train_features): {}'.format(len(train_labels), len(train_features)))
    return train_ds_hog


def get_hog_ds(take=None):
    (train_ds, val_ds), class_names = get_data()
    train_ds = train_ds.unbatch().take(take)
    val_ds = val_ds.unbatch().take(take)

    train_hog = []
    val_hog = []

    train_labels = []
    val_labels = []

    # Feature calculation for dataset
    for i, item in enumerate(train_ds):
        train_labels.append(item[1])
        hog_img = get_hog(np.array(item[0] / 255))
        hog_img = cv.merge((hog_img, hog_img, hog_img))
        train_hog.append(hog_img)

    print("\n\nvalidation\n")
    for i, item in enumerate(val_ds):
        val_labels.append(item[1])
        hog_img = get_hog(np.array(item[0] / 255))
        hog_img = cv.merge((hog_img, hog_img, hog_img))
        val_hog.append(hog_img)

    train_ds_hog = lists_to_ds(train_hog, train_labels)
    val_ds_hog = lists_to_ds(val_hog, val_labels)

    return train_ds_hog, val_ds_hog


def boost_contrast(img, ratio=3):
    return img * ratio


def write_hog_from_file():
    for category in os.listdir(CAT_PATH):
        if not os.path.exists(os.path.join(CAT_HOG_PATH, category)):
            os.makedirs(os.path.join(CAT_HOG_PATH, category))

        for file in os.listdir(os.path.join(CAT_PATH, category)):
            if file.endswith('jpg'):
                abspath = os.path.join(CAT_PATH, category, file)
                img = Image.open(abspath)
                hog_img = boost_contrast(get_hog(img))
                writepath = os.path.join(CAT_HOG_PATH, category, file)
                cv.imwrite(writepath, hog_img)

