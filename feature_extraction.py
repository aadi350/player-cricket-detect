import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from skimage.feature import hog
from tensorflow.python.keras.layers import Resizing, RandomRotation, RandomFlip
from dataset_generator import get_data
from dataset_generator import BATCH_SIZE
from read_input_frames import load_video_frames_batsman, load_video_frames_other
from model import get_callbacks

IMG_DIRECTORY = '/media/aadi/Library1/_assets/img'
FRAME_DIRECTORY = IMG_DIRECTORY + '/sahil_frames'
SHOW_PLOTS = False

# Visualise magnitude spectrum
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


# Plots feature description for 10 frames
i = 1
for frame in frames[:10]:
    if i == 6: i = 11
    if i > 20: break
    feature_plot = get_magnitude_spectrum(frame=frame, channel=None)
    # feature_plot = get_hog(frame)
    plt.subplot(4, 5, i), plt.imshow(frame)
    plt.title('Input Image: ' + str(i)), plt.xticks([]), plt.yticks([])
    plt.subplot(4, 5, i + 5), plt.imshow(feature_plot, cmap='gray')
    plt.title('Feature Plot'), plt.xticks([]), plt.yticks([])
    i += 1

plt.suptitle('Magnitude Spectrum for Frames')
plt.show(block=SHOW_PLOTS)
plt.clf()

i = 1
for frame in frames[:10]:
    if i == 6: i = 11
    if i > 20: break
    # feature_plot = get_magnitude_spectrum(frame=frame, channel=None)
    feature_plot = get_hog(frame)
    plt.subplot(4, 5, i), plt.imshow(frame)
    plt.title('Input Image: ' + str(i)), plt.xticks([]), plt.yticks([])
    plt.subplot(4, 5, i + 5), plt.imshow(feature_plot, cmap='gray')
    plt.title('Feature Plot'), plt.xticks([]), plt.yticks([])
    i += 1
plt.suptitle('HoG for Frames')
plt.show(block=SHOW_PLOTS)
plt.clf()

train_hog = []
val_hog = []

train_labels = []
val_labels = []

(train_ds, val_ds), class_names = get_data()
train_ds = train_ds.unbatch().take(64)
val_ds = val_ds.unbatch().take(54)

# Feature calculation for dataset
for i, item in enumerate(train_ds):
    print('i: {}, item[0]: {}, item[1]: {}'.format(i, item[0].shape, item[1].shape))
    train_labels.append(item[1])
    hog_img = get_hog(np.array(item[0] / 255))
    hog_img = cv.merge((hog_img, hog_img, hog_img))
    train_hog.append(hog_img)
    print('{}: hog.shape: {}'.format(i, hog_img.shape))

print("\n\nvalidation\n")
for i, item in enumerate(val_ds):
    print('i: {}, item[0]: {}, item[1]: {}'.format(i, item[0].shape, item[1].shape))
    val_labels.append(item[1])
    hog_img = get_hog(np.array(item[0] / 255))
    hog_img = cv.merge((hog_img, hog_img, hog_img))
    val_hog.append(hog_img)
    print('{}: {}'.format(i, hog_img.shape))

print('len(train_labels): {}, len(train_hog): {}'.format(len(train_labels), len(train_hog)))

images = train_hog
labels = train_labels
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i])
    plt.title(str(labels[i]))
    plt.axis("off")
plt.suptitle('(line 96): HoG')
# TODO save image
plt.show(block=SHOW_PLOTS)
plt.clf()

# Re-dataset data
train_ds_hog = tf.data.Dataset.from_tensor_slices((train_hog, train_labels))
train_ds_hog = train_ds_hog.batch(16)
val_ds_hog = tf.data.Dataset.from_tensor_slices((val_hog, val_labels))
val_ds_hog = val_ds_hog.batch(16)

# Model
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.layers import Flatten

image_height = 224
image_width = 224
IMAGE_SIZE = (image_width, image_height)
IMAGE_SIZE_DEPTH = (image_width, image_height, 3)
NUM_CLASSES = 2

# Model configuration
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
# feature extraction into
base_model = tf.keras.applications.MobileNetV2(input_shape=IMAGE_SIZE_DEPTH,
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False
# batch data
image_batch, label_batch = next(iter(train_ds_hog))
feature_batch = base_model(image_batch)
resize_layer = Resizing(224, 224, name='resize')

# convert to single 1280 element
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
flatten_layer = Flatten()
prediction_layer = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
prediction_batch = prediction_layer(feature_batch_average)

data_augmentation = tf.keras.Sequential([
    RandomFlip('horizontal'),
    RandomRotation(0.2),
])

# Model definition
inputs = tf.keras.Input(shape=(None, None, 3))
# inputs = tf.keras.Input(shape=(None, None))  # hog
x = resize_layer(inputs)
# x = data_augmentation(x)

x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer='adam',
              loss=CategoricalCrossentropy(from_logits=True),
              metrics=['categorical_accuracy'])
# from model import config_model
#
# model = config_model()[0]

callbacks = get_callbacks()

history = model.fit(train_ds_hog,
                    epochs=10,
                    validation_data=val_ds_hog,
                    callbacks=callbacks,
                    verbose=1)

model.save('not_small')

test_loss, test_acc = model.evaluate(val_ds_hog, verbose=2)
# TODO compare hog with normal
