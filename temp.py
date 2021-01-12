import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import ImageOps, Image
from tensorflow.keras.layers.experimental.preprocessing import Resizing
from read_input_frames import load_video_frames, load_video_frames_batsman
from util import draw_boxes, display_image
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation, Resizing

BLOCK = False

gpus = tf.config.experimental.list_physical_devices('GPU')
# tensorflow config
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


def _load_ssd_mobilenet(handle):
    print('starting load...')
    detector = hub.load(handle).signatures['default']
    print('loaded: {}'.format(type(detector)))
    return detector


def bbox_detector_single_frame(detector, frame):
    img = Image.fromarray(frame)
    pil_image = ImageOps.fit(img, (256, 256), Image.ANTIALIAS)
    pil_image_rgb = pil_image.convert("RGB")
    converted_img = tf.image.convert_image_dtype(
        np.array(img), dtype=tf.float32)[tf.newaxis, ...]
    print('converted_img - type: {} shape: {}'.format(type(converted_img), converted_img.shape))
    result = detector(converted_img)
    result = {key: value.numpy() for key, value in result.items()}
    print("Found %d objects." % len(result["detection_scores"]))
    detection_class_entities = result["detection_class_entities"]
    detection_scores = result['detection_scores']
    detection_boxes = result['detection_boxes']

    person_detection_scores = []
    person_class_entities = []
    person_bounding_boxes = []
    for i, entity in enumerate(detection_class_entities):
        if detection_class_entities[i] == b'Person':
            person_class_entities.append(detection_class_entities[i])
            person_bounding_boxes.append(detection_boxes[i])
            person_detection_scores.append(detection_scores[i])

    image_with_boxes = draw_boxes(
        np.array(img),
        np.array(person_bounding_boxes),
        np.array(person_class_entities),
        np.array(person_detection_scores))
    display_image(image_with_boxes)

    return result


class BBoxLayer(tf.keras.layers.Layer):
    def __init__(self, units=1, input_dim=(1, None, None, 3)):
        super(BBoxLayer, self).__init__()
        self._handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
        self._detector = self._load_ssd_mobilenet()

    def call(self, img):
        result = self._detector(img)
        detection_class_entities = result["detection_class_entities"]
        detection_scores = result['detection_scores']
        detection_boxes = result['detection_boxes']
        person_detection_scores = []
        person_class_entities = []
        person_bounding_boxes = []
        for i, entity in enumerate(detection_class_entities):
            if detection_class_entities[i] == b'Person':
                person_class_entities.append(detection_class_entities[i])
                person_bounding_boxes.append(detection_boxes[i])
                person_detection_scores.append(detection_scores[i])

        return person_bounding_boxes

    def _load_ssd_mobilenet(self):
        return hub.load(self._handle).signatures['default']


class Rescale(tf.keras.layers.Layer):
    def __init__(self, units=1, input_dim=()):
        super(Rescale, self).__init__()

    def call(self, image, bboxes):
        rescaled = []
        for box in bboxes:
            im_width, im_height, _ = image.shape
            ymin, xmin, ymax, xmax = tuple(box)
            (left, right, bottom, top) = (
                int(xmin * im_width),
                int(xmax * im_width),
                int(ymin * im_height),
                int(ymax * im_height))
            rescaled.append([left, right, top, bottom])
        return rescaled, image


class CropToSize(tf.keras.layers.Layer):
    def __init__(self, units=1, input_dim=()):
        super(CropToSize, self).__init__()

    def call(self, rescaled, image):
        cropped = []
        for r in rescaled:
            print(r)
            (left, right, top, bottom) = r
            cropped.append(image[bottom:top, left:right, :])

        print('len(cropped) : {}'.format(len(cropped)))
        return cropped


# resizing boxes to match image

frames, frames_names = load_video_frames(num_frames=10)
# bbox_layer = tf.function(_load_ssd_mobilenet(SSD_MODULE_HANDLE))
bbox_layer = BBoxLayer()

single_frame = tf.image.convert_image_dtype(
    np.array(frames[0]), dtype=tf.float32)[tf.newaxis, ...]

# returns list of Tensors containing floats of bounding box dimensions
##  multiply by image size to obtain crop values
##  output format ymin, xmin, ymax, max
box = bbox_layer(single_frame)
r = Rescale()
r_out = r(frames[0], box)
print('r_out: {}'.format(r_out))
c = CropToSize()
c_out = c(r_out[0], r_out[1])

print('c_out: {}'.format(c_out))
plt.imshow(c_out[0])
plt.show()

# SSD_MODULE_HANDLE = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
# CAT_PATH = '/media/aadi/Library1/_assets/img/sahil_frames'
#
# model = tf.keras.models.load_model('./very_small.h5')
#
# resize = Resizing(244, 244)
#
#
#
# data_augmentation = tf.keras.Sequential([
#     RandomFlip('horizontal'),
#     RandomRotation(0.2),
# ])
#
# # FOR MODEL.PY
# DATA_DIR = "/media/aadi/Library1/_assets/video/sahil_categories"
# AUTOTUNE = tf.data.AUTOTUNE
# NUM_CLASSES = 2
# BATCH_SIZE = 1
# image_height = 224
# image_width = 224
# IMAGE_SIZE = (image_width, image_height)
# IMAGE_SIZE_DEPTH = (image_width, image_height, 3)
# VALIDATION_SPLIT = 0.4
# LABEL_MODE = 'categorical'
# COLOR_MODE = 'rgb'
# SEED = 42
# EPOCHS = 1
#
# # Model configuration
# preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
# base_model = tf.keras.applications.MobileNetV2(input_shape=IMAGE_SIZE_DEPTH,
#                                                include_top=False,
#                                                weights='imagenet')
# base_model.trainable = False
#
# resize_layer = Resizing(224, 224, name='resize')
# global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
# prediction_layer = tf.keras.layers.Dense(NUM_CLASSES, activation='sigmoid')
#
# # Model definition
# inputs = tf.keras.Input(shape=(None, None, 3))
# x = resize_layer(inputs)
# x = data_augmentation(x)
# x = preprocess_input(x)
# x = base_model(x, training=False)
# x = global_average_layer(x)
# x = prediction_layer(x)
# outputs = tf.keras.layers.Dropout(0.3)(x)
# model = tf.keras.Model(inputs, outputs)
#
# base_learning_rate = 0.0001
# model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
#               loss='categorical_crossentropy',
#               metrics=['categorical_accuracy'])
