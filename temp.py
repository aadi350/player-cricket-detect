import os
import tensorflow_hub as hub
import time
import tensorflow as tf
import numpy as np
import cv2 as cv
from PIL import ImageOps, Image
from util import draw_boxes, display_image
from DeepLab import DeepLabModel
from read_input_frames import load_video_frames

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


def run_detector_single_frame(detector, frame):
    img = Image.fromarray(frame)
    pil_image = ImageOps.fit(img, (256, 256), Image.ANTIALIAS)
    pil_image_rgb = pil_image.convert("RGB")
    converted_img = tf.image.convert_image_dtype(
        np.array(img), dtype=tf.float32)[tf.newaxis, ...]

    start_time = time.perf_counter()
    result = detector(converted_img)
    end_time = time.perf_counter()
    print(result)
    result = {key: value.numpy() for key, value in result.items()}
    print("Found %d objects." % len(result["detection_scores"]))
    print("Inference time: ", end_time - start_time)

    detection_class_entities = result["detection_class_entities"]
    print(detection_class_entities)
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
        self._handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
        self._detector = self._load_ssd_mobilenet()

    def call(self, img):
        return self._detector(img)

    def _load_ssd_mobilenet(self):
        return hub.load(self._handle).signatures['default']


SSD_MODULE_HANDLE = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
CAT_PATH = '/media/aadi/Library1/_assets/img/sahil_frames'
frames, names = load_video_frames(path=CAT_PATH, num_frames=10)

detect_block = tf.function(_load_ssd_mobilenet(SSD_MODULE_HANDLE))
model = tf.keras.layers.Layer(detect_block)
res = run_detector_single_frame(detector=detect_block, frame=frames[0])
model = tf.keras.models.load_model('./very_small.h5')
