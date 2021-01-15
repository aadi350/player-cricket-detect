import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import ImageOps, Image
from skimage.transform import resize
from skimage.feature import hog
from cv2.cv2 import resize

from read_input_frames import load_video_frames, load_video_frames_batsman
from util import draw_boxes, display_image
import logging

BLOCK = False

# tensorflow config
if tf.config.experimental.list_physical_devices('GPU'):
    gpus = tf.config.experimental.list_physical_devices('GPU')
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
    logging.info('starting load...')
    detector = hub.load(handle).signatures['default']
    logging.info('loaded: {}'.format(type(detector)))
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
            print('det_score: {}'.format(detection_scores[i]))
            if entity == b'Person' and detection_scores[i] > 0.15:
                person_class_entities.append(detection_class_entities[i])
                person_bounding_boxes.append(detection_boxes[i])
                person_detection_scores.append(detection_scores[i])

        # returns LIST of Tensors, shape(4,) dtype=float32 of an array
        return person_bounding_boxes

    def _load_ssd_mobilenet(self):
        return hub.load(self._handle).signatures['default']


class MatchScale(tf.keras.layers.Layer):
    def __init__(self, units=1, input_dim=()):
        super(MatchScale, self).__init__()

    def call(self, image, bboxes):
        rescaled = []
        for box_id, box in enumerate(bboxes):
            batch, im_height, im_width, _ = image.shape
            ymin, xmin, ymax, xmax = tuple(box)
            (left, right, bottom, top) = (
                int(xmin * im_width),
                int(xmax * im_width),
                int(ymin * im_height),
                int(ymax * im_height))
            rescaled.append([box_id, (left, right, top, bottom)])
        return rescaled, image


class CropToSize(tf.keras.layers.Layer):
    def __init__(self, units=1, input_dim=()):
        super(CropToSize, self).__init__()

    def call(self, image, rescaled):
        cropped = []
        for i, r in enumerate(rescaled[0]):
            print('r1:', r[1])
            (left, right, top, bottom) = r[1]
            print('crop_sshape', image[bottom:top, left:right, :].shape)
            cropped.append([i, image[bottom:top, left:right, :]])

        print('len(cropped) : {}'.format(len(cropped)))
        # Returns list of images with integer IDs
        return cropped


# resizing boxes to match image

frames, frames_names = load_video_frames(num_frames=10)
# bbox_layer = tf.function(_load_ssd_mobilenet(SSD_MODULE_HANDLE))
bbox_layer = BBoxLayer()

single_frame = tf.image.convert_image_dtype(
    np.array(frames[0]), dtype=tf.float32)[tf.newaxis, ...]

print('single_frame.shape: {}'.format(single_frame.shape))
classifier = tf.keras.models.load_model('./very_small.h5')

obj_res = []
bbox = bbox_layer(single_frame)

for i in bbox:
    print(i)

print(type(bbox))
print(bbox)
scale_match = MatchScale()
size_crop = CropToSize()

print(single_frame.shape)
bbox_scale_matched = scale_match(single_frame, bbox)
objects = size_crop(single_frame[0], bbox_scale_matched)
for i in bbox_scale_matched[0]:
    print('i in bbox_scale_matched[0]: {}'.format(i))

for num, i in enumerate(objects):
    print('i in objects:')
    person_id, img = i
    # fd, img  = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=True)
    print(img.shape)
    print(type(img))
    plt.subplot(3, 3, num + 1)
    plt.imshow(img)
    img = tf.image.convert_image_dtype(np.array(img), dtype=tf.float32)[tf.newaxis, ...]
    img = tf.image.convert_image_dtype(np.array(img), dtype=tf.float32)[..., tf.newaxis]
    res = classifier(img)
    plt.title(res[0].numpy())
    # print(res[0])
    # plt.imshow(img.numpy())
    #
    # plt.show(block=SHOW_ALL_PERSONS)

plt.show()

# for c in c_out:
#     batsman_tensor = classifier(c[1])
#     batsman_id = c[0]
#     if batsman_tensor[0] > batsman_tensor[1]:
#         # only add boundign box if obbject is a batsman
#         obj_res.append([batsman_id, batsman_tensor])
#
# # draw bbounding box per frame
#
# print('c_out: {}'.format(c_out))
# plt.imshow(c_out[0])
# plt.show()

# SSD_MODULE_HANDLE = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
# CAT_PATH = '/media/aadi/Library1/_assets/img/sahil_frames'
