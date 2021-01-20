import os
import sys

import tensorflow as tf
import logging
import gc
import psutil
from imageai.Detection import ObjectDetection
from os.path import join
from time import perf_counter

from processingutils.util import VIDEOS_PATH, config_session_tf, numerical_sort, split_frame_name_ball, _logger_dev
from video_to_frames import FRAMES_TARGET_PATH

logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# config to allow memory growth and max GPU buffer size
tf.compat.v1.keras.backend.set_session(config_session_tf())
# Required for imageAI
tf.compat.v1.disable_eager_execution()

DETECTOR_MODEL_PATH = '/home/aadi/PycharmProjects/player-cricket-detect/data/savedmodels/yolo.h5'
EXTRACTED_OBJS_PATH = '/home/aadi/PycharmProjects/player-cricket-detect/data/img/sahil_extracted_objs'
EXTRACTED_FRAMES_PATH = '/home/aadi/PycharmProjects/player-cricket-detect/data/img/sahil_frames'


def init_detector(detector_model_path=DETECTOR_MODEL_PATH):
    """
    Configures YOLOv3 object detector for people detection and returns detector with custom_objects
    :param detector_model_path: path to .h5 model
    :return: ImageAI object detector
    """
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(detector_model_path)
    detector.loadModel()
    custom_objects = detector.CustomObjects(person=True)
    return detector, custom_objects


DETECT = init_detector()


# def get_objects_per_frame(frame, output_path=None, detect=DETECT):
#     """
#     :param frame: Input image frame (single)
#     :param output_path: folder to hold folder of detected objects (persons)
#     :param detect: object detector model
#     :return: detections and objects
#     """
#     video_name = VIDEOS_PATH.split('.')[0].split('/')[-1]
#     output_image_path = join(EXTRACTED_OBJS_PATH, video_name, frame) if output_path else None
#     input_image = frame
#     detector, custom_objects = detect
#
#     detections, objects = detector.detectObjectsFromImage(
#         custom_objects=custom_objects,
#         input_image=input_image,
#         minimum_percentage_probability=30,
#         extract_detected_objects=False
#     )
#
#     return detections, objects


def get_objects_per_frame_path(frame, output_path=None, detect=DETECT, ):
    """Extracts objects from frame and outputs cropped images into folder
    :param frame: input frame to perform object detection on
    :param output_path: path to hold folder of extracted output objects
    :param detect: object detector
    :return: detections and objects as list
    """
    start = perf_counter()
    video_name = frame.split('.')[0].split('/')[-1]
    output_image_path = join(EXTRACTED_OBJS_PATH, video_name)
    input_image_path = join(EXTRACTED_FRAMES_PATH, frame)

    detector, custom_objects = detect

    detections, objects = detector.detectObjectsFromImage(
        custom_objects=custom_objects,
        input_image=input_image_path,
        output_image_path=output_image_path,
        minimum_percentage_probability=30,
        extract_detected_objects=True
    )

    logger.debug('file: {}, virtual_memory().percent: {}, cpu_percent: {}, time: {}'.format(
        os.path.basename(frame),
        psutil.virtual_memory().percent,
        psutil.cpu_percent(),
        perf_counter() - start
    ))

    gc.collect()
    return detections, objects


if __name__ == '__main__':
    files = []
    time_taken = []
    detector, custom_objects = init_detector()

    BALL_NUM_RANGE = [i for i in range(45, 46)]
    print(BALL_NUM_RANGE)

    for i, file in enumerate(sorted(os.listdir(FRAMES_TARGET_PATH), key=numerical_sort)):
        try:
            if split_frame_name_ball(file) in BALL_NUM_RANGE:
                try:
                    get_objects_per_frame_path(
                        os.fsdecode(file),
                        detect=(detector, custom_objects))
                except Exception as e:
                    logger.error('get_objects_per_frame(): {}'.format(e))
                    continue
        except TypeError as t:
            logger.debug('TypeError: ' + str(t))
            continue


