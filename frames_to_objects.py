import os
import tensorflow as tf
import logging
import gc
import numpy as np
from imageai.Detection import ObjectDetection
from logging import info, debug, warning, error
from os.path import join
from time import perf_counter

from util import frames_path
from util import config_session_tf, numerical_sort, split_frame_name_ball, split_frame_name


# set logging level
logging.basicConfig(level=logging.DEBUG)
# config to allow memory growth and max GPU buffer size
tf.compat.v1.keras.backend.set_session(config_session_tf())
# Required for imageAI
tf.compat.v1.disable_eager_execution()


def init_detector():
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath('D:/_assets/models/yolo.h5')
    detector.loadModel()
    custom_objects = detector.CustomObjects(person=True)

    return detector, custom_objects


DETECT = init_detector()


def get_objects_per_frame(frame, output_path=None, detect=DETECT):
    output_image_path = join(frames_path.split(
        '.')[0], frame) if output_path else None
    input_image = frame
    detector, custom_objects = detect

    detections, objects = detector.detectCustomObjectsFromImage(
        custom_objects=custom_objects,
        input_image=input_image,
        minimum_percentage_probability=30,
        extract_detected_objects=False
    )

    return detections, objects


def get_objects_per_frame_path(frame, output_path=None, detect=DETECT, ):
    start = perf_counter()
    output_image_path = join(frames_path.split(
        '.')[0], frame) if output_path else None
    input_image_path = join(frames_path, frame)

    info('output_image_path: {}'.format(output_image_path))

    detector, custom_objects = detect

    detections, objects = detector.detectCustomObjectsFromImage(
        custom_objects=custom_objects,
        input_image=input_image_path,
        output_image_path=output_image_path,
        minimum_percentage_probability=30,
        extract_detected_objects=True
    )

    debug('file: {}, virtual_memory().percent: {}, cpu_percent: {}, time: {}'.format(
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

    BALL_NUM_RANGE = [i for i in range(42, 43)]
    print(BALL_NUM_RANGE)

    for i, file in enumerate(sorted(os.listdir(frames_path), key=numerical_sort)):
        try:
            if split_frame_name_ball(file) in BALL_NUM_RANGE:
                print('i: {}, file: {}'.format(i, os.fsdecode(file)))
                try:
                    get_objects_per_frame_path(
                        os.fsdecode(file),
                        detect=(detector, custom_objects))
                except Exception as e:
                    error('get_objects_per_frame(): {}'.format(e))
                    pass
        except TypeError as t:
            debug(t)
            continue
