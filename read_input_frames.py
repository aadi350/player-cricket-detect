import logging
import os
from six import BytesIO
import PIL
import numpy as np
import tensorflow as tf
from PIL import Image
from util import numerical_sort

CAT_PATH = '/media/aadi/Library1/_assets/img/sahil_frames'
BATSMAN_CAT_PATH = '/media/aadi/Library1/_assets/video/sahil_categories/others'


def _process_path(fname):
    img_str = tf.io.read_file(fname)
    img = tf.image.decode_jpeg(img_str, channels=3)
    return img


def load_video_frames_batsman(path=BATSMAN_CAT_PATH, num_frames=None):
    return load_video_frames(path, num_frames)


def load_video_frames(path=CAT_PATH, num_frames=None):
    frame_list = []
    frame_names = []
    frame_dict = {}

    try:
        for i, file in enumerate(sorted(os.listdir(path), key=numerical_sort)):
            if file.endswith('.jpg'):
                logging.debug(os.path.join(path, file))

                full_path = os.path.join(path, file)
                frame = np.asarray(Image.open(full_path))
                frame_list.append(frame)
                frame_names.append(str(file))
                frame_dict[str(file)] = frame
            if num_frames is not None:
                if i >= num_frames - 1:
                    return frame_list, frame_names
        return np.asarray(frame_list, dtype=np.uint8), frame_dict.keys()
    except ImportError:
        logging.error('Video read fail')


load_video_frames_batsman(num_frames=5)
