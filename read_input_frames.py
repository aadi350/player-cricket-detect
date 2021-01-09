import logging
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from util import numerical_sort

TARGET_SIZE = (720, 1280, 3)
CAT_PATH = 'D:/_assets/img/sahil_frames'


def load_video_frames(path=CAT_PATH, num_frames=None, output_names=False):
    frame_list = []
    frame_dict = {}
    try:
        for i, file in enumerate(sorted(os.listdir(path), key=numerical_sort)):
            if file.endswith('.jpg'):
                logging.debug(os.path.join(path, file))
                frame = load_img(os.path.join(path, file), target_size=TARGET_SIZE)
                frame_list.append(frame)
                frame_dict[str(file)] = frame
            if num_frames is not None:
                if i >= num_frames - 1:
                    return frame_list, frame_dict.keys()
        return np.array(frame_list), frame_dict.keys()
    except ImportError:
        logging.error('Video read fail')


