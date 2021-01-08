import logging
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from util import numerical_sort

TARGET_SIZE = (244, 244, 3)


def load_single_video_frames(path):
    frame_list = []
    try:
        for file in sorted(os.listdir(path), key=numerical_sort):
            if file.endswith('.jpg'):
                print(os.path.join(path, file))
                frame = load_img(os.path.join(path, file), target_size=TARGET_SIZE, color_mode='rgb')
                frame_temp = img_to_array(frame)
                frame_list.append(frame_temp)
        return np.array(frame_list)
    except ImportError:
        logging.error('Video read fail')
