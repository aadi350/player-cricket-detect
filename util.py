import tensorflow as tf
import re, os
import logging
from logging import FileHandler, DEBUG

# CONSTANTS
frames_path = '/media/aadi/Library1/_assets/video/sahil_videos'


# UTILITY FUNCTIONS
def numerical_sort(value):
    # sort by numerical part of file name
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


class DebugFileHandler(FileHandler):
    def __init__(self, filename, mode='a', encoding=None, delay=False):
        super().__init__(filename, mode, encoding, delay)

    def emit(self, record):
        if not record.levelno == DEBUG:
            return
        super().emit(record)


# FILE NAMING
def split_frame_name(filename):
    if filename.endswith('.jpg'):
        stem = filename.split('.')[0]
        inning_num = stem.split('_')[1].replace('Inn', '')
        ball_num = stem.split('_')[0].replace('Ball', '').lstrip('0')
        frame_num = stem.split('f')[-1]
        logging.debug('{}: {} {} {}'.format(filename, ball_num, inning_num, frame_num))
        return int(ball_num), int(inning_num), int(frame_num)
    else:
        raise TypeError('Not image: {}'.format(filename))


def split_frame_name_ball(filename):
    return split_frame_name(filename)[0]


# CONFIGURATION FUNCTIONS
def config_session_tf(allow_growth=True, per_process_gpu_memory_fraction=0.8):
    # config memory growth and max buffer size
    try:
        config = tf.compat.v1.ConfigProto(
            gpu_options=tf.compat.v1.GPUOptions(
                per_process_gpu_memory_fraction=per_process_gpu_memory_fraction
            ))

        config.gpu_options.allow_growth = allow_growth
        return tf.compat.v1.Session(config=config)
    except RuntimeError as e:
        logging.warning(e)
        return None


if __name__ == '__main__':
    for i in os.listdir(frames_path):
        print(i)
    print('utility function')
