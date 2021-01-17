import os
import re
import sys
import cv2 as cv
import tensorflow as tf
import logging

logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

logger.debug(tf.__version__)
numbers = re.compile(r'(\d+)')
tf.compat.v1.disable_eager_execution()

# utility sort function
def numerical_sort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def extract_single_video(path):
    vid = cv.VideoCapture(path)
    frames_path = path.split('.')[0]
    logger.debug(frames_path)
    if not os.path.isdir(frames_path):
        os.mkdir(frames_path)
        logging.debug('path not exist')
    logger.debug('frames_path: {}'.format(frames_path))
    frame_num = 0
    while True:
        success, frame = vid.read()
        if not success:
            logger.debug('fail: {}'.format(frame_num))
            break
        temp_name = os.path.join(frames_path, os.path.basename(path).split('.')[0] + '_f' + str(frame_num) + '.jpg')
        cv.imwrite(temp_name, frame)
        frame_num += 1
    vid.release()
    return


def get_file_path():
    paths = {
        "Windows": "D:/_assets/video/sahil_videos",
        "Linux": "/media/aadi/Library1/_assets/video/sahil_videos"
    }
    import platform
    return paths.get(platform.system())


def main():
    video_path = get_file_path()
    for file in sorted(os.listdir(video_path), key=numerical_sort):
        if file.endswith('.mp4'):
            extract_single_video(os.path.join(video_path, file))
        break


if __name__ == '__main__':
    main()
