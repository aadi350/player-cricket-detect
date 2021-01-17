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

VIDEO_PATH = '/home/aadi/PycharmProjects/player-cricket-detect/data/video/sahil_videos'
FRAMES_TARGET_PATH = '/home/aadi/PycharmProjects/player-cricket-detect/data/img/sahil_frames/'


# utility sort function
def numerical_sort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def extract_single_video(videopath, targetpath=FRAMES_TARGET_PATH):
    vid = cv.VideoCapture(videopath)
    video_name = (videopath.split('.')[0]).split('/')[-1]
    video_frames_path = os.path.join(targetpath, video_name)
    logger.debug('video_name: {}, video_frames_path: {}'.format(video_name, video_frames_path))
    if not os.path.isdir(video_frames_path):
        try:
            os.mkdir(video_frames_path)
            logging.debug('path not exist, path created')
            frame_num = 0
            while True:
                success, frame = vid.read()
                if not success:
                    logger.debug('fail: {}'.format(frame_num))
                    break
                temp_path = os.path.join(video_frames_path,
                                         os.path.basename(videopath).split('.')[0] + '_f' + str(frame_num) + '.jpg')
                logger.debug('temp_path: {}'.format(temp_path))
                cv.imwrite(temp_path, frame)
                frame_num += 1
            vid.release()
            return
        except Exception:
            logging.error('Could not create folder for: '.format(video_name))
            pass
        return


def main():
    if not os.path.isdir(FRAMES_TARGET_PATH):
        try:
            logging.debug('path not exist')
            os.mkdir(FRAMES_TARGET_PATH)
        except FileExistsError('cannot create frames folder'):
            pass

    for file in sorted(os.listdir(VIDEO_PATH), key=numerical_sort):
        if file.endswith('.mp4'):
            extract_single_video(os.path.join(VIDEO_PATH, file))


if __name__ == '__main__':
    main()
