import os
import re
import cv2 as cv
import tensorflow as tf
from imageai.Detection import ObjectDetection

print(tf.__version__)
numbers = re.compile(r'(\d+)')
tf.compat.v1.disable_eager_execution()


def numerical_sort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

video_path = '/media/aadi/Library1/_assets/video/sahil_videos'

def extract_frames(path):
    print(path)
    vid = cv.VideoCapture(path)
    frames_path = os.path.join(video_path, 'frames')
    # os.mkdir(frames_path)
    print(frames_path)
    i = 0
    while True:
        success, frame = vid.read()
        if not success:
            print('fail: {}'.format(i))
            break
        temp_name = os.path.join(path.split('.')[0] + '_f' + str(i) + '.jpg')
        cv.imwrite(temp_name, frame)
        i += 1

    vid.release()
    return


for file in sorted(os.listdir(video_path), key=numerical_sort):
    print(file)
    if file.endswith('.mp4'):
        extract_frames(os.path.join(video_path, file))

