import logging
import os
import numpy as np
from PIL import Image
from processingutils.util import numerical_sort, _logger_dev

root = logging.getLogger()
_logger_dev(root)

CAT_PATH = '/home/aadidev/projects/player-cricket-detect/data/img/sahil_frames'
BATSMAN_CAT_PATH = '/home/aadidev/projects/player-cricket-detect/data/img/sahil_categories/batsman'
OTHERS_CAT_PATH = '/home/aadidev/projects/player-cricket-detect/data/img/sahil_categories/others'


def load_video_frames_batsman(path=BATSMAN_CAT_PATH, num_frames=None):
    """
    :param path: path to batsman cropped images
    :param num_frames: number of batsman to load
    :return: list of frames and names
    """
    res = load_video_frames(path, num_frames)
    return res


# Loads only cropped images from others data directory
def load_video_frames_other(path=OTHERS_CAT_PATH, num_frames=None):
    """
    :param path: path to other cropped images
    :param num_frames: number of other to load
    :return: list of frames and names
    """
    return load_video_frames(path, num_frames)


def load_video_frames(path=CAT_PATH, num_frames=None):
    """
    :param path: path to frames/objectss
    :param num_frames: numbere of framess to return
    :return: list of frames and names
    """
    frame_list = []
    frame_names = []
    frame_dict = {}

    try:
        for i, file in enumerate(sorted(os.listdir(path), key=numerical_sort)):
            if file.endswith('.jpg'):
                full_path = os.path.join(path, file)
                frame = np.asarray(Image.open(full_path))
                frame_list.append(frame)
                frame_names.append(str(file))
                frame_dict[str(file)] = frame

            if num_frames is not None:
                if i >= num_frames - 1:
                    return frame_list, frame_names

        return frame_list, frame_names
    except ImportError:
        logging.error('Video read fail')


if __name__ == '__main__':
    raise NotImplementedError
