import logging
import os
import numpy as np
from PIL import Image
from processingutils.util import numerical_sort, _logger_dev

root = logging.getLogger()
_logger_dev(root)

CAT_PATH = '/home/aadi/PycharmProjects/player-cricket-detect/data/img/sahil_frames'
BATSMAN_CAT_PATH = '/home/aadi/PycharmProjects/player-cricket-detect/data/img/sahil_categories/batsman'
OTHERS_CAT_PATH = '/home/aadi/PycharmProjects/player-cricket-detect/data/img/sahil_categories/others'


# Loads only cropped images from batsman data directory
def load_video_frames_batsman(path=BATSMAN_CAT_PATH, num_frames=None):
    res = load_video_frames(path, num_frames)
    return res

# Loads only cropped images from others data directory
def load_video_frames_other(path=OTHERS_CAT_PATH, num_frames=None):
    return load_video_frames(path, num_frames)


# Loads uncropped from all video frames
# Returns array of frames and array of frame names
def load_video_frames(path=CAT_PATH, num_frames=None):
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
