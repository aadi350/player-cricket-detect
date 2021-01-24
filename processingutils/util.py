import tempfile
from urllib.request import urlopen
import numpy as np
import tensorflow as tf
import re
import sys
import logging
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageDraw, ImageColor, ImageFont
from six import BytesIO

frames_path = 'data/video/sahil_videos'


# UTILITY FUNCTIONS
def numerical_sort(value):
    # sort by numerical part of file name
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def _logger_dev(logger):
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return


# sets location for reading asset data
def get_file_path_prefix():
    paths = {
        "Windows": "D:/_assets",
        "Linux": "/media/aadi/Library1/_assets"
    }
    import platform
    return paths.get(platform.system())


"""
    Visualization code adapted from TF object detection API
"""


def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img


def display_image(image):
    fig = plt.figure(figsize=(20, 15))
    plt.grid(False)
    plt.imshow(image)
    plt.show()


def download_and_resize_image(url, new_width=256, new_height=256, display=False):
    _, filename = tempfile.mkstemp(suffix=".jpg")
    response = urlopen(url)
    image_data = response.read()
    image_data = BytesIO(image_data)
    pil_image = Image.open(image_data)
    pil_image = ImageOps.fit(
        pil_image, (new_width, new_height), Image.ANTIALIAS)
    pil_image_rgb = pil_image.convert("RGB")
    pil_image_rgb.save(filename, format="JPEG", quality=90)
    print("Image downloaded to %s." % filename)
    if display:
        display_image(pil_image)
    return filename


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color,
                               font,
                               thickness=4,
                               display_str_list=()):
    """
        Copies single bounding box to an image.
    """
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
               (left, top)],
              width=thickness,
              fill=color)

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = top + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                        (left + text_width, text_bottom)],
                       fill=color)
        draw.text((left + margin, text_bottom - text_height - margin),
                  display_str,
                  fill="black",)
                  #font=font)
        text_bottom -= text_height - 2 * margin


def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
    """Overlay labeled boxes on an image with formatted scores and label names."""
    colors = list(ImageColor.colormap.values())
    font = ImageFont.load_default()

    for i in range(min(boxes.shape[0], max_boxes)):

        # Unrolls tuple of bounding boxes and converts image from array to
        # np array of type uint8 with RGB color-map

        if scores[i] >= min_score:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            display_str = "{}: {}%".format(class_names[i].decode("ascii"),
                                           int(100 * scores[i]))
            color = colors[hash(class_names[i]) % len(colors)]
            image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
            draw_bounding_box_on_image(
                image_pil,
                ymin,
                xmin,
                ymax,
                xmax,
                color,
                font,
                display_str_list=[display_str])
            np.copyto(image, np.array(image_pil))
    return image


# FILE NAMING
def split_frame_name(filename):
    if filename.endswith('.jpg'):
        stem = filename.split('.')[0]
        inning_num = stem.split('_')[1].replace('Inn', '')
        ball_num = stem.split('_')[0].replace('Ball', '').lstrip('0')
        frame_num = stem.split('f')[-1]
        logging.debug('{}: {} {} {}'.format(
            filename, ball_num, inning_num, frame_num))
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
    raise NotImplementedError('utility function')
