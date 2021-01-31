import numpy as np
from skimage.transform import integral_image
from skimage.feature import haar_like_feature
from skimage.feature import haar_like_feature_coord
from skimage.feature import draw_haar_like_feature
import matplotlib.pyplot as plt
from PIL import Image
import numpy
import cv2
from dask import delayed

img = np.ones((5, 5), dtype=np.uint8)
img_ii = integral_image(img)
feature = haar_like_feature(img_ii, 0, 0, 5, 5, ['type-3-x', 'type-3-y'])

img = Image.open('data/img/sahil_categories_by_video/Ball0002_Inn1/batsman/Ball0002_Inn1_107.jpg')
img = np.asarray(img)
img = cv2.resize(img, dsize=(100, 100))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img = img.astype(np.uint64)
img = img/255


# feature_types = [
#                  'type-3-x', 'type-3-y',
#                  ]

# img

images = [img]

@delayed
def extract_feature_image(img, feature_type, feature_coord=None):
    """Extract the haar feature for the current image"""
    ii = integral_image(img)
    return haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1],
                             feature_type=feature_type,
                             feature_coord=feature_coord)

X = delayed(extract_feature_image(img, feature_types) for img in images)
# Compute the result
t_start = time()
X = np.array(X.compute(scheduler='threads'))
time_full_feature_comp = time() - t_start

# coord, _ = haar_like_feature_coord(img.shape[0], img.shape[1], 'type-3-x')
# haar_feature = draw_haar_like_feature(img, 0, 0,
#                                       img.shape[0],
#                                       img.shape[1],
#                                       coord,
#                                       max_n_features=1,
#                                       random_state=0)

# haar_feature = draw_haar_like_feature(img, 0, 0, 224, 224, coord, max_n_features=1)
# plt.imshow(haar_feature)

# fig, axs = plt.subplots(1, 2)
# for ax, img, feat_t in zip(np.ravel(axs), images, feature_types):
#     coord, _ = haar_like_feature_coord(img.shape[0], img.shape[1], feat_t)
#     haar_feature = draw_haar_like_feature(img, 0, 0,
#                                           img.shape[0],
#                                           img.shape[1],
#                                           coord,
#                                           max_n_features=1,
#                                           random_state=0)
#     ax.imshow(haar_feature)
#     ax.set_title(feat_t)
#     ax.set_xticks([])
#     ax.set_yticks([])
