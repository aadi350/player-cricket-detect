import logging
import os
import pickle
import csv

import pandas
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import skimage
from skimage.transform import resize
from skimage.io import imread
from skimage import color
import numpy as np

DATA_DIR = "/home/aadidev/projects/player-cricket-detect/data/img/sahil_categories"
DATA_CAT_BATSMAN = os.path.join(DATA_DIR, 'batsman')
DATA_CAT_OTHERS = os.path.join(DATA_DIR, 'others')

batsman, batsman_fname, batsman_labels = [], [], []
others, others_fname, others_labels = [], [], []

CATEGORY_DICT = {0: 'batsman', 1: 'others'}
NUM_FILES = 10

PPC = 4
hog_images = []
hog_features = []
img_labels = []
labels = []

# for file in os.listdir(DATA_CAT_BATSMAN)[:1000]:
#     print('bat',file)
#     file_path = os.path.join(DATA_CAT_BATSMAN, file)
#     image = skimage.io.imread(file_path)
#     image = color.rgb2gray(image)
#     image = skimage.transform.resize(image, (224, 224), anti_aliasing=True)
#     fd, hog_image = hog(
#         image,
#         orientations=9,
#         pixels_per_cell=(PPC, PPC),
#         cells_per_block=(4, 4),
#         block_norm='L2',
#         visualize=True
#     )
#     labels.append([1,0])
#     img_labels.append(file_path)
#     hog_images.append(hog_image)
#     hog_features.append(fd)
#     del image
#     del hog_image
#     del fd
#
#
# for file in os.listdir(DATA_CAT_OTHERS)[:1000]:
#     print(file)
#     file_path = os.path.join(DATA_CAT_OTHERS, file)
#     image = skimage.io.imread(file_path)
#     image = color.rgb2gray(image)
#     image = skimage.transform.resize(image, (224, 224), anti_aliasing=True)
#     fd, hog_image = hog(
#         image,
#         orientations=9,
#         pixels_per_cell=(PPC, PPC),
#         cells_per_block=(4, 4),
#         block_norm='L2',
#         visualize=True
#     )
#     labels.append([0,1])
#     img_labels.append(file_path)
#     hog_images.append(hog_image)
#     hog_features.append(fd)
#     del image
#     del hog_image
#     del fd
#
# df = np.hstack((hog_features, labels))
# print(df.shape)
# img_names_df = pd.DataFrame(img_labels)
# img_names_df.to_csv('image_paths.csv', encoding='utf-8')
# hog_df = pd.DataFrame(df)
# hog_df.to_pickle('hog_features')


# np.random.shuffle(df)

from models.svc import PCAAnalysis as pca
from models.svc import gridsearch

logging.debug('loading hog...')
data = pd.read_pickle('hog_features')
logging.debug('hog loaded')

# def split_data(data: pandas.DataFrame, ratio=0.7):
# 	partition = int(len(data) * ratio)
# 	print(data.shape)
# 	X_train, X_test = data.iloc[:partition, :-1], data.iloc[partition:, :-1]
# 	y_train, y_test = data.iloc[:partition, -1:], data.iloc[partition:, -1:]
#
# 	y_train = np.array(y_train).ravel()
# 	y_test = np.array(y_test).ravel()
#
# 	return X_train, X_test, y_train, y_test
#
#
# X_train, X_test, y_train, y_test = split_data(data)[:10]

# # _ = pca.fit_pca(X_train)
# #
# ipca = pca.load_pca()
# pca.plot_explained_variance(ipca=ipca, block=True)
# # X_train_pca = pca.transform_pca(X_train, ipca)
# # pickle.dump(X_train_pca, open('X_train_pca.pkl', 'wb'))
#
# X_train_pca = pickle.load(open('X_train_pca.pkl', 'rb'))
# gridsearch.fit(X_train_pca, y_train)


# plt.imshow(X_train[0])
# plt.show()


from models.svc import classify

file_paths = [
    '/home/aadidev/projects/player-cricket-detect/data/img/sahil_categories/batsman/Ball0001_Inn1_11.jpg',
    '/home/aadidev/projects/player-cricket-detect/data/img/sahil_categories/others/Ball0001_Inn1_17.jpg',
    '/home/aadidev/projects/player-cricket-detect/data/img/sahil_categories/batsman/Ball0001_Inn1_18.jpg',
    '/home/aadidev/projects/player-cricket-detect/data/img/sahil_categories/batsman/Ball0001_Inn1_103.jpg',
    '/home/aadidev/projects/player-cricket-detect/data/img/sahil_categories/batsman/Ball0001_Inn1_139.jpg',
    '/home/aadidev/projects/player-cricket-detect/data/img/sahil_categories/batsman/Ball0002_Inn2_133.jpg',
    '/home/aadidev/projects/player-cricket-detect/data/img/sahil_categories/others/Ball0002_Inn2_193.jpg',
    '/home/aadidev/projects/player-cricket-detect/data/img/sahil_categories/others/Ball0002_Inn2_381.jpg',
    '/home/aadidev/projects/player-cricket-detect/data/img/sahil_categories/others/Ball0003_Inn1_125.jpg',
]

classify.classify_batch(file_paths)
