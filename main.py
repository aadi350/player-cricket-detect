import os
import matplotlib.pyplot as plt
import pandas
import pandas as pd
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

batsman_hog_sum = pd.read_pickle('batsman_hog_sum')[:-2]
others_hog_sum = pd.read_pickle('others_hog_sum')[:-2]
fig_names = ['Batsman', 'Others']
sum_features = [batsman_hog_sum, others_hog_sum]

f, a = plt.subplots(1, 2)
a = a.ravel()
for idx, ax in enumerate(a):
    ax.hist(x=sum_features[idx], bins=15, log=True,
            alpha=0.7, rwidth=0.85)
    ax.set_title(fig_names[idx])
plt.tight_layout()
plt.savefig('Histogram of HoGs.jpg')
plt.show()


def split_data(data: pandas.DataFrame, ratio=0.7):
    partition = int(len(data) * ratio)
    print(data.shape)
    X_train, X_test = data.iloc[:partition, :-1], data.iloc[partition:, :-1]
    y_train, y_test = data.iloc[:partition, -1:], data.iloc[partition:, -1:]

    y_train = np.array(y_train).ravel()
    y_test = np.array(y_test).ravel()

    return X_train, X_test, y_train, y_test


data = pandas.read_pickle('hog_features')
from models.svc import PCAAnalysis as pca

ipca = pca.load_pca()

X_train, X_test, y_train, y_test = split_data(data)
X_test_pca = pca.transform_pca(X_test, ipca)


### FOR DEMO ONLY
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

classify.classify_batch(file_paths, visualise=True)
