import os
import plotly.express as px
import pandas
import pandas as pd
import numpy as np
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from datagen import dataset_generator
from processingutils.wavelettransform import transform_array, vis_array

DATA_DIR = "/home/aadidev/projects/player-cricket-detect/data/img/sahil_categories"
DATA_CAT_BATSMAN = os.path.join(DATA_DIR, 'batsman')[:10]
DATA_CAT_OTHERS = os.path.join(DATA_DIR, 'others')[:10]

batsman, batsman_fname, batsman_labels = [], [], []
others, others_fname, others_labels = [], [], []

CATEGORY_DICT = {0: 'batsman', 1: 'others'}
NUM_FILES = 10

data = dataset_generator.get_raw_img(num=10)

import matplotlib.pyplot as plt




# everything works up to here
def show_single_type_wavelet(show=False):
    '''
    :param show: boolean whether or not to show wavelet decomposition for 6 frames
    :return: None
    '''
    if not show: return None
    vis = vis_array(X_train)
    ROWS = 2
    COLS = 3

    fig = make_subplots(rows=ROWS, cols=COLS)
    row = 1
    col = 0

    for i in vis:
        if col >= COLS:
            row += 1
            col = 0
        col += 1
        if row > ROWS: break
        fig_i = go.Heatmap(z=i[0], colorscale='gray', colorbar=None)
        fig.add_trace(fig_i, row, col)

    fig.update_yaxes(autorange='reversed', scaleanchor='x', constrain='domain')
    fig.update_xaxes(constrain='domain')
    fig.update_layout(height=1000, width=1500, title_text="Approximation: Bior 1.1")
    fig.show()
    return None

show_single_type_wavelet(False)

images = np.array(data['images'])
labels = np.array(data['labels'])

single_frame = images[0]
from processingutils.wavelettransform import hist_single, hist
hist_single(single_frame, show=True)
count = hist(images, show=True)
# everything works up to here

quit()


def split_data(data: pandas.DataFrame, ratio=0.7):
    partition = int(len(data) * ratio)
    X_train, X_test = data['images'].iloc[:partition], data['images'].iloc[partition:]
    y_train, y_test = data.iloc[:partition, 1], data.iloc[partition:, 1]

    y_train = np.array(y_train).ravel()
    y_test = np.array(y_test).ravel()

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = split_data(data)



quit()

X_train = transform_array(X_train)
X_test = transform_array(X_test)

mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(y_train)
print(y_train)
print(np.array(X_train).shape)

sgd_clf = svm.SVC(random_state=42, max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train, y_train)
quit()


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
