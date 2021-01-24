import os
import pickle
import csv
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

for file in os.listdir(DATA_CAT_BATSMAN)[:1000]:
    print('bat',file)
    file_path = os.path.join(DATA_CAT_BATSMAN, file)
    image = skimage.io.imread(file_path)
    image = color.rgb2gray(image)
    image = skimage.transform.resize(image, (224, 224), anti_aliasing=True)
    fd, hog_image = hog(
        image,
        orientations=9,
        pixels_per_cell=(PPC, PPC),
        cells_per_block=(4, 4),
        block_norm='L2',
        visualize=True
    )
    labels.append([1,0])
    img_labels.append(file_path)
    hog_images.append(hog_image)
    hog_features.append(fd)
    del image
    del hog_image
    del fd


for file in os.listdir(DATA_CAT_OTHERS)[:1000]:
    print(file)
    file_path = os.path.join(DATA_CAT_OTHERS, file)
    image = skimage.io.imread(file_path)
    image = color.rgb2gray(image)
    image = skimage.transform.resize(image, (224, 224), anti_aliasing=True)
    fd, hog_image = hog(
        image,
        orientations=9,
        pixels_per_cell=(PPC, PPC),
        cells_per_block=(4, 4),
        block_norm='L2',
        visualize=True
    )
    labels.append([0,1])
    img_labels.append(file_path)
    hog_images.append(hog_image)
    hog_features.append(fd)
    del image
    del hog_image
    del fd

df = np.hstack((hog_features, labels))
print(df.shape)
img_names_df = pd.DataFrame(img_labels)
img_names_df.to_csv('image_paths.csv', encoding='utf-8')
hog_df = pd.DataFrame(df)
hog_df.to_pickle('hog_features')
quit()

#TODO run from here

data = pd.read_pickle('hog_features')
# np.random.shuffle(df)

train_test_split = 0.7
partition = int(len(hog_features) * train_test_split)
print(data.shape)
X_train, X_test = data.iloc[:partition, :-1], data.iloc[partition:, :-1]
y_train, y_test = data.iloc[:partition, -1:], data.iloc[partition:, -1:]

y_train = np.array(y_train).ravel()
y_test  = np.array(y_test).ravel()


pca = PCA(n_components=3, svd_solver='randomized', whiten=True).fit(X_train)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
param_grid = {'C': [3e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6],
              'gamma': [0.0001, 0.0003, 0.0005, 0.001, 0.005, 0.01, 0.1],
              'kernel': ['linear', 'rbf']}


clf = GridSearchCV(
    SVC(class_weight='balanced'), param_grid, verbose=2
)

clf = clf.fit(X_train_pca, y_train)

print('Best-Params: \n', clf.best_params_)

y_pred = clf.predict(X_test_pca)
print("Accuracy: "+str(accuracy_score(y_test, y_pred)))
print('\n')
print(classification_report(y_test, y_pred))

filename = 'svcpca_hog.sav'
# pickle.dump(clf, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test_pca, y_test)


INDEX = 2
image = skimage.io.imread(img_labels[INDEX])
res = loaded_model.predict([X_train_pca[INDEX]])
plt.imshow(image)
plt.title(str(res))
plt.show()
