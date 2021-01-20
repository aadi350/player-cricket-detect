import os
import pickle

from skimage.feature import hog
from sklearn.experimental import enable_halving_search_cv  # noqa
# now you can import normally from model_selection
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import skimage
from skimage.transform import resize
from skimage.io import imread
from skimage import color
import numpy as np


DATA_DIR = "/home/aadi/PycharmProjects/player-cricket-detect/data/img/sahil_categories"
DATA_CAT_BATSMAN = os.path.join(DATA_DIR, 'batsman')
DATA_CAT_OTHERS = os.path.join(DATA_DIR, 'others')

batsman, batsman_fname, batsman_labels = [], [], []
others, others_fname, others_labels = [], [], []

BATSMAN = 0
OTHERS = 1

PPC = 4
hog_images = []
hog_features = []
img_labels = []
labels = []

for file in os.listdir(DATA_CAT_BATSMAN)[:10]:
    file_path = os.path.join(DATA_CAT_BATSMAN, file)
    image = skimage.io.imread(file_path)
    image = color.rgb2gray(image)
    image = skimage.transform.resize(image, (224, 224), anti_aliasing=True)
    fd, hog_image = hog(
        image,
        orientations=8,
        pixels_per_cell=(PPC, PPC),
        cells_per_block=(4, 4),
        block_norm='L2',
        visualize=True
    )
    labels.append([1, 0])
    img_labels.append(file)
    hog_images.append(hog_image)
    hog_features.append(fd)

for file in os.listdir(DATA_CAT_OTHERS)[:10]:
    file_path = os.path.join(DATA_CAT_OTHERS, file)
    image = skimage.io.imread(file_path)
    image = color.rgb2gray(image)
    image = skimage.transform.resize(image, (224, 224), anti_aliasing=True)
    fd, hog_image = hog(
        image,
        orientations=8,
        pixels_per_cell=(PPC, PPC),
        cells_per_block=(4, 4),
        block_norm='L2',
        visualize=True
    )
    labels.append([0, 1])
    img_labels.append(file)
    hog_images.append(hog_image)
    hog_features.append(fd)

labels =  np.array(labels).reshape(len(labels), 2)

df = np.hstack((hog_features, labels))
np.random.shuffle(df)


train_test_split = 0.7
partition = int(len(hog_features) * train_test_split)

X_train, X_test = df[:partition, :-1], df[partition:, :-1]
y_train, y_test = df[:partition, -1:].ravel(), df[partition:, -1:].ravel()


pca = PCA(n_components=150, svd_solver='randomized',whiten=True).fit(X_train)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }

# TODO
# clf = HalvingGridSearchCV(
#     SVC(kernel='rbf', class_weight='balanced')
# )

clf = GridSearchCV(
    SVC(kernel='rbf', class_weight='balanced'), param_grid
)

# clf = clf.fit(X_train_pca, y_train)
print(clf.param_grid)
print(clf.best_params_)
MLPClassifier(random_state=1, learning_rate='invscaling', max_iter=500, solver='sgd', verbose=True).fit(X_train_pca, y_train)
y_pred = clf.predict(X_test_pca)
print("Accuracy: "+str(accuracy_score(y_test, y_pred)))
print('\n')
print(classification_report(y_test, y_pred))

filename = 'nnpca_hog.sav'
pickle.dump(clf, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test_pca, y_test)

print(loaded_model.predict([X_test[0]]))
