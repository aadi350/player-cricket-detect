import pickle
import matplotlib.pyplot as plt
import skimage
from skimage import color
from skimage.feature import hog
from sklearn.decomposition import IncrementalPCA
import numpy
from skimage.metrics import peak_signal_noise_ratio
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

_PCA_PATH = '/home/aadidev/projects/player-cricket-detect/pca.pkl'
_CLF_PATH = '/home/aadidev/projects/player-cricket-detect/svc_balanced.sav'
_PPC = 4

CATEGORY_DICT = {0: 'batsman', 1: 'others'}


def _load_pca(ipca: IncrementalPCA = None) -> IncrementalPCA:
    if ipca is None:
        ipca = pickle.load(open(_PCA_PATH, 'rb'))
    return ipca


def _load_clf(clf: str = _CLF_PATH) -> object:
    return pickle.load(open(clf, 'rb'))


def eval_clf(arr, labels, clf: object = _CLF_PATH, ipca: IncrementalPCA = None) -> None:
    clf = _load_clf()
    ipca = _load_pca(ipca)
    res = clf.predict(arr)
    return classification_report(labels, res)


def classify_batch(img_arr, clf: SVC = None, ipca: IncrementalPCA = None, visualise: bool=False
                   ) -> None:
    assert len(img_arr) <= 12 #TODO update for multiple
    clf = _load_clf(clf)
    ipca = _load_pca(ipca)
    for i, img in enumerate(img_arr):
        image_original = skimage.io.imread(img)
        fd = _get_hog(image_original)
        hog_img_pca = ipca.transform([fd])
        res = clf.predict(hog_img_pca)[0]
        plt.subplot(4, 3, i+1)
        plt.imshow(image_original)
        plt.title(CATEGORY_DICT[res])
    plt.savefig('batchclassify_20210126.jpg')
    plt.show(block=(not visualise))


def _get_hog(img) -> numpy.array:
    image = color.rgb2gray(img)
    image = skimage.transform.resize(image, (224, 224), anti_aliasing=True)
    fd, hog_image = hog(
        image,
        orientations=9,
        pixels_per_cell=(_PPC, _PPC),
        cells_per_block=(4, 4),
        block_norm='L2',
        visualize=True
    )
    fd = numpy.append(fd, [1])
    return fd


def classify(img, clf: SVC = None, ipca: IncrementalPCA = None):
    clf = _load_clf(clf)
    ipca = _load_pca(ipca)
    image_original = skimage.io.imread(img)
    fd = _get_hog(image_original)
    hog_img_pca = ipca.transform([fd])
    res = clf.predict(hog_img_pca)[0]

    plt.imshow(image_original)
    plt.title(CATEGORY_DICT[res])
    plt.show()
