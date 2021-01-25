from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt
import numpy as np
import logging
import pickle

logger = logging.getLogger('root')
FORMAT = "[%(filename)s - %(funcName)10s() ] %(message)s"
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.DEBUG)

N_COMPONENTS = 150
BATCH_SIZE = 500


def fit_pca(data: np.array) -> IncrementalPCA:
    """
    :param data: 2-dimensional array of input features
    :return: PCA object fitted on input data
    """
    ipca = IncrementalPCA()
    ipca_fitted = ipca.fit(data)
    logger.debug('PCA Fit Complete')
    pickle.dump(ipca_fitted, open("pca.pkl", "wb"))
    return ipca_fitted


def transform_pca(data: np.array, ipca: IncrementalPCA):
    """
    :param data: 2-dimensional array of input features
    :param ipca: trained PCA object
    :return: transformed data
    """
    logger.debug('')
    return ipca.transform(data)


def load_pca(pca=None):
    """
    :param pca: [OPTIONAL] PCA object file
    :return: loaded PCA object
    """
    if pca is None:
        PCA_PATH = '/home/aadidev/projects/player-cricket-detect/pca.pkl'
        ipca = pickle.load(open(PCA_PATH, 'rb'))
    return ipca


def plot_explained_variance(ipca, block=False) -> None:
    """
    :param ipca: trained IPCA object
    :param block: boolean specifying whether to pause program execution for matplotlib
    :return: None
    """
    cum_var_explained = np.cumsum(ipca.explained_variance_ratio_)
    plt.plot(np.cumsum(cum_var_explained))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.savefig('Explained Variance.jpg')
    plt.show(block=block)


def plot_covariance(data: np.array) -> np.array:
    """
    :param data: 2-dimensional array
    :return: covariance map
    """
    import seaborn as sns
    covariance_map = np.cov(data.T)
    plt.figure(figsize=(15, 15))
    sns.set(font_scale=2)

    _ = sns.heatmap(covariance_map,
                     cbar=True,
                     annot=True,
                     square=True,
                     fmt='.2f',
                     annot_kws={'size': 12}, )
    plt.title('Covariance matrix showing correlation coefficients')
    plt.tight_layout()
    plt.show()
    return covariance_map


if __name__ == '__main__':
    raise NotImplementedError
