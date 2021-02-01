import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import learning_curve, cross_val_score

from datagen import dataset_generator
from processingutils.wavelettransform import vis_array, transform_array_hist

# Gets or creates a logger
logger = logging.getLogger(__name__)

# set log level
logger.setLevel(logging.INFO)

# define file handler and set formatter
file_handler = logging.FileHandler('logfile.log')
formatter = logging.Formatter('%(asctime)s : %(message)s')
file_handler.setFormatter(formatter)

# add file handler to logger
logger.addHandler(file_handler)

batsman, batsman_fname, batsman_labels = [], [], []
others, others_fname, others_labels = [], [], []

CATEGORY_DICT = {0: 'batsman', 1: 'others'}
NUM_FILES = 10

data = dataset_generator.get_raw_img()
from sklearn.utils import shuffle

data = shuffle(data)


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

from processingutils.wavelettransform import hist_single, hist

single_frame = images[0]
hist_single(single_frame, show=False)
count = hist(images, show=False)


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)

    train_scores_var = np.var(train_scores, axis=1)
    test_scores_var = np.var(test_scores, axis=1)

    train_scores_std_avg = np.sqrt(train_scores_var.mean())
    test_scores_std_avg = np.sqrt(test_scores_var.mean())
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)
    logger.info(

        f'{title} '
        f'train_size: {train_sizes}, train_scores_mean: {train_scores_mean.mean()}, test_scores_mean: {test_scores_mean.mean()}, '
        f'train_scores_std: {train_scores_std_avg}, test_scores_var: {test_scores_std_avg} '
    )
    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")
    plt.savefig(title + '.jpg')
    return plt


def split_data(data: pandas.DataFrame, ratio=0.7):
    partition = int(len(data) * ratio)
    X_train, X_test = data['images'].iloc[:partition], data['images'].iloc[partition:]
    y_train, y_test = data.iloc[:partition, 1], data.iloc[partition:, 1]

    y_train = np.array(y_train).ravel()
    y_test = np.array(y_test).ravel()

    return X_train, X_test, y_train, y_test


# ITERATION BEGINS
X_train, X_test, y_train, y_test = split_data(data)

from processingutils.wavelettransform import waves

print(waves)
i = 0
for wave in waves:

    i+=1
    # SPECIFY TYPE OF TRANSFORM
    X_train, X_test, y_train, y_test = split_data(data)
    X_train = transform_array_hist(X_train, wavelet=wave)
    X_test = transform_array_hist(X_test, wavelet=wave)

    clf = svm.SVC(random_state=42, kernel='rbf', max_iter=1000, tol=1e-3, probability=False)
    logging.info(str('SVC: ' + wave) + str(cross_val_score(clf, X_train, y_train).mean())
                 )
    plot_learning_curve(clf, str('SVC: ' + wave), X_train, y_train)

    if i > 2: break


quit()
# clf.fit(np.array(X_train, dtype=object), y_train)

y_pred = clf.predict(X_test)
print(y_pred)
print(classification_report(y_test, y_pred, zero_division=False))
# print(precision_recall_curve(y_test, y_pred))
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
