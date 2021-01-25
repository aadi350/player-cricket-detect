import pickle

from sklearn.model_selection import GridSearchCV
import numpy
from sklearn.svm import SVC

PARAM_GRID = {'C': [3e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6],
              'gamma': [0.0001, 0.0003, 0.0005, 0.001, 0.005, 0.01, 0.1],
              'kernel': ['linear', 'rbf']}


def fit(data_in: numpy.array, labels: numpy.array, param_grid: dict = PARAM_GRID):
	clf: GridSearchCV = GridSearchCV(
		SVC(class_weight='balanced'),
		param_grid,
		verbose=2)
	clf.fit(data_in, labels)
	with open('gridsearch_bestparams.pkl', 'wb') as params:
		pickle.dump(clf.best_params_, params)
	with open('svc_balanced.sav', 'wb') as saved_model:
		pickle.dump(clf, saved_model)


if __name__ == '__main__':
	raise NotImplementedError
