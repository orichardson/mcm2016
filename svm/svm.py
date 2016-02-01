import sys
sys.path.insert(0, "../modules")


from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVR
from sklearn import cross_validation
import numpy as np

def train_SVM(X, Y, kernel='rbf', shrinking=True,  tol=0.001, cache_size=1500, verbose=True, max_iter=-1):
	"""Assumes all irrelevant features have been removed from X and Y"""
	"""Learns several hundred SVMs"""

	clf = SVC(kernel=kernel, tol=tol, cache_size=cache_size, verbose=verbose, max_iter=max_iter)
	pipeline = Pipeline(zip([ "imputate", "vart", "scale", "svm" ], [ Imputer(), VarianceThreshold(), StandardScaler(), clf ]))
	
	param_grid = dict(svm__C=[0.1, 1, 10, 100, 1000],
										svm__gamma=[0.001, 0.01, 1, 10])

	
	grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=3)
	
	results = []

	for i in range(Y[0].shape[1]):
		Y_new = np.fromiter((x[:, i][0, 0] for x in Y), np.double)
		X_new = np.array([np.matrix(x.data).flatten().tolist() for x in X], np.double)
		#X_new = np.fromiter((np.matrix(x.data) for x in X), np.double)

		X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X_new, Y_new, test_size = 0.2)
		X_train = flatten(X_train)
		X_test = flatten(X_test)

		grid_search.fit(X_train, Y_train)
		results.append( (grid_search.best_estimator_, clf.score(X_test, Y_test)))	
		print("Best estimators (C): {0}, Score: {1}".format(grid_search.best_estimator_, clf.score(X_test, Y_test)))
	return results
	
def flatten(X):
	return np.array([x.flatten().tolist() for x in X], np.double)
