import sys
sys.path.insert(0, "../modules")


from sklearn.svm import SVC
import numpy as np

def train_SVM(X, Y, X_val, Y_val, C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None, random_state=None):
	"""Assumes all irrelevant features have been removed from X and Y"""

	params = [None] * len(Y[0])
	clf = SVC(C, kernel, degree, gamma, coef0, shrinking, probability, tol, cache_size, class_weight, verbose, max_iter, decision_function_shape, random_state)

	for i in range(len(Y[0])):
		Y_new = np.fromiter((x[i] for x in Y), np.double)
		X_new = np.fromiter((x.flatten() for x in X), np.double)
		clf.fit(X_new, Y_new)

		Y_val_new = np.fromiter((x[i] for x in Y_val), np.double)
		X_val_new = np.fromiter((x.flatten() for x in X_val), np.double)
		
		print("y-value: y_{0}, score: {1}".format(i, clf.score(X_val_new, Y_val_new)))
		params[i] = clf.get_params()
	
	return params
	
