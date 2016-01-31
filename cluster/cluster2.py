# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 11:47:23 2016

@author: Oliver
"""

from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline

from sklearn.cluster import FeatureAgglomeration, KMeans, DBSCAN

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def plot(x2d, flabels, title='' ):
	# Black removed and is used for noise instead.
	unique_labels = set(flabels)
	colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
	for k, col in zip(unique_labels, colors):
	    if k == -1:
	        # Black used for noise.
	        col = 'k'
	
	    class_member_mask = (flabels == k)
	
	    xy = x2d[class_member_mask]
	    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
	             markeredgecolor='k', markersize=5)
	
	plt.title(title +' (clusters: %d)' % len(unique_labels))
	plt.show()

def makePlots(Z):
	imp = Imputer()
	scal = StandardScaler()
	vart = VarianceThreshold()
	
	pipe = Pipeline([("imputer", imp), ("var theshold", vart), ("scaler", scal) ])
	
	# Require Z
	X1 = pipe.fit_transform(Z)
	pca = PCA(n_components=2)
	x2d = pca.fit_transform(X1.T)
	
	labels = {}
	
	for n in [2, 3, 5, 10]:
		agglo = FeatureAgglomeration(n_clusters=n).fit(X1)
		labels['ag%d'%n] = agglo.labels_
		plot(x2d, agglo.labels_, "Feature Agglomeration")
		
		km = KMeans(n_clusters=n).fit(X1.T)
		labels['km%d'%n] = km.labels_
		plot(x2d, km.labels_, "K-Means")
	
	dbs = DBSCAN(eps = 100 ,min_samples=10).fit(X1.T)
	labels['DBSCAN'] = dbs.labels_
	plot(x2d, dbs.labels_, "DBSCAN")
		
	return labels
	

