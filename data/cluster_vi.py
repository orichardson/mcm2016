# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 16:50:24 2016

@author: Oliver
"""

import numpy as np

from sklearn.cluster import DBSCAN, KMeans
from sklearn import metrics
from sklearn.preprocessing import Imputer


def cluster(Z, algo='kmeans'):
    descr = Z.columns
    X = Imputer().fit_transform(Z).T

    ##############################################################################
    if algo == 'dbscan':
        # Compute DBSCAN
        db = DBSCAN(eps=0.3, min_samples=10).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        
        print('Estimated number of clusters: %d' % n_clusters_)
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
        print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
        print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
        print("Adjusted Rand Index: %0.3f"
              % metrics.adjusted_rand_score(labels_true, labels))
        print("Adjusted Mutual Information: %0.3f"
              % metrics.adjusted_mutual_info_score(labels_true, labels))
        print("Silhouette Coefficient: %0.3f"
              % metrics.silhouette_score(X, labels))
        
    elif algo == 'kmeans':
        km = KMeans(n_clusters=3)
        km.fit(X)
        return km