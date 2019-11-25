"""
pytest -svv test_kmeans.py
"""

import pytest

import numpy as np
from kmeans import KMeans

from sklearn import cluster



def test_basic():

    X = np.array([[1, 2], [1, 4], [1, 0],
                  [10, 2], [10, 4], [10, 0]])
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)



    sk_kmeans = cluster.KMeans(n_clusters=2, random_state=0, init='random').fit(X)


    # print(kmeans.labels_, sk_kmeans.labels_)
    assert np.array_equal(kmeans.labels_, sk_kmeans.labels_)
    
    preds = kmeans.predict(np.array([[0, 0], 
                                     [12, 3]]))
    
    sk_preds = sk_kmeans.predict(np.array([[0, 0], 
                                           [12, 3]]))
    # print(preds, sk_preds)
    assert np.array_equal(preds, sk_preds)
    # print()
    # print(kmeans.cluster_centers_, sk_kmeans.cluster_centers_)
    


    assert np.array_equal(kmeans.cluster_centers_, sk_kmeans.cluster_centers_)