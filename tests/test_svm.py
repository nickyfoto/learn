"""
pytest -svv test_svm.py
"""

import pytest
import numpy as np


from learn import SVC
from sklearn import svm
from sklearn.datasets import make_classification


def get_dataset_for(X1, y1, X2, y2):
    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))
    return X, y

def get_training_examples():
    X1 = np.array([[8, 7], [4, 10], [9, 7], [7, 10],
                   [9, 6], [4, 8], [10, 10]])
    y1 = np.ones(len(X1))
    X2 = np.array([[2, 7], [8, 3], [7, 5], [4, 4],
                   [4, 6], [1, 3], [2, 5]])
    y2 = np.ones(len(X2)) * -1
    return X1, y1, X2, y2

def get_dataset(get_examples):
    X1, y1, X2, y2 = get_examples()
    X, y = get_dataset_for(X1, y1, X2, y2)
    return X, y


def test_basic_svm():


    X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    y = np.array([1, 1, 2, 2])

    # X, y = get_dataset(get_training_examples)
    sk_clf = svm.SVC(gamma='auto', random_state=0, verbose=False)
    sk_clf.fit(X, y) 
    print()
    # print(sk_clf)
    # print(clf.predict([[-0.8, -1]]))
    # print(np.array_equal(y, sk_clf.predict(X)))
    assert np.array_equal(y, sk_clf.predict(X))
    clf = SVC()
    clf.fit(X, y)
    # print(clf.predict(X))
    # print(np.array_equal(y, clf.predict(X)))
    assert np.array_equal(y, clf.predict(X))

    X, y = make_classification(n_features=2, random_state=1, n_samples=4, n_redundant=0)
    clf = svm.LinearSVC(random_state=0, tol=1e-5, loss='hinge')
    clf.fit(X, y)

    myclf = SVC()
    myclf.fit(X, y)
    print(myclf.coef_, myclf.intercept_)
    assert np.array_equal(myclf.predict(X), clf.predict(X))


    assert np.array_equal(clf.predict(X),  y)