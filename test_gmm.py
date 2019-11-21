"""
pytest -svv test_gmm.py
pytest -svv -m 'not smoke' test_lm.py
pytest -vv --html=report.html --capture=sys

"""

import pytest
from pytest import approx

import numpy as np

from gmm import GaussianMixture
from sklearn import mixture
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from evaluation import within


def test_basic_gmm():
    iris = datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data,
                                                        iris.target, 
                                                        test_size = 0.25, 
                                                        random_state = 0,
                                                        stratify = iris.target)

    n_classes = len(np.unique(y_train))

    sk_gmm = mixture.GaussianMixture(n_components=3,
                                        covariance_type='full', 
                                        max_iter=20, 
                                        random_state=0,
                                        verbose=2,
                                        tol=1e-6,
                                        verbose_interval=1)
    

    print()
    gmm = GaussianMixture(n_components=3,
                            # covariance_type='full', 
                            # max_iter=100, 
                            random_state=0,
                            abs_tol=1e-6,
                            rel_tol=1e-6,
                            verbose=1)
    print(gmm)
    gmm.fit(X_train)
    # sk_gmm.means_init = np.array([X_train[y_train == i].mean(axis=0)
                                        # for i in range(n_classes)])
    

    my_train_acc = accuracy_score(y_true=y_train, y_pred=gmm.predict(X_train))
    my_test_acc = accuracy_score(y_true=y_test, y_pred=gmm.predict(X_test))
    print('my train acc:', my_train_acc)
    print('my test acc:', my_test_acc)




    sk_gmm.fit(X_train)
    
    
    print(sk_gmm)
    
    
    sk_train_acc = accuracy_score(y_true=y_train, y_pred=sk_gmm.predict(X_train))
    sk_test_acc = accuracy_score(y_true=y_test, y_pred=sk_gmm.predict(X_test))
    print('sk train acc:', sk_train_acc)
    print('sk test acc:', sk_test_acc)

    assert my_train_acc == sk_train_acc
    assert my_test_acc == sk_test_acc

    assert np.allclose(sk_gmm.means_, gmm.means_, rtol=0, atol=1e-02)
    assert np.allclose(sk_gmm.weights_, gmm.weights_, rtol=0, atol=1e-02)
    assert np.allclose(sk_gmm.covariances_, gmm.covariances_, rtol=0, atol=1e-02)



