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
                                        random_state=0)
    

    print()
    gmm = GaussianMixture(n_components=3,
                            # covariance_type='full', 
                            # max_iter=100, 
                            random_state=0)
    print(gmm)
    gmm.fit(X_train)
    # sk_gmm.means_init = np.array([X_train[y_train == i].mean(axis=0)
                                        # for i in range(n_classes)])
    


    print('my train acc:', accuracy_score(y_true=y_train, y_pred=gmm.predict(X_train)))
    print('my test acc:', accuracy_score(y_true=y_test, y_pred=gmm.predict(X_test)))

    sk_gmm.fit(X_train)
    
    print()
    print(sk_gmm)
    y_train_pred = sk_gmm.predict(X_train)
    # print(sk_gmm.score(X_train))
    print('sk train acc:', accuracy_score(y_true=y_train, y_pred=y_train_pred))
    print('sk test acc:', accuracy_score(y_true=y_test, y_pred=sk_gmm.predict(X_test)))
    print('sk mean', sk_gmm.means_)
    print('my mean', gmm.means_)


