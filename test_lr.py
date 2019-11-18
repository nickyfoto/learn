"""
following command to test
pytest -svv -m 'smoke' test_lr.py --disable-warnings
pytest -svv test_lr.py

pytest -svv test_lr.py --disable-warnings



pytest -svv -m 'sgd' test_lr.py --disable-warnings
pytest -svv -m 'l2' test_lr.py --disable-warnings
pytest -svv -m 'loss' test_lr.py --disable-warnings
"""


import pytest
from pytest import approx

import numpy as np
from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score

from lr import LogisticRegression
from evaluation import within

from pprint import pprint




@pytest.mark.loss
def test_loss_func():
    
    X, y = datasets.load_iris(return_X_y=True)
    X = X[:100]
    y = y[:100]
    max_iter = 100
    clf = LogisticRegression(print_cost = False, max_iter=max_iter, learning_rate=1e-1, sgd=True)
    clf.fit(X, y)
    # print(clf.costs)
    assert clf.costs.shape == (max_iter,)

    X, y = datasets.load_iris(return_X_y=True)
    clf.fit(X, y)
    print(clf.costs)
    print(clf.costs.T.shape, clf.costs.T.mean(axis=1).shape)
    assert clf.costs.shape == (3,max_iter)



# @pytest.mark.l2
def test_l2_regularization():
    X, y = datasets.load_iris(return_X_y=True)

    # print(clf.Cs_)
    sksgd = linear_model.SGDClassifier(loss='log', penalty='none', tol=None,
                                        shuffle=False, 
                                        verbose=0
                                        )
    print()
    print(sksgd)
    sksgd.fit(X, y)
    t_score = sksgd.score(X, y)
    print('training:', t_score)

    sksgd = linear_model.SGDClassifier(loss='log', tol=None,
                                        shuffle=False, 
                                        verbose=0
                                        )
    params = {'alpha': np.logspace(-4, 4, 10)}
    clf = GridSearchCV(sksgd, param_grid = params, cv=5)
    clf.fit(X, y)
    pprint(clf.cv_results_['params'])
    pprint(clf.cv_results_['mean_test_score'])
    print('best param', clf.best_params_)
    print('best cv score', clf.best_score_)
    # print(np.logspace(-4, 4, 10))
    params = {'c_lambda': np.logspace(-4, 4, 10)}
    sgd_r = LogisticRegression(penalty='l2', sgd=True, learning_rate=1e-2,
                               max_iter=1000)
    print(sgd_r)
    clf = GridSearchCV(sgd_r, param_grid = params, cv=5)
    clf.fit(X, y)
    pprint(clf.cv_results_['params'])
    pprint(clf.cv_results_['mean_test_score'])
    print('best param', clf.best_params_)
    print('best cv score', clf.best_score_)














@pytest.mark.sgd
def test_basic_sgd():
    X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    y = np.array([0, 0, 1, 1])
    y = np.array([2, 2, 5, 5])
    clf = LogisticRegression(sgd = True, max_iter=100)
    clf.fit(X, y)
    print()
    preds = clf.predict(X)
    # print(preds)
    # print(clf.predict_proba(X))
    assert np.array_equal(preds, y)



@pytest.mark.sgd
def test_basic_multiclass_sgd():
    X = np.array([[-1, -1], 
                  [-2, -1], 
                  [1, 1], 
                  [2, 1],
                  [5, 6],
                  [7, 8]])
    y = np.array([1, 1, 2, 2, 3, 3])
    sksgd = linear_model.SGDClassifier(loss='log', penalty='none', tol=None,
                                        shuffle=False, 
                                        verbose=0
                                        )
    print()
    print(sksgd)
    sksgd.fit(X, y)
    preds = sksgd.predict(X)
    assert np.array_equal(preds, y)

    sgd = LogisticRegression(sgd=True, max_iter=100)
    print(sgd)
    sgd.fit(X, y)
    preds = sksgd.predict(X)
    # print(preds)
    assert np.array_equal(preds, y)



@pytest.mark.sgd
def test_multiclass_sgd():
    X, y = datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y, 
                                                        test_size = 0.20, 
                                                        random_state = 0,
                                                        stratify = y)
    sksgd = linear_model.SGDClassifier(loss='log', penalty='none', tol=None,
                                        shuffle=False, 
                                        verbose=0
                                        )
    print()
    print(sksgd)
    sksgd.fit(X, y)
    
    sksgd_train = sksgd.score(X_train, y_train)
    sksgd_test = sksgd.score(X_test, y_test)
    print('sk training:', sksgd_train)
    print('sk testing:', sksgd_test)

    sgd = LogisticRegression(sgd=True, max_iter=2000, learning_rate=1e-2)
    print(sgd)
    sgd.fit(X, y)
    sgd_train = sgd.score(X_train, y_train)
    sgd_test = sgd.score(X_test, y_test)
    print('my training:', sgd_train)
    print('my testing:', sgd_test)


    assert within(sk_val = sksgd_train, val=sgd_train, tol = 0.1)
    assert within(sk_val = sksgd_test, val=sgd_test, tol = 0.1)

@pytest.mark.smoke
def test_basic_multiclass():
    X = np.array([[-1, -1], 
                  [-2, -1], 
                  [1, 1], 
                  [2, 1],
                  [5, 6],
                  [7, 8]])
    y = np.array([1, 1, 2, 2, 3, 3])
    clf = LogisticRegression()
    clf.fit(X, y)
    print()
    preds = clf.predict(X)
    # print(preds)
    clf.predict_proba(X)
    assert np.array_equal(preds, y)






@pytest.mark.smoke
def test_basic():
    X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    y = np.array([0, 0, 1, 1])
    y = np.array([2, 2, 5, 5])
    clf = LogisticRegression()
    clf.fit(X, y)
    print()
    preds = clf.predict(X)
    # print(preds)
    # print(clf.predict_proba(X))
    assert np.array_equal(preds, y)



@pytest.mark.smoke
def test_binary_score():
    X, y = datasets.load_iris(return_X_y=True)
    X = X[:100]
    y = y[:100]

    y += 1

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y, 
                                                        test_size = 0.20, 
                                                        random_state = 0,
                                                        stratify = y)
    skclf = linear_model.LogisticRegression(random_state=0, solver='lbfgs',
                          multi_class='ovr').fit(X_train, y_train)
    clf = LogisticRegression(print_cost = False)
    clf.fit(X_train, y_train)
    print()
    sk_train = skclf.score(X_train, y_train)
    sk_test = skclf.score(X_test, y_test)
    print('sk training:', sk_train)
    print('sk testing:', sk_test)


    # print(skclf.coef_, skclf.intercept_)
    # print(clf.coef_, clf.intercept_)

    clf_train = clf.score(X_train, y_train)
    clf_test = clf.score(X_test, y_test)
    print('my training:', clf_train)
    print('my testing:', clf_test)



    assert sk_train == approx(clf_train)
    assert sk_test == approx(clf_test)
    



@pytest.mark.smoke
def test_multi_class_score():
    X, y = datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y, 
                                                        test_size = 0.20, 
                                                        random_state = 0,
                                                        stratify = y)
    skclf = linear_model.LogisticRegression(random_state=0, solver='lbfgs',
                          multi_class='ovr').fit(X_train, y_train)

    clf = LogisticRegression(print_cost = False, learning_rate=1e-2, max_iter=3e3)
    clf.fit(X_train, y_train)
    

    print()
    sk_train = skclf.score(X_train, y_train)
    sk_test = skclf.score(X_test, y_test)
    print('sk training:', sk_train)
    print('sk testing:', sk_test)

    print('Confusion Matrix: \n',confusion_matrix(y_train,skclf.predict(X_train)))
    print('Confusion Matrix: \n',confusion_matrix(y_test,skclf.predict(X_test)))

    clf_train = clf.score(X_train, y_train)
    clf_test = clf.score(X_test, y_test)
    print('my training:', clf_train)
    print('my testing:', clf_test)

    # print(skclf.coef_, skclf.intercept_)
    # print(clf.coef_, clf.intercept_)

    print('Confusion Matrix: \n',confusion_matrix(y_train,clf.predict(X_train)))
    print('Confusion Matrix: \n',confusion_matrix(y_test,clf.predict(X_test)))


    assert within(sk_val = sk_train, val=clf_train, tol = 0.1)
    assert within(sk_val = sk_test, val=clf_test, tol = 0.1)









