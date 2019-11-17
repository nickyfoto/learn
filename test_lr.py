import pytest
from pytest import approx

import numpy as np
from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split


from lr import LogisticRegression

# test two class
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

# clf = LogisticRegression_v2(print_cost = False)
# clf.fit(X_train, y_train)



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
    print(preds)
    assert np.array_equal(preds, y)





# pytest -svv -m 'smoke' test_lr.py
# @pytest.mark.smoke
def test_basic():
    X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    y = np.array([0, 0, 1, 1])
    y = np.array([2, 2, 5, 5])
    clf = LogisticRegression()
    clf.fit(X, y)
    print()
    preds = clf.predict(X)
    print(preds)
    assert np.array_equal(preds, y)



# @pytest.mark.smoke
def test_binary_score():
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
    

def within(sk_val, val, tol):
    return sk_val - tol < val < sk_val + tol

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



    clf_train = clf.score(X_train, y_train)
    clf_test = clf.score(X_test, y_test)
    print('my training:', clf_train)
    print('my testing:', clf_test)

    print(skclf.coef_, skclf.intercept_)
    print(clf.coef_, clf.intercept_)



    assert within(sk_val = sk_train, val=clf_train, tol = 0.1)
    assert within(sk_val = sk_test, val=clf_test, tol = 0.1)









