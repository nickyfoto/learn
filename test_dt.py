"""
pytest -svv test_dt.py
pytest -svv -m 'not smoke' test_dt.py
pytest -vv --html=report.html --capture=sys

"""

import pytest
import numpy as np
import pandas as pd

from dt import DecisionTree
from dt import _find_best_feature

from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



@pytest.mark.smoke
def test_find_best_feature():
    test_X = [  
                [3, 1, 10],
                [1, 1, 22],
                [2, 1, 28],
                [5, 1, 32],
                [4, 1, 32]
             ]
    test_y = [1,1,0,0,1]


    best_feature, best_split_val = _find_best_feature(np.array(test_X), np.array(test_y))
    print("best_split_feature:", best_feature, "best_split_val:", best_split_val)



@pytest.mark.smoke
def test_basic_dt():
    X = np.array([[0, 0], 
                  [1, 1]])
    Y = np.array([0, 1])
    sk_clf = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
    sk_clf.fit(X, Y)
    print()
    # print(clf.predict([[2., 2.]]))
    print(sk_clf)
    print(sk_clf.predict(X))


    clf = DecisionTree(criterion='entropy')
    clf.fit(X, Y)
    print(clf)
    print(clf.predict(X))
    print(clf.tree)


@pytest.mark.smoke
def test_iris_dt():
    from sklearn.datasets import load_iris
    iris = load_iris()


    X_train, X_test, y_train, y_test = train_test_split(iris.data,
                                                        iris.target, 
                                                        test_size = 0.25, 
                                                        random_state = 0,
                                                        stratify = iris.target)


    sk_clf = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
    sk_clf.fit(X_train, y_train)
    print()
    # print(clf.predict([[2., 2.]]))
    print(sk_clf)

    sk_acc = accuracy_score(y_true=y_train, y_pred=sk_clf.predict(X_train))
    sk_test_acc = accuracy_score(y_true=y_test, y_pred=sk_clf.predict(X_test))

    clf = DecisionTree(criterion='entropy')
    clf.fit(X_train, y_train)
    print()
    # print(clf.predict([[2., 2.]]))
    print(clf)

    my_acc = accuracy_score(y_true=y_train, y_pred=clf.predict(X_train))
    my_test_acc = accuracy_score(y_true=y_test, y_pred=clf.predict(X_test))
    print('sk_test_acc', sk_test_acc)
    print('my_test_acc', my_test_acc)
    assert sk_acc == my_acc




# @pytest.mark.smoke
def test_hw4_dt():
    data_test = pd.read_csv("datasets/hw4_data_test.csv")
    data_valid = pd.read_csv("datasets/hw4_data_valid.csv")
    data_train = pd.read_csv("datasets/hw4_data_train.csv")

    categorical = ['workclass', 'education', 'marital-status', 'occupation', 
                       'relationship', 'race', 'sex', 'native-country']
    numerical = ['age', 'fnlwgt', 'education-num','capital-gain', 'capital-loss',
                    'hours-per-week']

    for feature in categorical:
            le = LabelEncoder()
            data_train[feature] = le.fit_transform(data_train[feature])
            data_test[feature] = le.fit_transform(data_test[feature])
            
    X_train = pd.concat([data_train[categorical], data_train[numerical]], axis=1)
    y_train = data_train['high-income']
    X_test = pd.concat([data_test[categorical], data_test[numerical]], axis=1)
    y_test = data_test['high-income']
    X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

    for feature in categorical:
            le = LabelEncoder()
            data_valid[feature] = le.fit_transform(data_valid[feature])  
            
    X_valid = pd.concat([data_valid[categorical], data_valid[numerical]], axis=1)
    y_valid = data_valid['high-income']
    X_valid, y_valid = np.array(X_valid), np.array(y_valid)



    sk_clf = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
    sk_clf.fit(X_train, y_train)
    print()
    # print(clf.predict([[2., 2.]]))
    print(sk_clf)

    sk_acc = accuracy_score(y_true=y_train, y_pred=sk_clf.predict(X_train))
    sk_test_acc = accuracy_score(y_true=y_test, y_pred=sk_clf.predict(X_test))
    print(sk_acc, sk_test_acc)

    clf = DecisionTree(criterion='entropy', max_depth=8)
    # clf = DecisionTree()
    clf.fit(X_train, y_train)
    
    print(clf)

    my_acc = accuracy_score(y_true=y_train, y_pred=clf.predict(X_train))
    my_test_acc = accuracy_score(y_true=y_test, y_pred=clf.predict(X_test))
    print(my_acc, my_test_acc)
    # assert sk_acc == my_acc

# test_iris_dt()





