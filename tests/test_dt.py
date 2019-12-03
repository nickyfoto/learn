"""
pytest -svv test_dt.py
pytest -svv -m 'not smoke' test_dt.py
pytest -vv --html=report.html --capture=sys

"""

import pickle
from pprint import pprint
from copy import deepcopy

import pytest
import numpy as np
import pandas as pd

from learn.dt import DecisionTree, DecisionTreeD
# from learn.dt import _find_best_feature

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
    # print(clf.tree)


    clf_d = DecisionTreeD()
    clf_d.fit(X, Y)
    print(clf_d)
    print(clf_d.predict(X))


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

    # print(clf.tree)
    clf_d = DecisionTreeD()
    clf_d.fit(X_train, y_train)
    print(clf_d)
    # pprint(clf_d.tree)
    my_acc_d = accuracy_score(y_true=y_train, y_pred=clf_d.predict(X_train))
    my_test_acc_d = accuracy_score(y_true=y_test, y_pred=clf_d.predict(X_test))
    assert sk_acc == my_acc_d
    print('my_test_acc_d', my_test_acc_d)


@pytest.mark.smoke
def test_hw4_dt():
    data_test = pd.read_csv("../datasets/hw4_data_test.csv")
    data_valid = pd.read_csv("../datasets/hw4_data_valid.csv")
    data_train = pd.read_csv("../datasets/hw4_data_train.csv")

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
    print(sk_clf)

    sk_acc = accuracy_score(y_true=y_train, y_pred=sk_clf.predict(X_train))
    sk_test_acc = accuracy_score(y_true=y_test, y_pred=sk_clf.predict(X_test))
    print(sk_acc, sk_test_acc)

    # clf8 = DecisionTreeD(max_depth=8)
    # clf8.fit(X_train, y_train)
    
    # print(clf8)

    # my_acc = accuracy_score(y_true=y_train, y_pred=clf8.bunch_predict(X_train))
    # my_test_acc = accuracy_score(y_true=y_test, y_pred=clf8.bunch_predict(X_test))
    # print('acc when depth=8', my_acc, my_test_acc)

    clf_d = DecisionTreeD()
    clf_d.fit(X_train, y_train)
    my_acc_d = accuracy_score(y_true=y_train, y_pred=clf_d.bunch_predict(X_train))
    my_test_acc_d = accuracy_score(y_true=y_test, y_pred=clf_d.bunch_predict(X_test))
    print()
    print(my_acc_d, my_test_acc_d)

    pickle.dump( clf_d, open( "../models/dt_clf.p", "wb" ) )

@pytest.mark.smoke
def test_pruning():

    data_test = pd.read_csv("../datasets/hw4_data_test.csv")
    data_valid = pd.read_csv("../datasets/hw4_data_valid.csv")
    data_train = pd.read_csv("../datasets/hw4_data_train.csv")

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


    clf = pickle.load( open( "../models/dt_clf.p", "rb" ) )
    my_acc = accuracy_score(y_true=y_train, y_pred=clf.bunch_predict(X_train))
    my_test_acc = accuracy_score(y_true=y_test, y_pred=clf.bunch_predict(X_test))
    print()
    print(my_acc, my_test_acc)

    def DecisionTreeEvalution(dt,X,y, verbose=True):
        #print(X.shape, y.shape)
        # Make predictions
        # For each test sample X, use our fitting dt classifer to predict
        y_predicted = []
        for record in X: 
            y_predicted.append(dt.predict(record))

        # Comparing predicted and true labels
        results = [prediction == truth for prediction, truth in zip(y_predicted, y)]

        # Accuracy
        accuracy = float(results.count(True)) / float(len(results))
        if verbose:
            print("accuracy: %.4f" % accuracy)
        return accuracy

    def pruning(dt, X, y):
        # print(dt.leaves)
        pt = deepcopy(dt)
        # list we iterate to check whether to change a tree node into a leaf 
        # node
        trees = []
        for leaf in pt.leaves:
            trees.append(leaf['parent'])
        while trees:
            tree = trees.pop(0)
            if not tree['is_leaf']:
                curr_error_rate = DecisionTreeEvalution(pt,X,y, verbose=False)
                tree['is_leaf'] = True
                updated_error_rate = DecisionTreeEvalution(pt,X,y, 
                                                            verbose=False)
                print(curr_error_rate, updated_error_rate, len(trees))
                if updated_error_rate < curr_error_rate:
                    tree['is_leaf'] = False
                if tree.get('parent'):
                    if not tree['parent']['is_leaf']:
                        trees.append(tree['parent'])
        return pt

            
    dt_pruned = pruning(clf, X_test, y_test)
    print(DecisionTreeEvalution(dt_pruned, X_valid, y_valid, False))




# @pytest.mark.smoke
def test_pruning():

    data_test = pd.read_csv("../datasets/hw4_data_test.csv")
    data_valid = pd.read_csv("../datasets/hw4_data_valid.csv")
    data_train = pd.read_csv("../datasets/hw4_data_train.csv")

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


    clf = pickle.load( open( "../models/dt_clf.p", "rb" ) )
    my_acc = accuracy_score(y_true=y_train, y_pred=clf.bunch_predict(X_train))
    my_test_acc = accuracy_score(y_true=y_test, y_pred=clf.bunch_predict(X_test))
    print()
    print(my_acc, my_test_acc)

    def DecisionTreeEvalution(dt,X,y, verbose=True):
        #print(X.shape, y.shape)
        # Make predictions
        # For each test sample X, use our fitting dt classifer to predict
        y_predicted = []
        for record in X: 
            y_predicted.append(dt.predict(record))

        # Comparing predicted and true labels
        results = [prediction == truth for prediction, truth in zip(y_predicted, y)]
        # print(y, results)
        # Accuracy
        accuracy = float(results.count(True)) / float(len(results))
        if verbose:
            print("accuracy: %.4f" % accuracy)
        return accuracy


    def _partition_classes(X, y, split_attribute, split_val):

        left_indices = X[:,split_attribute] <= split_val
        X_left = X[left_indices]
        y_left = y[left_indices]
        X_right = X[~left_indices]
        y_right = y[~left_indices]
        return X_left, X_right, y_left, y_right

    def DecisionTreeError(y):
        num_ones = np.sum(y)
        num_zeros = len(y) - num_ones
        return 1.0 - max(num_ones, num_zeros) / float(len(y))

    def pruning(dt, X, y):
        if dt.is_leaf or X.shape[0] == 0:
            return dt

        X_left, X_right, y_left, y_right = _partition_classes(
            X, y, dt.feature_to_split, dt.split_val)

        dt.left = pruning(dt.left, X_left, y_left)
        dt.right = pruning(dt.right, X_right, y_right)
        error = 1 - DecisionTreeEvalution(dt, X, y, verbose=False)
        leaf_error = DecisionTreeError(y)
        if error >= leaf_error:
            dt.is_leaf = True
        return dt
        

    
    dt_pruned = pruning(clf, X_test, y_test)
    print(DecisionTreeEvalution(dt_pruned, X_valid, y_valid, False))
