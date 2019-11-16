import pytest
from pytest import approx

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

clf = LogisticRegression(print_cost = False)
clf.fit(X_train, y_train)




def test_binary_score():
    print()
    sk_train = skclf.score(X_train, y_train)
    sk_test = skclf.score(X_test, y_test)
    print('sk training:', sk_train)
    print('sk testing:', sk_test)

    clf_train = clf.score(X_train, y_train)
    clf_test = clf.score(X_test, y_test)
    print('my training:', clf_train)
    print('my testing:', clf_test)



    assert sk_train == approx(clf_train)
    assert sk_test == approx(clf_test)
    
    
# def test_multi_class_score():
#     X, y = datasets.load_iris(return_X_y=True)
#     X_train, X_test, y_train, y_test = train_test_split(X,
#                                                         y, 
#                                                         test_size = 0.20, 
#                                                         random_state = 0,
#                                                         stratify = y)
#     skclf = linear_model.LogisticRegression(random_state=0, solver='lbfgs',
#                           multi_class='ovr').fit(X_train, y_train)

#     clf = LogisticRegression(print_cost = False)
#     clf.fit(X_train, y_train)
    

#     print()
#     sk_train = skclf.score(X_train, y_train)
#     sk_test = skclf.score(X_test, y_test)
#     print('sk training:', sk_train)
#     print('sk testing:', sk_test)




#     clf_train = clf.score(X_train, y_train)
#     clf_test = clf.score(X_test, y_test)
#     print('my training:', clf_train)
#     print('my testing:', clf_test)











