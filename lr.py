"""
Logistic Regression
"""
import numpy as np
import math, inspect
from collections import defaultdict
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator
from scipy.special import expit as sigmoid
from sklearn.preprocessing import LabelEncoder


class LogisticRegression(BaseEstimator):
    """
    Attributes
    ----------
    coef_ : array, shape (1, n_features) if n_classes == 2 else (n_classes,\
            n_features)
        Weights assigned to the features.
    intercept_ : array, shape (1,) if n_classes == 2 else (n_classes,)
        Constants in decision function.
    """
    def __init__(self, num_iterations = 2000, 
                       learning_rate = 0.5, 
                       fit_intercept = True,
                       print_cost = True,
                       steps = 10,
                       C = 0,
                       penalty=None):
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.fit_intercept = fit_intercept
        self.print_cost = print_cost
        self.costs = []
        self.steps = steps
        self.penalty = penalty
        self.C = 1 / C if C != 0 else 0

    def _fit_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def log_likelihood(self, preds, target):
        ll = - (np.dot(target,  np.log(preds + np.finfo(float).eps)) + 
                np.dot((1 - target), np.log(1 - preds + np.finfo(float).eps))) / self.m
        if self.penalty:
            ll += self.C * np.square(self.weights[1:]).sum() / (2 * self.m)
        return ll


    def fit(self, X, y):

        self.le = LabelEncoder()
        y = self.le.fit_transform(y)

        if self.fit_intercept:
            X = self._fit_intercept(X)
            
        self.m, self.n = X.shape
        self.weights = np.zeros(self.n)
        for step in range(self.num_iterations):
                    
            preds = sigmoid(np.dot(X, self.weights))
            error = preds - y
            gradient = np.dot(X.T, error) / self.m

            if self.penalty:
                self.weights[0] -= self.learning_rate * gradient[0]
                self.weights[1:] -= self.learning_rate * (gradient[1:] + self.C * self.weights[1:] / self.m)
            else:
                self.weights -= self.learning_rate * gradient

            if step % (self.num_iterations // self.steps) == 0:
                cost = self.log_likelihood(preds = preds, 
                                                target = y)
                if self.print_cost: print(step, cost)
                self.costs.append(cost)
        return self
    
    @property
    def intercept_(self):
        return self.weights[:1]
    
    @property
    def coef_(self):
        return self.weights[1:].reshape(1, self.n - 1)
    

    def predict_proba(self, X):
        if self.fit_intercept:
            X = self._fit_intercept(X)
    
        return sigmoid(np.dot(X, self.weights))
    
    def predict(self, X):
        return self.le.inverse_transform(self.predict_proba(X).round().astype(int))

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    

class LogisticRegressionSGD(BaseEstimator):
    """
    Logistic regression with stochastic gradient descent
    """

    def __init__(self, num_iterations = 100, 
                        learning_rate = .1, 
                        fit_intercept = True,
                        print_cost = True,
                        C = 0,
                        penalty=None):
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.fit_intercept = fit_intercept
        self.print_cost = print_cost
        self.C = 1 / C if C != 0 else 0
        self.penalty = penalty

    def _fit_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def log_likelihood(self, preds, target):
        ll = - (target * np.log(preds + np.finfo(float).eps) + 
                (1 - target) * np.log(1 - preds + np.finfo(float).eps)).sum() / self.m
        return ll

    def fit(self, features, target):

        if self.fit_intercept:
            features = self._fit_intercept(features)
        self.m, self.n = features.shape
        self.weights = np.zeros(self.n)

        for step in range(self.num_iterations):
            for i, x in enumerate(features):
                score = np.dot(x, self.weights)
                pred = sigmoid(score)
                error = pred - target[i]
                gradient = np.dot(error, x)
                if self.penalty:
                    self.weights[0] -= self.learning_rate * gradient[0]
                    self.weights[1:] -= self.learning_rate * (gradient[1:] + self.C * self.weights[1:] / self.m)
                else:
                    self.weights -= self.learning_rate * gradient
            # Print log-likelihood every so often
            if self.print_cost and step % (self.num_iterations//10) == 0:
                print(step, self.log_likelihood(preds = sigmoid(np.dot(features, self.weights)),
                                                target = target,
                                                m = self.m))     
        return self

    @property
    def intercept_(self):
        return self.weights[:1]
    
    @property
    def coef_(self):
        return self.weights[1:].reshape(1, self.n - 1)

    def predict(self, X):
        return self.predict_proba(X).round()

    def predict_proba(self, X):     
        if self.fit_intercept:
            X = self._fit_intercept(X)
    
        return sigmoid(np.dot(X, self.weights.T))


if __name__ == '__main__':
    from time import time

    from evaluation import test, TestSK
    from utils import load_beckernick, load_iris_2D, load_data
    from sklearn.model_selection import train_test_split
    from sklearn import linear_model
    

    # def sklearn_compare(learn, sk, X_train, X_test, y_train, y_test):
    #     data = (X_train, X_test, y_train, y_test)
    #     ts = TestSK(learn=learn,
    #                 sk=sk, 
    #                 data=data)

    #     attributes = ['intercept_', 'coef_']
    #     ts.compare_performance()
    #     print()
    #     ts.compare_attributes(attributes)
    #     print("="*80)


    # def test_lr(X, y, clf1, clf2):
    #     X_train, X_test, y_train, y_test = train_test_split(X,
    #                                                         y, 
    #                                                         test_size = 0.20, 
    #                                                         random_state = 0,
    #                                                         stratify = y)
    #     sklearn_compare(clf1, clf2, X_train, X_test, y_train, y_test)

    # clf = LogisticRegression(num_iterations=300000, learning_rate =.1,
    #                                 print_cost=False)
    # skclf = linear_model.LogisticRegression(penalty='none', solver='lbfgs')

    # sgd = LogisticRegressionSGD(num_iterations=500, learning_rate =1e-2,
    #                                 print_cost=False)

    # clf_r = LogisticRegression(num_iterations=300000, learning_rate =.1,
    #                          print_cost=False, penalty='l2', C=0.3)
    # skclf_r = linear_model.LogisticRegression(C=0.3, solver='lbfgs')

    # sgd_r = LogisticRegressionSGD(num_iterations=500, learning_rate =1e-2,
    #                             print_cost=False, penalty='l2', C=0.3)

    # # test command
    # # python3 lr.py > test_reports/lr.txt
    # start = time()
    # print("Start testing")

    # g1 = [clf, sgd, clf_r, sgd_r]
    # g2 = [skclf, skclf, skclf_r, skclf_r]


    # datasets = [
    #             load_beckernick, 
    #             load_iris_2D           
    #             ]
    # for dataset in datasets:
    #     X, y = dataset()
    #     [test_lr(X=X, y=y, clf1=c1, clf2=c2) for c1, c2 in zip(g1, g2)]

    # print('End testing')
    # end = time()
    # print(end - start)


