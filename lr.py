import numpy as np
import math, inspect
from collections import defaultdict
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator
from scipy.special import expit as sigmoid

# def sigmoid(scores):
    # return 1 / (1 + np.exp(-scores))

class LogisticRegression(BaseEstimator):

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

    def log_likelihood(self, preds, target, m):
        ll = - (target * np.log(preds + np.finfo(float).eps) + 
                (1 - target) * np.log(1 - preds + np.finfo(float).eps)).sum() / m
        return ll


    def fit(self, features, target):
        if self.fit_intercept:
            features = self._fit_intercept(features)
            
        self.m, self.n = features.shape
        self.weights = np.zeros(self.n)
        
        for step in range(self.num_iterations):
                    
            scores = np.dot(features, self.weights)
            predictions = sigmoid(scores)
            output_error_signal = predictions - target
            gradient = np.dot(features.T, output_error_signal) / self.m

            if self.penalty:
                self.weights[0] -= self.learning_rate * gradient[0]
                self.weights[1:] -= self.learning_rate * (gradient[1:] + self.C * self.weights[1:] / self.m)
            else:
                self.weights -= self.learning_rate * gradient

            if step % (self.num_iterations // self.steps) == 0:
                cost = self.log_likelihood(preds = predictions, 
                                                target = target, m=self.m)
                if self.print_cost: print(step, cost)
                self.costs.append(cost)
    
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
        return self.predict_proba(X).round()

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
        """
        Initialization of model parameters
        """
        self.learning_rate = learning_rate
        self.fit_intercept = fit_intercept
        self.num_iterations = num_iterations
        self.print_cost = print_cost
        if C is 0: 
            self.C = 0
        else:
            self.C = 1 / C
        self.penalty = penalty

    def _fit_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def log_likelihood(self, preds, target, m):
        ll = - (target * np.log(preds) + (1 - target) * np.log(1 - preds)).sum() / m
        return ll

    def fit(self, features, target):
        """
        Update model using a pair of training sample
        """
        if self.fit_intercept:
            # intercept = np.ones((features.shape[0], 1))
            features = self._fit_intercept(features)
        m, n = features.shape
        self.weights = np.zeros(n)

        for step in range(self.num_iterations):
            for i, x in enumerate(features):
                score = np.dot(x, self.weights)
                pred = sigmoid(score)
                error = pred - target[i]
                gradient = np.dot(error, x)
                # self.weights[0] -= self.weights[1:] - self.C * self.weights[1:]
                if self.penalty:
                    self.weights[0] -= self.learning_rate * gradient[0]
                    self.weights[1:] -= self.learning_rate * (gradient[1:] + self.C * self.weights[1:] / m)
                else:
                    self.weights -= self.learning_rate * gradient
            # Print log-likelihood every so often
            if self.print_cost and step % (self.num_iterations//10) == 0:
                print(step, self.log_likelihood(preds = sigmoid(np.dot(features, self.weights)),
                                                target = target,
                                                m = m))     
        # mimic sklearn props
        self.intercept_ = np.array([self.weights[0]])
        self.coef_ = self.weights[1:].reshape(1, n - 1)

    def predict(self, X):
        return self.predict_prob(X).round()

    def predict_prob(self, X):     
        if self.fit_intercept:
            X = self._fit_intercept(X)
    
        return sigmoid(np.dot(X, self.weights))


if __name__ == '__main__':
    from evaluation import test
    from utils import load_beckernick, load_iris_2D, load_data
    from sklearn.model_selection import train_test_split
    from sklearn import linear_model
    

    def sklearn_compare(own, sk, X_train, X_test, y_train, y_test):
        own = test(own, X_train, X_test, y_train, y_test)
        sk = test(sk, X_train, X_test, y_train, y_test)
        print('comparing weights difference:')
        print(own.intercept_, own.coef_)
        print(sk.intercept_, sk.coef_)

    def test_lr(X, y, clf1, clf2):
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y, 
                                                            test_size = 0.20, 
                                                            random_state = 0,
                                                            stratify = y)
        sklearn_compare(clf1, clf2, X_train, X_test, y_train, y_test)

    clf = LogisticRegression(num_iterations=300000, learning_rate =.1,
                                    print_cost=False)
    skclf = linear_model.LogisticRegression(penalty='none', solver='lbfgs')

    sgd = LogisticRegressionSGD(num_iterations=500, learning_rate =1e-2,
                                    print_cost=False)

    clf_r = LogisticRegression(num_iterations=300000, learning_rate =.1,
                             print_cost=False, penalty='l2', C=0.3)
    skclf_r = linear_model.LogisticRegression(C=0.3, solver='lbfgs')

    sgd_r = LogisticRegressionSGD(num_iterations=500, learning_rate =1e-2,
                                print_cost=False, penalty='l2', C=0.3)

    X, y = load_beckernick()
    test_lr(X=X, y=y, clf1=clf, clf2=skclf)
    test_lr(X=X, y=y, clf1=sgd, clf2=skclf)
    test_lr(X=X, y=y, clf1=clf_r, clf2=skclf_r)
    test_lr(X=X, y=y, clf1=sgd_r, clf2=skclf_r)

    X, y = load_iris_2D()
    test_lr(X=X, y=y, clf1=clf, clf2=skclf)
    test_lr(X=X, y=y, clf1=sgd, clf2=skclf)
    test_lr(X=X, y=y, clf1=clf_r, clf2=skclf_r)
    test_lr(X=X, y=y, clf1=sgd_r, clf2=skclf_r)



