"""
Linear Model

Linear Regression
"""

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse

class LinearRegression(BaseEstimator):
    """
    Solve theta analytically using normal equation
    """
    def __init__(self, 
                    fit_intercept = True,
                    penalty = None,
                    c_lambda = 0):
        
        self.fit_intercept = fit_intercept
        self.penalty = penalty
        self.c_lambda = c_lambda
        self.penalty = penalty

    def _fit_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.hstack((intercept, X))

    def fit(self, X, y):
        """
        Args:
            X: mxn numpy array
            y: (m,)
        Return:
            self
        """
        if self.fit_intercept:
            X = self._fit_intercept(X)
            self.m, self.n = X.shape
            if self.penalty is None:
                self.weights = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
            else:
                L = np.eye(self.n)
                L[0,0] = 0
                self.c_lambda *= L
                self.weights = np.linalg.pinv(X.T.dot(X) + self.c_lambda).dot(X.T).dot(y)
        else:
            self.m, self.n = X.shape
            if self.penalty is None:
                self.weights = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
            else:
                L = np.eye(self.n)
                self.c_lambda *= L
                self.weights = np.linalg.pinv(X.T.dot(X) + self.c_lambda).dot(X.T).dot(y)

        return self

    @property
    def intercept_(self):
        if self._fit_intercept:
            return self.weights[0]
        return np.array([0])

    @property
    def coef_(self):
        if self._fit_intercept:
            return self.weights[1:]
        return self.weights

    def predict(self, X):
        if self.fit_intercept:
            X = self._fit_intercept(X)
        
        return np.dot(X, self.weights)



class SGDRegressor(BaseEstimator):

    def __init__(self, fit_intercept=True, max_iter=1000,
                    learning_rate=0.001,
                    penalty=None,
                    c_lambda=0,
                    batch=False):

        self.fit_intercept = fit_intercept
        self.max_iter = int(max_iter)
        self.learning_rate = learning_rate
        self.penalty = penalty
        self.c_lambda = c_lambda
        self.batch = batch


    def fit(self, X, y):
        if self.fit_intercept:
            self.intercept_ = np.zeros(shape=(1,))
        self.m, n_features = X.shape
        self.coef_ = np.zeros(shape=(1, n_features))
        self.costs = np.empty((self.max_iter, ))

        y.shape = (self.m, 1)

        for i in range(self.max_iter):
            if self.batch:
                if self.fit_intercept:
                    preds = np.dot(X, self.coef_.T) + self.intercept_
                    error = preds - y
                    gradient = np.dot(X.T, error)
                    if self.penalty:
                        self.coef_ -= self.learning_rate * (gradient.T + self.c_lambda * self.coef_) / self.m
                    else:
                        self.coef_ -= self.learning_rate * gradient.T / self.m
                        self.intercept_ -= self.learning_rate * error.sum() / self.m
                else:
                    preds = np.dot(X, self.coef_.T)
                    # print(preds.T)
                    # print(np.isnan(preds).any(), i)
                    error = preds - y
                    gradient = np.dot(X.T, error)
                    if self.penalty:
                        self.coef_ -= self.learning_rate * (gradient.T + self.c_lambda * self.coef_) / self.m
                    else:
                        # print(gradient.T.shape, gradient.T, error)
                        self.coef_ -= self.learning_rate * gradient.T / self.m
                train_mse = mse(y_pred=preds, y_true=y)
                self.costs[i] = np.sqrt(train_mse)
            else:
                for idx, x in enumerate(X):
                    if self.fit_intercept:
                        pred = np.dot(x, self.coef_.T) + self.intercept_
                        error = pred - y[idx]
                        gradient = x * error
                        if self.penalty:
                            self.coef_ -= self.learning_rate * (gradient.T + self.c_lambda * self.coef_ / self.m)
                        else:
                            self.coef_ -= self.learning_rate * gradient.T
                        self.intercept_ -= self.learning_rate * error
                    else:
                        pred = np.dot(x, self.coef_.T)
                        error = pred - y[idx]
                        gradient = x * error
                        if self.penalty:
                            self.coef_ -= self.learning_rate * (gradient.T + self.c_lambda * self.coef_ / self.m)
                        else:
                            self.coef_ -= self.learning_rate * gradient.T
                train_mse = mse(y_pred=self.predict(X), y_true=y)
                self.costs[i] = np.sqrt(train_mse)
        return self

    def predict(self, X):
        if self.fit_intercept:
            return np.dot(X, self.coef_.T) + self.intercept_
        return np.dot(X, self.coef_.T)


    def score(self, X, y):
        return r2_score(y_true=y.flatten(), y_pred = self.predict(X))

class Ridge(BaseEstimator):
    """
    Note:
        No equivalent Normal Equation solver in sklearn Ridge implementation
    """
    def __init__(self, fit_intercept=True, alpha=1.0):
        self.weights = None
        self.fit_intercept = fit_intercept
        self.c_lambda = alpha

    def _fit_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.hstack((intercept, X))

    def fit(self, X, y):
        if self.fit_intercept:
            X = self._fit_intercept(X)
        self.m, self.n = X.shape
        
        L = np.eye(self.n)
        L[0,0] = 0
        self.c_lambda *= L
        self.weights = np.linalg.inv(X.T.dot(X) + self.c_lambda).dot(X.T).dot(y)
        return self

    @property
    def coef_(self):
        if self.n == 1:
            return self.weights[1:]
        return self.weights[1:]

    def predict(self, X):
        if self.fit_intercept:
            X = self._fit_intercept(X)
        return np.dot(X, self.weights)


if __name__ == '__main__':
    
    from time import time

    from sklearn import datasets
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn import linear_model
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    from evaluation import TestSK

    def sklearn_compare(learn, sk, X_train, X_test, y_train, y_test):
        data = (X_train, X_test, y_train, y_test)
        ts = TestSK(learn=learn,
                                sk=sk, 
                                data=data)

        attributes = ['intercept_', 'coef_']
        metrics = [r2_score, mean_squared_error]
        descriptions = ['r2:', 'MSE:']
        ts.compare_performance(metrics, descriptions)
        print()
        ts.compare_attributes(attributes)
        print("="*80)

    def test_reg(X, y, reg1, reg2):
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y, 
                                                            test_size = 0.20, 
                                                            random_state = 0)
        sklearn_compare(reg1, reg2, X_train, X_test, y_train, y_test)


    # test command
    # python3 lm.py > test_reports/reg.txt
    start = time()
    print("Start testing")

    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    
    # SGD is sensitive to feature scale 
    # so standarize X before training



    reg = LinearRegression()
    skreg = linear_model.LinearRegression()

    print('Testing Standardized boston dataset')

    #test_reg(X, y, reg, skreg)

    scaler = StandardScaler()
    X = scaler.fit_transform(X) 

    #test_reg(X, y, sgd, sksgd)


    sgd_r = SGDRegressor(penalty='l2', alpha=1e2, max_iter=2000)
    sksgd_r = linear_model.SGDRegressor(random_state=0, verbose=1, shuffle=False)
    test_reg(X, y, sgd_r, sksgd_r)

    print('Testing 1D diabetes')
    diabetes = datasets.load_diabetes()
    # Use only one feature
    X = diabetes.data[:, np.newaxis, 2]
    y = diabetes.target
    # test_reg(X, y, reg, skreg)


    scaler = StandardScaler()
    X = scaler.fit_transform(X) 

    # test_reg(X, y, sgd, sksgd)

    # Testing SGD with l2 regularization
    
    

    print('End testing')
    end = time()
    print(end - start)

    # def f(x):
    #     return x * np.sin(x)
    # x = np.linspace(0, 10, 100)
    # rng = np.random.RandomState(0)
    # rng.shuffle(x)
    # x = np.sort(x[:20])
    # y = f(x)

    # # create matrix versions of these arrays
    # X = x[:, np.newaxis]
    # # print(X.shape)
    # reg = Ridge()
    # reg.fit(X, y)
    # print(reg.predict(X))