"""
pytest -svv test_lm.py
pytest -svv -m 'not smoke' test_lm.py
pytest -vv --html=report.html --capture=sys

"""

import pytest
from pytest import approx

import numpy as np
from sklearn import linear_model, datasets
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as mse

from lm import LinearRegression, SGDRegressor
from utils import construct_polynomial_feats
from evaluation import within

def test_batch_sgdregressor():
    # test when fit_intercept = False

    POLY_DEGREE = 5
    NUM_OBS = 1000
    # random generator
    rng = np.random.RandomState(seed=4)
    true_weight = rng.rand(POLY_DEGREE + 1, 1)
    true_weight[2:, :] = 0
    x_all = np.linspace(-5, 5, NUM_OBS)
    x_all_feat = construct_polynomial_feats(x_all, POLY_DEGREE)
    y_all = np.dot(x_all_feat, true_weight) + rng.randn(x_all_feat.shape[0], 1) 
    indices = rng.permutation(NUM_OBS)
    train_indices = indices[:NUM_OBS//2]
    test_indices = indices[NUM_OBS//2:]
    # big learning rate will produce underflow error
    reg = SGDRegressor(fit_intercept=False, batch=True, learning_rate=1e-7,
                        max_iter=500000)
    reg.fit(x_all_feat[train_indices], y_all[train_indices])
    y_test_pred = reg.predict(x_all_feat[test_indices])
    test_mse = mse(y_pred=y_test_pred, y_true=y_all[test_indices])
    # print('test rmse: %.4f' % np.sqrt(test_mse))
    assert within(sk_val=1.1695 , val=np.sqrt(test_mse), tol = 1e-4)


@pytest.mark.smoke
def test_basic_sgdregressor():
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    # y = 1 * x_0 + 2 * x_1 + 3
    y = np.dot(X, np.array([1, 2])) + 3
    sksgdreg = linear_model.SGDRegressor(max_iter=1000, tol=None, penalty='none', random_state=0,
                                      learning_rate='constant', verbose=0, shuffle=False, eta0=1e-1)
    sksgdreg.fit(X, y)
    print()
    sgdreg = SGDRegressor(learning_rate=1e-1)
    sgdreg.fit(X, y)
    print(sgdreg)


    sgdreg_batch = SGDRegressor(learning_rate=1e-1, batch=True, max_iter=1e4)
    print(sgdreg_batch)
    sgdreg_batch.fit(X, y)

    print('sk score', sksgdreg.score(X, y))
    print('my score', sgdreg.score(X, y))
    print('my batch', sgdreg_batch.score(X, y))


    print(sksgdreg.coef_, sksgdreg.intercept_)
    print(sgdreg.coef_.flatten(), sgdreg.intercept_)    
    print(sgdreg_batch.coef_.flatten(), sgdreg_batch.intercept_)    

    assert sgdreg_batch.score(X, y) == 1.0

    assert sksgdreg.score(X, y) == sgdreg.score(X, y)
    assert np.allclose(sksgdreg.coef_,  sgdreg.coef_)
    assert np.allclose(sksgdreg.coef_,  sgdreg_batch.coef_)
    assert np.allclose(sksgdreg.intercept_,  sgdreg.intercept_)
    assert np.allclose(sksgdreg.intercept_,  sgdreg_batch.intercept_)
# sgdreg.coef_, sgdreg.intercept_ 
# sgdreg.predict(np.array([[3, 5]]))












boston = datasets.load_boston()
X = boston.data
y = boston.target

reg = LinearRegression()
skreg = linear_model.LinearRegression()

sgd = SGDRegressor()
sksgd = linear_model.SGDRegressor(random_state=0, verbose=0, #shuffle=False, 
                                        penalty='none')

scaled_X = StandardScaler().fit_transform(X) 











test1 = X, y, reg, skreg
test2 = scaled_X, y, sgd, sksgd
ids = ['boston, myLR, skLR',
       'scaled boston, mySGD, skSGD']



@pytest.mark.smoke
@pytest.mark.parametrize("X, y, reg1, reg2", [
                          test1,
                          test2
                          ], 
                          ids=ids)
class TestSK:

    def _print_results(self, clf_names, metric, description, preds):
        metrics = []
        for i, name in enumerate(clf_names):
            m = metric(self.y_train, preds[i])
            print(f"{name} {description}", m)
            metrics.append(m)
        print('abs diff: ', np.abs(metrics[0] - metrics[1]))
        # assert metrics[0] == approx(metrics[1])
        assert metrics[0] == approx(metrics[1], abs = 1e-1)
        print()

    def compare_performance(self, metrics, descriptions):
        clf_names = ['mylearn', 'sklearn']
        
        for name, clf in zip(clf_names, [self.learn, self.sk]):
            print(name, clf)
        print()
        preds = [clf.predict(self.X_train) for clf in [self.learn_clf, self.sk_clf]]

        for i, metric in enumerate(metrics):
            self._print_results(clf_names, metric, descriptions[i], preds)


    def compare_attributes(self, attributes):
        for attribute in attributes:
            print('comparing', attribute)
            m, s = getattr(self.learn_clf, attribute), getattr(self.sk_clf, attribute)
            print('mylearn:\n', m)
            print('sklearn:\n', s)
            print('abs diff:\n', np.abs(m - s))
            assert m == approx(s, abs = 1)
            print()

    def init(self, learn, sk, data):
        
        self.learn = learn
        self.sk = sk
        self.X_train, self.X_test, self.y_train, self.y_test = data
        self.learn_clf = self.learn.fit(self.X_train, self.y_train)
        self.sk_clf = self.sk.fit(self.X_train, self.y_train)

        self.learn_clf = learn.fit(self.X_train, self.y_train)
        self.sk_clf = sk.fit(self.X_train, self.y_train)

    def sklearn_compare(self, learn, sk, X_train, X_test, y_train, y_test):
        data = (X_train, X_test, y_train, y_test)
        self.init(learn, sk, data)

        

        attributes = ['intercept_', 'coef_']
        metrics = [r2_score, mean_squared_error]
        descriptions = ['r2:', 'MSE:']
        self.compare_performance(metrics, descriptions)
        print()
        self.compare_attributes(attributes)
        print("="*80)

    def test_reg(self, X, y, reg1, reg2):
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y, 
                                                            test_size = 0.20, 
                                                            random_state = 0)
        self.sklearn_compare(reg1, reg2, X_train, X_test, y_train, y_test)

