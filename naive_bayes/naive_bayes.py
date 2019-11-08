import numpy as np

class NaiveBayes_v0:
    """
    First implementation of Naive Bayes
    """
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.phi_y_1 = y.sum()/len(y)
        self.phi_y_0 = 1 - self.phi_y_1
        indices = y == 1
        self.phi_j_y1 = (X[indices].sum(axis=0) + 1)/(y.sum() + 2)
        self.phi_j_y0 = (X[~indices].sum(axis=0) + 1)/((len(y) - y.sum()) + 2)

    def predict(self, X):
        res = np.empty(len(X))
        for r in range(len(X)):
            p_y1 = np.sum(np.log(self.phi_j_y1[X[r] == 1])) + np.log(self.phi_y_1)
            p_y0 = np.sum(np.log(self.phi_j_y0[X[r] == 1])) + np.log(self.phi_y_0)
            res[r] = np.argmax(np.array([p_y0, p_y1]))
        return res
