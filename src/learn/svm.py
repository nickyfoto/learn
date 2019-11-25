


from sklearn.base import BaseEstimator
import numpy as np
import cvxopt.solvers
cvxopt.solvers.options['show_progress'] = False

def compute_w(multipliers, X, y):
    return sum(multipliers[i] * y[i] * X[i] for i in range(len(y)))

def compute_b(w, X, y):
    return sum([y[i] - np.dot(w, X[i]) for i in range(len(X))])/len(X)




class SVC(BaseEstimator):

    def __init__(self):
        pass


    def fit(self, X, y):
        """
        convert y into -1, 1, dtype: float
        """
        self.classes_, y = np.unique(y, return_inverse=True)
        y = np.where(y == 0, -1, y).astype(np.float64)

        self.m, self.n = X.shape
        K = np.array([np.dot(X[i], X[j])
                        for j in range(self.m)
                        for i in range(self.m)]).reshape((self.m, self.m))


        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-1 * np.ones(self.m))
        # print(P.size, q.size)
        # Equality constraints
        A = cvxopt.matrix(y, (1, self.m))
        b = cvxopt.matrix(0.0)
        # Inequality constraints
        G = cvxopt.matrix(np.diag(-1 * np.ones(self.m)))
        h = cvxopt.matrix(np.zeros(self.m))

        # Solve the problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        # Lagrange multipliers
        multipliers = np.ravel(solution['x'])

        has_positive_multiplier = multipliers > 1e-7
        sv_multipliers = multipliers[has_positive_multiplier]
        # print(sv_multipliers)
        support_vectors = X[has_positive_multiplier]
        support_vectors_y = y[has_positive_multiplier]

        w = compute_w(multipliers, X, y)
        self.w_from_sv = compute_w(sv_multipliers, support_vectors, support_vectors_y)
        self.b = compute_b(w, support_vectors, support_vectors_y) # -9.666668268506335



    def predict(self, X):
        y = np.dot(X, self.w_from_sv) + self.b
        return self.classes_.take(np.asarray(np.where(y > 0, 1, 0), dtype=np.intp))














