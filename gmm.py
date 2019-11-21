"""
GMM
"""
import numpy as np

from sklearn.base import BaseEstimator
# from kmeans import KMeans
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal as MVN
from scipy.special import softmax
from scipy.special import logsumexp



# https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.softmax.html
# def softmax(logits):
#     """
#     Args:
#         logits: N x D numpy array
#     """
#     # using logsumexp
#     # return np.exp(logits - logsumexp(logits))
#     epsilon = np.finfo(float).eps
#     logits = np.exp(logits - np.max(logits, axis=1, keepdims=True) + epsilon)
#     return logits/np.sum(logits, axis=1, keepdims=True)


# https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html#scipy.special.logsumexp
# def logsumexp(logits):
#     """
#     Args:
#         logits: N x D numpy array
#     Return:
#         s: N x 1 array where s[i,0] = logsumexp(logits[i,:])
#     """
#     # raise NotImplementedError
#     epsilon = np.finfo(logits.dtype).eps
#     m = np.max(logits, axis=1, keepdims=True)
#     logits = np.log(1 + np.exp((np.sum(logits - m + epsilon, axis=1, keepdims=True))))
#     return m + logits

# _LOG_2PI = np.log(2 * np.pi)   

# class MVN:
    
#     def __init__(self, mean, cov, eps = 1e-5):
#         self.mean = mean
#         self.n_features = len(mean)
#         s, u = np.linalg.eigh(cov)
#         s_pinv = np.array([0 if abs(x) <= eps else 1/x for x in s], dtype=float)
#         self.U = np.multiply(u, np.sqrt(s_pinv))
#         self.log_pdet = np.sum(np.log(s))

#     def logpdf(self, x):
#         x = x[np.newaxis, :]
#         x_m = x - self.mean
#         a = np.sum(np.square(np.dot(x_m, self.U)), axis=-1)
#         return -0.5 * (self.n_features * _LOG_2PI + self.log_pdet + a)

class GaussianMixture(BaseEstimator):

    def __init__(self, n_components=3,
                        covariance_type='full', 
                        max_iter=100, 
                        random_state=0,
                        verbose=0,
                        abs_tol=1e-16,
                        rel_tol=1e-16):
        self.max_iter = max_iter
        self.n_components =n_components
        self.reg_covar = 1e-6
        self.verbose = verbose
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol 
        self.costs = []

    def _ll_joint(self, X):
        """
        Return:
            ll.T: shape(m, n_features)
        """
        ll = np.empty((self.n_components, X.shape[0]))
        for k in range(self.n_components):
            # mvn = MVN(mean=self.means_[k], cov=self.covariances_[k], eps = eps)
            mvn = MVN(mean=self.means_[k], cov=self.covariances_[k])
            # print(mvn.logpdf(x=X).shape, np.log(self.pi[k]))
            
            ll[k] = mvn.logpdf(x=X) + np.log(self.weights_[k])
        # print(ll.T.shape)
        return ll.T

    # def _E_step(self, points, pi, mu, sigma, **kwargs):
    def _E_step(self, X):
        """
        Args:
            X: NxD numpy array, the observations
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the covariance matrix of each gaussian.
        Return:
            gamma: NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            
        Hint: You should be able to do this with just a few lines of code by using _ll_joint() and softmax() defined above. 
        """
        
        jll = self._ll_joint(X)
        resp = softmax(jll, axis=1)


        return jll, resp
                
    def _update_covariances(self, resp, X, total_weights_):
        """
        self.covariances_: shape(self.n_components, self.n, self.n)
        """
        self.covariances_ = np.empty((self.n_components, self.n, self.n))
        for k in range(self.n_components):
            diff = X - self.means_[k]
            self.covariances_[k] = np.dot(resp[:, k] * diff.T, diff) / total_weights_[k]
            # https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.flat.html
            # A 1-D iterator over the array.
            # add self.reg_cover to diagonal element of self.covariances_[k]
            # no significant difference found with iris dataset
            self.covariances_[k].flat[::self.n + 1] += self.reg_covar
        
    def _M_step(self, X, resp):
        """
        self.means_: shape(n_components, n_features)
        """
        total_weights_ = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        self.means_ = np.dot(resp.T, X) / total_weights_[:, np.newaxis]
        self._update_covariances(resp, X, total_weights_)
        self.weights_ = total_weights_ / self.m

    def _init_components(self, X):
        """
        Args:
            points: NxD numpy array, the observations
            K: number of components
            kwargs: any other args you want
        Return:
            pi: numpy array of length K, prior
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the covariance matrix of each gaussian.
            
        Hint: You could use the K-means results to initial GMM. It will help to converge. 
        For instance, you could use ids, mu = KMeans()(points, K)  to initialize.
        """
        # raise NotImplementedError
        
        self.m, self.n = X.shape
        

        resp = np.zeros((self.m, self.n_components))
        kmeans = KMeans(n_clusters=self.n_components, random_state=0)
        label = kmeans.fit(X).labels_
        resp[np.arange(self.m), label] = 1
        self._M_step(X, resp)
        

    def fit(self, X):
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            max_iters: maximum number of iterations
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            kwargs: any additional arguments you want
        Return:
            gamma: NxK array, the posterior distribution (a.k.a, the soft cluster assignment) 
                    for each observation.
            (pi, mu, sigma): (1xK np array, KxD numpy array, KxD numpy array), mu and sigma.
        
        Hint: You do not need to change it. For each iteration, we process E and M steps, then 
        """        
        self._init_components(X)
        # pbar = tqdm(range(max_iters))
        for it in range(self.max_iter):
            # E-step
            jll, resp = self._E_step(X)
            
            # M-step
            self._M_step(X, resp)
            
            # calculate the negative log-likelihood of observation
            loss = np.mean(logsumexp(jll, axis=1, keepdims=True))
            self.costs.append(loss)
            if it:
                diff = np.abs(prev_loss - loss)
                if self.verbose: print('diff', diff)
                if diff < self.abs_tol and diff / prev_loss < self.rel_tol:
                    break
            prev_loss = loss
            if self.verbose: print('iter %d, loss: %.4f' % (it, loss))
        self.costs = np.array(self.costs)
        return self


    def predict_proba(self, X):
        return self._E_step(X)[1]



    def predict(self, X):
        
        return np.argmax(self.predict_proba(X), axis=1)





