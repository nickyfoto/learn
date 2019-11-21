import numpy as np
from sklearn.base import BaseEstimator
from scipy.spatial import distance


def pairwise_dist(x, y):
    """
    Args:
        x: N x D numpy array
        y: M x D numpy array
    Return:
        dist: N x M array, where dist2[i, j] is the euclidean distance between 
        x[i, :] and y[j, :]
    """
    # raise NotImplementedError
    return (np.sum((x[np.newaxis,:] - y[:, np.newaxis])**2, axis=-1)**0.5).T


class KMeans(BaseEstimator):

    def __init__(self, n_clusters=8, 
                        max_iter=100,
                        abs_tol=1e-16,
                        rel_tol=1e-16,
                        verbose=False,
                        random_state=None):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.verbose = verbose
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.costs = []

    def _init_centers(self, X, n_clusters):
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            kwargs: any additional arguments you want
        Return:
            centers: K x D numpy array, the centers. 
        """
        # remove duplicate row from X
        unique_X = np.unique(X, axis=0)
        m, _ = unique_X.shape
        indices = np.arange(m)
        if self.random_state is not None:
            np.random.seed(self.random_state)
        np.random.shuffle(indices)
        self.cluster_centers_ = unique_X[indices[:self.n_clusters]]
    
    def predict(self, X):
        """
        update_assignment
        """
        dists = distance.cdist(X, self.cluster_centers_, metric='euclidean')
        return np.argmin(dists, axis=1)
    
    # def _update_centers(self, old_centers, cluster_idx, points):
    def _update_centers(self, X):
        """
        Args:
            old_centers: old centers KxD numpy array, where K is the number of clusters, and D is the dimension
            cluster_idx: numpy array of length N, the cluster assignment for each point
            points: NxD numpy array, the observations
        Return:
            centers: new centers, K x D numpy array, where K is the number of clusters, and D is the dimension.
        Note:
            It is possible to have fewer centers after this step.
        """
        
        # clusters = np.unique(cluster_idx)
        # clusters.shape = (clusters.shape[0], 1)
        clusters = np.arange(self.n_clusters).reshape(self.n_clusters, 1)
        self.cluster_centers_ = np.apply_along_axis(
                                    lambda label: X[self.labels_ == label.item()].mean(axis=0), 
                                    axis=1, arr=clusters)
        # return centers

    def _get_loss(self, X):
        """
        Args:
            centers: KxD numpy array, where K is the number of clusters, and D is the dimension
            cluster_idx: numpy array of length N, the cluster assignment for each point
            X: NxD numpy array, the observations
        Return:
            loss: a single float number, which is the objective function of KMeans. 
        """
        
        return sum( np.sum(np.square(X[self.labels_ == c] - self.cluster_centers_[c])) 
            for c in range(self.n_clusters) )


    def fit(self, X):
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            max_iters: maximum number of iterations (Hint: You could change it when debugging)
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            verbose: boolean to set whether method should print loss (Hint: helpful for debugging)
            kwargs: any additional arguments you want
        Return:
            cluster assignments: Nx1 int numpy array
            cluster centers: K x D numpy array, the centers
            loss: final loss value of the objective function of KMeans
        """
        self._init_centers(X, self.n_clusters)
        for it in range(self.max_iter):
            self.labels_ = self.predict(X)
            self._update_centers(X)
            loss = self._get_loss(X)
            self.costs.append(loss)
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < self.abs_tol and diff / prev_loss < self.rel_tol:
                    break
            prev_loss = loss
            if self.verbose:
                print('iter %d, loss: %.4f' % (it, loss))
        #return cluster_idx, centers, loss
         # = 
        self.costs = np.array(self.costs)
        return self