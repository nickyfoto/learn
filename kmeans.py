import numpy as np
from sklearn.base import BaseEstimator

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

    def _init_centers(self, X, n_clusters):
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            kwargs: any additional arguments you want
        Return:
            centers: K x D numpy array, the centers. 
        """
        n_samples, _ = X.shape
        pi = np.arange(n_samples)
        if self.random_state is not None:
            np.random.seed(self.random_state)
        np.random.shuffle(pi)
        centers = X[pi[:self.n_clusters]]
        while np.unique(centers, axis=0).shape[0] != self.n_clusters:
            np.random.shuffle(pi)
            centers = X[pi[:self.n_clusters]]
        return centers
    
    def _update_assignment(self, centers, points):
        """
        Args:
            centers: KxD numpy array, where K is the number of clusters, and D is the dimension
            points: NxD numpy array, the observations
        Return:
            cluster_idx: numpy array of length N, the cluster assignment for each point
            
        Hint: You could call pairwise_dist() function.
        """
        # raise NotImplementedError
        distances = pairwise_dist(points, centers)
        # distances = distance.cdist(points, centers, metric='euclidean')
        cluster_idx = np.argmin(distances, axis=1)
        return cluster_idx
    
    def _update_centers(self, old_centers, cluster_idx, points):
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
        # raise NotImplementedError
        clusters = np.unique(cluster_idx)
        clusters.shape = (clusters.shape[0], 1)
        centers = np.apply_along_axis(lambda i: points[cluster_idx==i[0]].mean(axis=0), axis=1, arr=clusters)
        return centers

    def _get_loss(self, centers, cluster_idx, points):
        """
        Args:
            centers: KxD numpy array, where K is the number of clusters, and D is the dimension
            cluster_idx: numpy array of length N, the cluster assignment for each point
            points: NxD numpy array, the observations
        Return:
            loss: a single float number, which is the objective function of KMeans. 
        """
        # raise NotImplementedError
        #center_matrix = np.arange(points.shape[0])
        #center_matrix.shape = (points.shape[0], 1)
        #center_matrix = np.apply_along_axis(lambda i: centers[cluster_idx[i[0]]], axis=1, arr=center_matrix)
        # loss = np.sum((points - center_matrix)**2, axis = 1).mean()
        #loss = np.sum((points - center_matrix)**2)
        k_centers = centers.shape[0]
        loss = 0
        for k in range(k_centers):
            points_k = points[cluster_idx == k]
            loss += np.sum(np.square(points_k - centers[k]))
        return loss
        
    def predict(self, X):
        dist = pairwise_dist(X, self.cluster_centers_)
        return np.argmin(dist, axis=1)

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
        self.cluster_centers_ = self._init_centers(X, self.n_clusters)
        for it in range(self.max_iter):
            self.labels_ = self._update_assignment(self.cluster_centers_, X)
            self.cluster_centers_ = self._update_centers(self.cluster_centers_, self.labels_, X)
            loss = self._get_loss(self.cluster_centers_, self.labels_, X)
            K = self.cluster_centers_.shape[0]
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < self.abs_tol and diff / prev_loss < self.rel_tol:
                    break
            prev_loss = loss
            if self.verbose:
                print('iter %d, loss: %.4f' % (it, loss))
        #return cluster_idx, centers, loss
         # = 
        return self