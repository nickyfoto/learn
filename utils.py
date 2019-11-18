import pandas as pd
import numpy as np
import h5py
from skimage import io, transform
import matplotlib.pyplot as plt
from sklearn import datasets

def load_data(fn, path='./datasets/', sep=','):
    data = pd.read_csv(path + fn, sep=sep)
    return data

def load_beckernick():
    """
    Generated data example from
    https://github.com/beckernick/logistic_regression_from_scratch/blob/master/logistic_regression_scratch.ipynb
    """
    print(load_beckernick.__doc__)
    np.random.seed(12)
    num_observations = 5000

    x1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_observations)
    x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)

    simulated_separableish_features = np.vstack((x1, x2)).astype(np.float32)
    
    simulated_labels = np.hstack((np.zeros(num_observations),
                                  np.ones(num_observations)))
    return simulated_separableish_features, simulated_labels
    
def load_iris_2D():
    """
    load iris and combine label 1, 2 into 1
    only use the first two features of X
    """
    print(load_iris_2D.__doc__)
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    y = (iris.target != 0) * 1
    return X, y

def load_cat_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    # return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
    # train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_cat_dataset()
    
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    train_set_x = train_set_x_flatten/255.
    test_set_x = test_set_x_flatten/255.
    X_train = train_set_x.T
    X_test = test_set_x.T
    
    y_train = train_set_y_orig.flatten()
    y_test = test_set_y_orig.flatten()

    num_px = train_set_x_orig.shape[1]
    return X_train, X_test, y_train, y_test, num_px, classes




def predict_image(clf, fname, num_px, classes, plot_image=False):
    image_file = "images/" + fname
    image = io.imread(image_file) / 255.
    resized_image = transform.resize(image, output_shape=(num_px,num_px)).reshape((1, num_px*num_px*3))
    my_predicted_image = clf.predict(resized_image)
    if plot_image: plt.imshow(image)
    print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + 
          classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")


# plots utils

def plot_costs(clf):
    if clf.costs.ndim > 1:
        plt.plot(clf.costs.T.mean(axis=1))
    else:
        plt.plot(clf.costs)
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title("Learning rate =" + str(clf.learning_rate))
    plt.show()

def scatter_plot(X, y):
    plt.figure(figsize=(10, 6))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='b', label='0')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='r', label='1')
    plt.legend()
    return plt

def contour_plot(X, y, model):
    scatter_plot(X, y)
    x1_min, x1_max = X[:,0].min(), X[:,0].max(),
    x2_min, x2_max = X[:,1].min(), X[:,1].max(),
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    grid = np.c_[xx1.ravel(), xx2.ravel()]
    probs = model.predict_proba(grid)[:,0].reshape(xx1.shape)
    plt.contour(xx1, xx2, probs, [0.5], linewidths=1, colors='black');


def plot_boundary(clf, X, y, grid_step=.01, poly_featurizer=None):
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_step),
                         np.arange(y_min, y_max, grid_step))


    # to every point from [x_min, m_max]x[y_min, y_max]
    # we put in correspondence its own color
    Z = clf.predict(poly_featurizer.transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, cmap=plt.cm.Paired)


def construct_polynomial_feats(x, degree):
    """
    Args:
        x: numpy array of length N, the 1-D observations
        degree: the max polynomial degree
    Return:
        feat: numpy array of shape Nx(degree+1), remember to include 
        the bias term. feat is in the format of:
        [[1.0, x1, x1^2, x1^3, ....,],
         [1.0, x2, x2^2, x2^3, ....,],
         ......
        ]
    """
    
    n = x.shape[0]
    res = np.empty((degree+1,n))
    for i in range(degree+1):
        res[i] = np.power(x,i)
    
    return res.T


def plot_curve(x, y, curve_type='.', color='b', lw=2):
    plt.plot(x, y, curve_type, color=color, linewidth=lw)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)


def rmse(pred, label): 
    '''
    This is the root mean square error.
    Args:
        pred: numpy array of length N * 1, the prediction of labels
        label: numpy array of length N * 1, the ground truth of labels
    Return:
        a float value
    '''
    #raise NotImplementedError
    N = pred.shape 
    return np.sqrt(((label - pred) ** 2).sum()/N)[0]
