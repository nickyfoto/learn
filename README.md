# ml-learn

This is a repo can help you to truly understand all the nuts and bolts of fundamental machine learning algorithms by implementing them step by step. There are many ml libraries offer robust APIs for user to experiment with different algorithm. We believe it is much more fun dive into the core and demystify these encapsulations.

## Steps in step by step tutorial

To make the content easy to follow, every tutorial follow a coherent structure to present. 

- **Intro**: we begin with a toy-toy example to show you what you can do with this algorithm.
- **Inference**: The toy-toy example helps us to know what are parameters we need and how the model use these parameters to make predictions.
- **Training**: After presenting the inference part, we are motivated to know how the algorithm learns these parameters during the training.
- **Code**: Here we develop the necessary steps for the algorithm to be able to learn parameters with the data you fit in.
- **Theory**: You will find that most part of the code of is not as complex as you think but it's necessary for you to understand why you write the code as it is by consulting the theoretical support for this algorithm.
- **Application**: Finally we show some real world examples for you to have a grasp of the scenarios where the algorithm is applicable.

## Installation

```
git clone https://github.com/nickyfoto/ml-learn.git
pip3 install ./ml-learn
```

Install in editable mode

```
pip3 install -e ./ml-learn
```

## Supervised Learning

### Linear Model

- #### Linear Regression

	- [theory](https://nickyfoto.github.io/blog/entries/linear-regression) | [implementation](https://github.com/nickyfoto/ml-learn/blob/master/linear_regression.ipynb) | [examples](https://github.com/nickyfoto/ml-learn/blob/master/linear_regression_example.ipynb) | [code](https://github.com/nickyfoto/ml-learn/blob/master/lm.py)

- #### Logistic Regression

	- [theory](https://nickyfoto.github.io/blog/entries/logistic-regression) | [implementation](https://github.com/nickyfoto/ml-learn/blob/master/logistic_regression.ipynb) | examples | [code](https://github.com/nickyfoto/ml-learn/blob/master/lr.py)

- #### Perceptron

	- [implementation](https://github.com/nickyfoto/ml-learn/blob/master/tutorials/perceptron.ipynb)

- #### SVM

### Naive Bayes

- [theory](https://nickyfoto.github.io/blog/entries/naive-bayes) | [implementation](https://github.com/nickyfoto/ml-learn/blob/master/naive_bayes.ipynb) | [examples](https://github.com/nickyfoto/ml-learn/blob/master/tutorials/naive_bayes_examples.ipynb) | [code](https://github.com/nickyfoto/ml-learn/blob/master/src/learn/naive_bayes.py)

### Decisison Tree

## Unsupervised Learning

### KMeans

- [theory](http://cs229.stanford.edu/notes/cs229-notes7a.pdf) from cs229 | [implementation](https://github.com/nickyfoto/ml-learn/blob/master/kmeans.ipynb) | [examples](https://github.com/nickyfoto/ml-learn/blob/master/kmeans_example.ipynb) | [code](https://github.com/nickyfoto/ml-learn/blob/master/kmeans.py)

### Gaussian Mixture Model

- theory | [implementation](https://github.com/nickyfoto/ml-learn/blob/master/gmm.ipynb) | [examples](https://github.com/nickyfoto/ml-learn/blob/master/gmm_example.ipynb) | [code](https://github.com/nickyfoto/ml-learn/blob/master/gmm.py)

### Principal Component Analysis (PCA)

- [theory](https://nickyfoto.github.io/blog/entries/svd) | [examples](https://github.com/nickyfoto/ml-learn/blob/master/pca_example.ipynb) | [code](https://github.com/nickyfoto/ml-learn/blob/master/pca_example.ipynb)

## Todo

- [ ] adopt [scikit-learn-contrib template](https://github.com/scikit-learn-contrib/project-template)
- [ ] explain `softmax` and `logsumexp` in GMM.
- [ ] how to implement multinomial in logistic regression
- [ ] [Linear Discriminant Analysis (LDA)](https://web.stanford.edu/~hastie/Papers/ESLII.pdf)
- [ ] Linter: flake8, pylint