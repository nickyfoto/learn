import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelBinarizer
from scipy.special import logsumexp

class NaiveBayes(BaseEstimator):
    """
    sklearn's implementation of Naive Bayes MultinomialNB
    """
    def __init__(self, alpha=1.0):
        self.alpha = 1.0

    def _count(self, X, Y):
        """Count and smooth feature occurrences."""
        self.feature_count_ += np.dot(Y.T, X)
        self.class_count_ += Y.sum(axis=0)


    def _update_class_log_prior(self):
        log_class_count = np.log(self.class_count_)
        self.class_log_prior_ = (log_class_count -
                                     np.log(self.class_count_.sum()))
    

    def _update_feature_log_prob(self):
        """Apply smoothing to raw counts and recompute log probabilities"""
        smoothed_fc = self.feature_count_ + self.alpha
        smoothed_cc = smoothed_fc.sum(axis=1)
        
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html
        # One shape dimension can be -1. In this case, the value is inferred
        # from the length of the array and remaining dimensions.
        self.feature_log_prob_ = (np.log(smoothed_fc) -
                                  np.log(smoothed_cc.reshape(-1, 1)))


    def fit(self, X, y):

        _, n_features = X.shape

        labelbin = LabelBinarizer()
        Y = labelbin.fit_transform(y)
        # if y is binary, we need to add
        # one extra column
        if Y.shape[1] == 1:
            Y = np.concatenate((1 - Y, Y), axis=1)

        self.classes_ = labelbin.classes_


        n_effective_classes = Y.shape[1]
        self.class_count_ = np.zeros(n_effective_classes, dtype=np.float64)
        self.feature_count_ = np.zeros((n_effective_classes, n_features),
                                       dtype=np.float64)

        self._count(X, Y)
        self._update_feature_log_prob()
        self._update_class_log_prior()
        return self

    def _joint_log_likelihood(self, X):
        """Calculate the posterior log probability of the samples X"""
        return np.dot(X, self.feature_log_prob_.T) + self.class_log_prior_

    def predict_log_proba(self, X):
        jll = self._joint_log_likelihood(X)
        log_prob_x = logsumexp(jll, axis=1)
        return jll - np.atleast_2d(log_prob_x).T

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]






if __name__ == '__main__':
    from evaluation import test
    from utils import load_data
    
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB

    X = np.random.randint(5, size=(7, 10))
    y = np.array([1, 2, 3, 4, 5, 6, 1])
    print(X)
    # from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB()
    clf.fit(X, y)
    print(clf.predict(X[2:3]))


    clf = NaiveBayes()
    clf.fit(X, y)
    print(clf.predict(X[2:3]))

    # emails = load_data('emails.csv')
    # emails.drop_duplicates(inplace = True)
    # emails_bow = CountVectorizer(stop_words='english').fit_transform(emails['text'])

    # X_train, X_test, y_train, y_test = train_test_split(emails_bow, emails['spam'], 
    #                                                     test_size = 0.20, 
    #                                                     random_state = 0,
    #                                                     stratify = emails['spam'])

    # test(NaiveBayes_v1(), X_train, X_test, y_train, y_test)



    # categories = ['alt.atheism', 'soc.religion.christian',
    #               'comp.graphics', 'sci.med']
    # twenty_train = fetch_20newsgroups(subset='train',
    #     categories=categories, shuffle=True, random_state=42)
    # vectorizer = CountVectorizer(stop_words='english')
    # X_train = vectorizer.fit_transform(twenty_train.data)
    # y_train = twenty_train.target

    # twenty_test = fetch_20newsgroups(subset='test',
    #  categories=categories, shuffle=True, random_state=42)
    # X_test = vectorizer.transform(twenty_test.data)
    # y_test = twenty_test.target

    # test(NaiveBayes_v1(), X_train, X_test, y_train, y_test)