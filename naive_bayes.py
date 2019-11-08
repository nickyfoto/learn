import numpy as np
from collections import Counter

from sklearn.metrics import classification_report,confusion_matrix, accuracy_score

class NaiveBayes_v0:
    """
    First implementation of Naive Bayes
    """
    def __init__(self):
        pass
    
    def fit(self, X, y):

        c = Counter(y)
        self.k = len(c)
        self.phi = [c[i]/len(y) for i in range(self.k)]

        m, n = X.shape
        self.phi_feature = np.empty(shape=(self.k, n))
        for i, _ in enumerate(self.phi_feature):
            indices = y == i
            row = (X[indices].sum(axis=0) + 1) / (c[i] + self.k)
            self.phi_feature[i] = row

    def predict(self, X):
        res = np.empty(len(X))
        for r in range(len(X)):
            prob = []
            for i in range(self.k):
                p = np.sum(np.log(self.phi_feature[i][X[r] > 0])) + np.log(self.phi[i])
                prob.append(p)
            res[r] = np.argmax(np.array(prob))
        return res


class NaiveBayes_v1:
    """
    First implementation of Naive Bayes MultinomialNB
    """
    def __init__(self):
        pass
    
    def fit(self, X, y):
        # n_classes = np.unique(y)
        # print(n_classes)
        c = Counter(y)
        self.k = len(c)
        # print(c)
        self.phi = [c[i]/len(y) for i in range(self.k)]
        # print(self.phi, sum(self.phi))
        m, n = X.shape
        self.phi_feature = np.empty(shape=(self.k, n))
        # print(self.phi_feature.shape)
        if self.k > 2:
            s = X.sum(axis=0)
            for i, _ in enumerate(self.phi_feature):
                indices = y == i
                row = (X[indices].sum(axis=0) + 1) / (s + self.k)
                self.phi_feature[i] = row
        else:
            for i, _ in enumerate(self.phi_feature):
                indices = y == i
                row = (X[indices].sum(axis=0) + 1) / (c[i] + self.k)
                self.phi_feature[i] = row
        # print(self.phi_feature.sum(axis=1))
        return self

    def predict(self, X):
        n, _ = X.shape
        res = np.empty(n)
        for r in range(n):
            prob = []
            for i in range(self.k):
                # print(X[r][X[r] == 1].shape)
                # print([X[r][X[r] == 1]].shape)
                p = np.sum(np.log( self.phi_feature[i][X[r] > 0] )) + np.log(self.phi[i])
                # print(i, p)
                prob.append(p)
            # print(prob)
            res[r] = np.argmax(np.array(prob))
        return res

def test(classifier, X_train, X_test, y_train, y_test):

    classifier.fit(X_train.toarray(), y_train)
    pred = classifier.predict(X_train.toarray())
    print(classification_report(y_train ,pred))
    print('Confusion Matrix: \n',confusion_matrix(y_train,pred))
    print()
    print('Accuracy: ', accuracy_score(y_train,pred))


    pred = classifier.predict(X_test.toarray())
    print(classification_report(y_test ,pred ))

    print('Confusion Matrix: \n', confusion_matrix(y_test,pred))
    print()
    print('Accuracy: ', accuracy_score(y_test,pred))
    return classifier

if __name__ == '__main__':
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import CountVectorizer
    




    df = pd.read_csv('emails.csv')
    df.drop_duplicates(inplace = True)
    messages_bow = CountVectorizer(stop_words='english').fit_transform(df['text'])

    X_train, X_test, y_train, y_test = train_test_split(messages_bow, df['spam'], 
                                                        test_size = 0.20, 
                                                        random_state = 0,
                                                        stratify = df['spam'])

    test(NaiveBayes_v1(), X_train, X_test, y_train, y_test)



    categories = ['alt.atheism', 'soc.religion.christian',
                  'comp.graphics', 'sci.med']
    twenty_train = fetch_20newsgroups(subset='train',
        categories=categories, shuffle=True, random_state=42)
    vectorizer = CountVectorizer(stop_words='english')
    X_train = vectorizer.fit_transform(twenty_train.data)
    y_train = twenty_train.target

    twenty_test = fetch_20newsgroups(subset='test',
     categories=categories, shuffle=True, random_state=42)
    X_test = vectorizer.transform(twenty_test.data)
    y_test = twenty_test.target

    test(NaiveBayes_v1(), X_train, X_test, y_train, y_test)