import numpy as np
import math

def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))

class LogisticRegression:

    def __init__(self, num_iterations = 2000, learning_rate = 0.5, 
                       fit_intercept = True,
                       print_cost = True):
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.fit_intercept = fit_intercept
        self.print_cost = print_cost

    def _fit_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def log_likelihood(self, preds, target, m):
        ll = - (target * np.log(preds) + (1 - target) * np.log(1 - preds)).sum() / m
        return ll


    def fit(self, features, target):
        if self.fit_intercept:
            features = self._fit_intercept(features)
            
        m, n = features.shape
        self.weights = np.zeros(n)
        
        for step in range(self.num_iterations):
            scores = np.dot(features, self.weights)
            predictions = sigmoid(scores)

            # Update weights with log likelihood gradient
            output_error_signal = predictions - target
            
            # gradient = np.dot(features.T, output_error_signal)
            gradient = np.dot(output_error_signal, features) / m
            self.weights -= self.learning_rate * gradient

            # Print log-likelihood every so often
            if self.print_cost and step % (self.num_iterations//10) == 0:
                print(step, self.log_likelihood(preds = predictions, 
                                                target = target, m=m))
    
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self._fit_intercept(X)
    
        return sigmoid(np.dot(X, self.weights))
    
    def predict(self, X):
        return self.predict_prob(X).round()

class LogisticRegressionSGD:
    """
    Logistic regression with stochastic gradient descent
    """

    def __init__(self, learning_rate = .1, 
                       num_iterations = 100, 
                       C = 0.0, 
                       fit_intercept = True,
                       print_cost = True):
        """
        Initialization of model parameters
        """
        self.learning_rate = learning_rate
        self.fit_intercept = fit_intercept
        self.num_iterations = num_iterations
        self.print_cost = print_cost
        self.C = C
        
    def _fit_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def log_likelihood(self, preds, target, m):
        ll = - (target * np.log(preds) + (1 - target) * np.log(1 - preds)).sum() / m
        return ll

    def fit(self, features, target):
        """
        Update model using a pair of training sample
        """
        if self.fit_intercept:
            # intercept = np.ones((features.shape[0], 1))
            features = self._fit_intercept(features)
        m, n = features.shape
        self.weights = np.zeros(n)

        for step in range(self.num_iterations):
            for i, x in enumerate(features):
                score = np.dot(x, self.weights)
                pred = sigmoid(score)
                error = pred - target[i]
                gradient = np.dot(error, x)
                self.weights[1:] -= self.C * self.weights[1:]
                self.weights -= self.learning_rate * gradient
            # Print log-likelihood every so often
            if self.print_cost and step % (self.num_iterations//10) == 0:
                print(step, self.log_likelihood(preds = sigmoid(np.dot(features, self.weights)),
                                                target = target,
                                                m = m))     

    def predict(self, X):
        return self.predict_prob(X).round()

    def predict_prob(self, X):     
        if self.fit_intercept:
            X = self._fit_intercept(X)
    
        return sigmoid(np.dot(X, self.weights))


if __name__ == '__main__':
    from evaluation import test
    from sklearn.model_selection import train_test_split
    from sklearn import linear_model
    from sklearn import datasets

    def main():
        np.random.seed(12)
        num_observations = 5000

        x1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_observations)
        x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)

        simulated_separableish_features = np.vstack((x1, x2)).astype(np.float32)
        
        simulated_labels = np.hstack((np.zeros(num_observations),
                                      np.ones(num_observations)))
        X_train, X_test, y_train, y_test = train_test_split(simulated_separableish_features,
                                                            simulated_labels, 
                                                            test_size = 0.20, 
                                                            random_state = 0,
                                                            stratify = simulated_labels)
        clf = LogisticRegression(num_iterations=300000, learning_rate =.1, fit_intercept=True,
                                    print_cost=True)
        clf = test(clf, X_train, X_test, y_train, y_test)
        print(clf.weights)
        

        sgd_clf = LogisticRegressionSGD(num_iterations=200, learning_rate = 1e-2)
        sgd_clf = test(sgd_clf, X_train, X_test, y_train, y_test)
        print(sgd_clf.weights)
        

        skclf = linear_model.LogisticRegression(penalty='none', fit_intercept=True, solver='lbfgs')
        skclf = test(skclf, X_train, X_test, y_train, y_test)
        print(skclf.intercept_, skclf.coef_)

        # model_sgd.fit(simulated_separableish_features, simulated_labels)
        # print(model_sgd.weights)
        

        # clf.fit(simulated_separableish_features, simulated_labels)
        # # print(clf.weights)
        # from sklearn.metrics import accuracy_score
        # pred = model.predict(simulated_separableish_features)
        # print('Accuracy: ', accuracy_score(simulated_labels,pred))


        
        iris = datasets.load_iris()
        X = iris.data[:, :2]
        y = (iris.target != 0) * 1
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.20, 
                                                    random_state = 0,
                                                    stratify = y)

        clf = LogisticRegression(num_iterations=300000, learning_rate=0.1, fit_intercept=True,
                                   print_cost=True)

        clf = test(clf, X_train, X_test, y_train, y_test)
        print(clf.weights)
        skclf = linear_model.LogisticRegression(penalty='none', fit_intercept=True, solver='lbfgs')
        skclf = test(skclf, X_train, X_test, y_train, y_test)
        print(skclf.intercept_, skclf.coef_)
        
                  
    main()