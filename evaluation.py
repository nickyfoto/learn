from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from scipy.sparse import issparse

class TestSK:
    """
    Compare the performance of our implementation
    with sklearn
    """
    def __init__(self, learn, sk, data):
        """
        Args:
            attributes: list of attributes we need to compare
            data: X_train, X_test, y_train, y_test returned from train_test_split
        """
        
        self.learn = learn
        self.sk = sk
        self.X_train, self.X_test, self.y_train, self.y_test = data

        self.learn_clf = self.learn.fit(self.X_train, self.y_train)
        self.sk_clf = self.sk.fit(self.X_train, self.y_train)
        # print('finished fitting')


    def _print_results(self, clf_names, metric, description, preds):
        for i, name in enumerate(clf_names):
            print(f"{name} {description}", metric(self.y_train, preds[i]))
            print()

    def compare_performance(self):
        clf_names = ['mylearn', 'sklearn']
        
        for name, clf in zip(clf_names, [self.learn, self.sk]):
            print(name, clf)
        
        preds = [clf.predict(self.X_train) for clf in [self.learn_clf, self.sk_clf]]
        
        metrics = [confusion_matrix, accuracy_score]
        descriptions = ["Confusion Matrix: \n", "Training Accuracy:"]

        for i, metric in enumerate(metrics):
            self._print_results(clf_names, metric, descriptions[i], preds)


    def compare_attributes(self, attributes):
        
        for attribute in attributes:
            print('comparing', attribute)
            print('mylearn:', getattr(self.learn_clf, attribute))
            print('sklearn:', getattr(self.sk_clf, attribute))
            print()

def test(classifier, X_train, X_test, y_train, y_test,
         report=False):
    """
    Evaluate the performance of One single classifier
    print confusion matrix and accuracy
    Args:
        report: T/F, whether to print classification report

    Returns:
        classifier: fitted classifier
    """
    print(f'testing {classifier}')
    if issparse(X_train):
        X_train = X_train.toarray()
        X_test = X_test.toarray()
    classifier.fit(X_train, y_train)
    pred = classifier.predict(X_train)
    if report: print(classification_report(y_train ,pred))
    print('Confusion Matrix: \n',confusion_matrix(y_train,pred))
    print()
    print('Training Accuracy: ', accuracy_score(y_train,pred))


    pred = classifier.predict(X_test)
    if report: print(classification_report(y_test ,pred ))

    print('Confusion Matrix: \n', confusion_matrix(y_test,pred))
    print()
    print('Testing Accuracy: ', accuracy_score(y_test,pred))
    print("=" * 80)
    return classifier