from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from scipy.sparse import issparse

def test(classifier, X_train, X_test, y_train, y_test,
         report=False):
    """
    evaluate the performance of a classifier
    print confusion matrix and accuracy
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