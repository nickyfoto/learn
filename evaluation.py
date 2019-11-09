from sklearn.metrics import classification_report,confusion_matrix, accuracy_score

def test(classifier, X_train, X_test, y_train, y_test):
    """
    evaluate the performance of a classifier
    print confusion matrix and accuracy
    """
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