from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def getROC(
        clf,
        X_train,
        X_test,
        y_train,
        y_test
):
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)

    return roc_auc_score(y_test, prediction)

def run(
        X_train, y_train, X_test, y_test
):

    clf = LogisticRegression(class_weight='balanced')
    return getROC(clf, X_train, X_test, y_train, y_test)
