from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import roc_auc_score

from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.gaussian_process.kernels import RBF

def getROC(
        clf,
        X_train,
        X_test,
        y_train,
        y_test
):
    clf.fit(X_train, y_train)
    prediction = clf.predict_proba(X_test)

    return roc_auc_score(y_test, prediction[:,1])

def run(
        X_train, y_train, X_test, y_test
):
    classifiers = [
        # (
        #     'Logistic Regression',
        #     LogisticRegression(class_weight='balanced')
        # ),
        # (
        #     'Nearest Neighbors',
        #     KNeighborsClassifier(3),
        # ),
        # (
        #     'Linear SVM',
        #     SVC(kernel="linear", C=0.025)
        # ),
        # (
        #     'RBF SVM',
        #     SVC(gamma=2, C=1),
        # ),
        # (
        #     'Gaussian Process',
        #     GaussianProcessClassifier(1.0 * RBF(1.0))
        # ),
        # (
        #     'Decision Tree',
        #     DecisionTreeClassifier(max_depth=5)
        # ),
        (
            'Random Forest',
            RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=1,
            oob_score=False, random_state=1234, verbose=0,
            warm_start=False)
        ),
        (
            'AdaBoost',
            AdaBoostClassifier(
                n_estimators=100,
                learning_rate=0.5,
                random_state=None
            )
        ),
        (
            'Gradient Tree Boosting',
            GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=1.0,
                max_depth=1,
                random_state=0
            )
        ),
        # (
        #     'Neural Net',
        #     MLPClassifier(alpha=1)
        # ),
        # (
        #     'Naive Bayes',
        #     GaussianNB()
        # ),
        # (
        #     'QDA',
        #     QuadraticDiscriminantAnalysis()
        # ),
    ]
    for (name, clf) in classifiers:
        print(name)
        print(getROC(clf, X_train, X_test, y_train, y_test))

    print('Volting Classifier')
    clfe = VotingClassifier(
        estimators=classifiers,
        voting='soft'
    )
    print(getROC(clfe, X_train, X_test, y_train, y_test))
