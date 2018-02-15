from sklearn.neural_network import MLPClassifier
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

def getActivation(i):
    values = ['identity', 'logistic', 'tanh', 'relu']
    return values[i]

def getSolver(i):
    values = ['lbfgs', 'sgd', 'adam']
    return values[i]

def run(
        X_train, y_train, X_test, y_test,
        hidden_layer_sizes,
        activation,
        solver,
        alpha,
        learning_rate,
        momentum
):

    clf = MLPClassifier(
        # The ith element represents the number of neurons in the ith hidden layer.
        hidden_layer_sizes=hidden_layer_sizes,
        # Activation function for the hidden layer.
        activation=getActivation(activation),
        # The solver for weight optimization.
        solver=getSolver(solver),
        # L2 penalty (regularization term) parameter.
        alpha=alpha,
        # The initial learning rate used.
        learning_rate_init=learning_rate,
        # Momentum for gradient descent update. Should be between 0 and 1.
        momentum=momentum,
        max_iter=5000
    )
    return getROC(clf, X_train, X_test, y_train, y_test)
