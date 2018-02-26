from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import roc_auc_score

def getROC(
        clf,
        X_train,
        X_test,
        y_train,
        y_test
):
    clf.fit(X_train, y_train)
    prediction = clf.predict_proba(X_test)

    return roc_auc_score(y_test, prediction[:, 1])

def getBool(n):
    if n == 0:
        return False
    if n == 1:
        return True

def getLoss(n):
    if n == 0:
        return 'deviance' # Refers to deviance (= logistic regression) for classification with probabilistic outputs
    if n == 1:
        return 'exponential' # For loss 'exponential' gradient boosting recovers the AdaBoost algorithm

def run(
        # data
        X_train,
        y_train,
        X_test,
        y_test,
        # Random Forest parameters
        n_estimators_rf,
        bootstrap,
        random_state_rf,
        # Ada Boost parameters
        n_estimators_ab,
        learning_rate_ab,
        random_state_ab,
        # Gradient Tree Boosting parameters
        loss,
        learning_rate_gtb,
        n_estimators_gtb,
        max_depth,
        random_state_gtb,
):
    classifiers = [
        (
            'Random Forest',
            RandomForestClassifier(
                n_estimators=n_estimators_rf, # The number of trees in the forest.
                max_features=None, # The number of features to consider when looking for the best split (None: max_features=n_features)
                max_depth=None, # The maximum depth of the tree (None: nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples)
                bootstrap=getBool(bootstrap), # Whether bootstrap samples are used when building trees.
                n_jobs=-1, # The number of jobs to run in parallel for both fit and predict (-1: the number of jobs is set to the number of cores)
                random_state=random_state_rf, # Seed used by the random number generator
            )
        ),
        (
            'AdaBoost',
            AdaBoostClassifier(
                n_estimators=n_estimators_ab, # The maximum number of estimators at which boosting is terminated
                learning_rate=learning_rate_ab, # Learning rate shrinks the contribution of each classifier
                random_state=random_state_ab, # Seed used by the random number generator
            )
        ),
        (
            'Gradient Tree Boosting',
            GradientBoostingClassifier(
                loss=getLoss(loss), # Loss function to be optimized
                learning_rate=learning_rate_gtb, # Learning rate shrinks the contribution of each tree
                n_estimators=n_estimators_gtb, # The number of boosting stages to perform (large number to avoid over-fitting)
                max_depth=max_depth, # Maximum depth of the individual regression estimators
                random_state=random_state_gtb, # Seed used by the random number generator
            )
        ),
    ]

    # for (name, clf) in classifiers:
    #     print(name, getROC(clf, X_train, X_test, y_train, y_test))

    clfe = VotingClassifier(
        estimators=classifiers,
        voting='soft',
        n_jobs=-1, # The number of jobs to run in parallel for both fit and predict (-1: the number of jobs is set to the number of cores)
    )
    roc = getROC(clfe, X_train, X_test, y_train, y_test)

    print('Volting Classifier', roc)

    # Return ROC value and the classifier itself
    return roc, clfe
