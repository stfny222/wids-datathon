import pandas as pd
from sklearn.model_selection import train_test_split
import ga
import classifiers

# Load train data into a dataframe
df_train0 = pd.read_csv('data/train.csv', low_memory = False)
print('Full train dataframe size:', df_train0.shape)

# Drop columns with null values
df_train = df_train0.dropna(axis = 1)
print('Clean train dataframe size:', df_train.shape)


# Store correlation values
correlations = []
excluded = []
for col in df_train.columns.values:
    # Get correlation between every column and the target
    corr = df_train['is_female'].corr(df_train[col])
    # Consider only 4 decimals for correlation values
    corr = float("{0:.4f}".format(corr))
    # Exclude repeated correlations
    if corr not in correlations:
        correlations.append(corr)
    else:
        excluded.append(col)
    # Exclude correlations with values closer to 0
    # if corr < 0.005 and corr > -0.005:
    #     excluded.append(col)

# Exclude also the train_id and the target columns
excluded.append('train_id')
excluded.append('is_female')

# Consider is_female as the target
y = df_train.is_female
# Consider every column except for the excluded ones as features
X = df_train.drop(excluded, axis = 1)

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print('Train and test data shape:')
print(X_train.shape, X_test.shape)

print('--- RUNNING GENETIC ALGORITHM ---')
POPULATION_SIZE = 8
MAX_ITER = 20
MUTATION_PROB = 10
parameter_values = ga.run(
    X_train,
    y_train,
    X_test,
    y_test,
    POPULATION_SIZE,
    MAX_ITER,
    MUTATION_PROB,
)

print('--- RUNNING FINAL CLASSIFIER ---')
roc, clf = classifiers.run(
    # data
    X_train,
    y_train,
    X_test,
    y_test,
    # Random Forest parameters
    parameter_values[0],
    parameter_values[1],
    parameter_values[2],
    parameter_values[3],
    parameter_values[4],
    # Ada Boost parameters
    parameter_values[5],
    parameter_values[6],
    parameter_values[7],
    # Gradient Tree Boosting parameters
    parameter_values[8],
    parameter_values[9],
    parameter_values[10],
)
print('Final ROC: ', roc)

# Load test data into dataframe
df_test0 = pd.read_csv('data/test.csv', low_memory = False)
# print('Full test dataframe size:', df_test0.shape)

# Reindex columns according to the training data
df_test = df_test0.reindex(columns=X.columns, fill_value=0)
# print('Clean test dataframe size:', df_train.shape)

# Run predictions for test data
print('Running predictions for test data...')
predictions = clf.predict_proba(df_test)[:,1]
# Fill a new 'is_female' column with the probability predictions
df_test0['is_female'] = predictions
# Generate a new dataframe containing only the test_id and the probability
submit = df_test0[['test_id', 'is_female']]
# Generate a csv file for submission
print('Saving submission...')
submit.to_csv('submit.csv', index=None)
