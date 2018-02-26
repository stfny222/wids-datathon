import pandas as pd
from sklearn.model_selection import train_test_split
import feature_selection_ga
import classifiers

# Load train data into a dataframe
df_train0 = pd.read_csv('data/train.csv', low_memory = False)
print('Full train dataframe size:', df_train0.shape)

# Drop columns with null values
df_train = df_train0.dropna(axis = 1)

excluded = []
# Exclude the train_id and the target columns
excluded.append('train_id')
excluded.append('is_female')

# Consider is_female as the target
y = df_train.is_female
# Consider every column except for the excluded ones as features
df_train = df_train.drop(excluded, axis = 1)

print('--- RUNNING GENETIC ALGORITHM ---')
POPULATION_SIZE = 5
MAX_ITER = 30
MUTATION_PROB = 1
final_excluded_indexes = feature_selection_ga.run(
    df_train,
    y,
    POPULATION_SIZE,
    MAX_ITER,
    MUTATION_PROB,
)

# Generate an array containing the final excluded columns
final_excluded = []
for index, col in enumerate(df_train.columns.values):
    # Only exclude the columns which index correspond to a negative value in the excluded_indexes array
    if final_excluded_indexes[index] == 0:
        excluded.append(col)

X = df_train.drop(final_excluded, axis = 1)

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print('Final train and test data shape:')
print(X_train.shape, X_test.shape)

print('--- RUNNING FINAL CLASSIFIER ---')
# Use the optimum values found running the parameter_values_ga (ROC: 0.965156618062)
roc, clf = classifiers.run(
    # data
    X_train,
    y_train,
    X_test,
    y_test,
    # Random Forest parameters
    15,
    1,
    1074,
    # Ada Boost parameters
    72,
    0.6746530840498172,
    2940,
    # Gradient Tree Boosting parameters
    1,
    0.18546035948574746,
    139,
    4,
    9090,
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
submit.to_csv('submit2.csv', index=None)
