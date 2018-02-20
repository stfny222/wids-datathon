import pandas as pd
from sklearn.model_selection import train_test_split
import classifiers

# Load data into a dataframe
dataframe0 = pd.read_csv('data/train.csv', low_memory = False)
print('Full dataframe size:', dataframe0.shape)

# Drop columns with null values
dataframe = dataframe0.dropna(axis = 1)
print('Clean dataframe size:', dataframe.shape)


# Store correlation values
correlations = []
not_correlated = []
for col in dataframe.columns.values:
    # Get correlation between every column and the target
    corr = dataframe['is_female'].corr(dataframe[col])
    corr = float("{0:.4f}".format(corr))
    # Exclude repeated correlations
    if corr not in correlations:
        correlations.append(corr)
    else:
        not_correlated.append(col)
    # if corr < 0.005 and corr > -0.005:
    #     not_correlated.append(col)

# print('CORRELATION BELOW 0', not_correlated)
not_correlated.append('train_id')
not_correlated.append('is_female')

# Consider is_female as the target
y = dataframe.is_female
# Consider every column except for the train_id as features
X = dataframe.drop(not_correlated, axis = 1)

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print('Train and test data shape:')
print(X_train.shape, X_test.shape)

print('--- RUNNING CLASSIFIERS ---')
classifiers.run(X_train, y_train, X_test, y_test)
