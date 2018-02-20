from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import classifiers

# Function to parse either int or float values
def parse(n):
    try:
        return int(n)
    except ValueError:
        try:
            return float(n)
        except ValueError:
            return None

# Function that creates the initial array from the data
def dataToArray(data):
    headers = []
    nulls = []
    array = []
    # Store values in an array
    for index, line in enumerate(data):
        if index == 0:
            # The first row stores the name of the headers
            headers = line.split(',')
            # Create an array of zeros to store null values for every column
            nulls = [0] * len(headers)
        else:
            line = line.strip('\n')
            values = line.split(',')
            if len(values) > len(headers):
                print('Incorrect split with comma at position: ' + str(index))
            else:
                # Each item is an array of values
                item = []
                for i, value in enumerate(values):
                    if not value:
                        # Check if the cell is empty
                        item.append(None)
                        nulls[i] += 1
                    else:
                        # Append parsed value (int or float)
                        item.append(parse(value))
                # Append item to the main array
                array.append(item)
    # Return main array, headers array and the amount of nulls per column
    return array, headers, nulls

# Function that cleans the initial array
def getArray(data):
    array, headers, nulls = dataToArray(data)
    print('Total amount of rows: ' + str(len(array)))

    # Consider only columns without null values
    valid_headers = []
    for i in range (len(nulls)):
        # Exclude train_id from the features
        if nulls[i] == 0 and headers[i] != 'train_id':
            valid_headers.append(i)
    print('Total headers: ' + str(len(headers)))
    print('Valid headers (no null values): ' + str(len(valid_headers)))

    # Save the names of the final headers
    clean_headers = []
    for i in valid_headers:
        clean_headers.append(headers[i])

    # Setup new array with only selected columns
    valid_array = []
    for row in array:
        valid_row = []
        for i in valid_headers:
            valid_row.append(row[i])
        valid_array.append(valid_row)

    # Consider only rows with no null values
    clean_array = []
    for row in valid_array:
        valid = True
        for value in row:
            if value is None:
                valid = False
                break
        if valid:
            clean_array.append(row)
    print('Clean array size: ' + str(len(clean_array)))

    return clean_array, clean_headers

# Function to prepare data.
# It will split the data from 'train.csv' into train and test data (70/30)
def prepareData(data):
    print('--- PREPARE TRAINING DATA ---')
    array, headers = getArray(data)
    print('HEADERS')
    print(headers)
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    # 70%
    training_size = round(len(array) * 0.7)
    print('Training size: ' + str(training_size))
    for i in range(training_size):
        feature_row = []
        row = array[i]
        for j in range(len(row)):
            # Target feature: is_female
            if headers[j] == 'is_female':
                y_train.append(row[j])
            else:
                feature_row.append(row[j])
        X_train.append(feature_row)
    # 30%
    print('Test size: ' + str(len(array) - training_size))
    for i in range(training_size, len(array)):
        feature_row = []
        row = array[i]
        for j in range(len(row)):
            if headers[j] == 'is_female':
                y_test.append(row[j])
            else:
                feature_row.append(row[j])
        X_test.append(feature_row)
    return X_train, y_train, X_test, y_test, headers

if __name__ == '__main__':
    # Read file
    f_train = open('data/train.csv', 'r')
    # Get train and test data
    X_train, y_train, X_test, y_test, headers = prepareData(f_train)

    classifiers.run(X_train, y_train, X_test, y_test)

    # Print ROC for Multi-layer Perceptron classifier model
    # TODO:
    # Find optimum values for every parameter. This could be achieved with a genetic algorithm
    # print(
    #     'MLP ROC',
    #     neural.run(
    #         X_train, y_train, X_test, y_test,
    #         (72, 55, 14, 90, 49, 30, 27, 59),
    #         2,
    #         0,
    #         0.22330154978311556,
    #         0.8739521619101964,
    #         0.18876873023087026
    #     )
    # )

    # Print ROC for Logistic Regression model
    # TODO:
    # Find optimum values for every parameter.
    # User guide: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    # print(
    #     'LogisticRegression ROC',
    #     regression.run(X_train, y_train, X_test, y_test)
    # )
