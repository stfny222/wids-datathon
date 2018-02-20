# def getTestArray(data, selected_headers):
#     array, headers, nulls = dataToArray(data)
#
#     # consider only columns that correspond to the training headers
#     valid_headers = []
#     for i in range (len(headers)):
#         # Exclude test_id from the features
#         if headers[i] in selected_headers and headers[i] != 'test_id' or headers[i] == 'is_female':
#             valid_headers.append(i)
#     print('Total headers: ' + str(len(headers)))
#     print('Valid headers: ' + str(len(valid_headers)))
#
#     # Save the names of the final headers
#     clean_headers = []
#     for i in valid_headers:
#         clean_headers.append(headers[i])
#
#     # Setup new array with only selected columns
#     valid_array = []
#     for row in array:
#         valid_row = []
#         for i in valid_headers:
#             valid_row.append(row[i])
#         valid_array.append(valid_row)
#
#     # Consider only rows with no null value
#     clean_array = []
#     for row in valid_array:
#         valid = True
#         for value in row:
#             if value is None:
#                 valid = False
#                 break
#         if valid:
#             clean_array.append(row)
#     print('Clean array: ' + str(len(clean_array)))
#
#     return clean_array, clean_headers
#
# def prepareTest(data, selected_headers):
#     print('--- PREPARE TEST DATA ---')
#     array, headers = getTestArray(data, selected_headers)
#     print('TEST HEADERS')
#     print(headers)
#     features = []
#     target = []
#     for i in range(10):
#         feature_row = []
#         row = array[i]
#         for j in range(len(row)):
#             feature_row.append(row[j])
#         features.append(feature_row)
#     return features
