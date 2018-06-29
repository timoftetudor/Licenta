

# def compare(value1, value2):
#     arithmeticalValue = 0
#     for idx1, val1 in enumerate(value1):
#         # print(idx1, val1)  # in index I have the position and in val the value at that position
#         numbersIdentical = 0
#         for idx2, val2 in enumerate(value1[idx1]):
#             if value1[idx1][idx2] == value2[idx1][idx2]:
#                 numbersIdentical += 1
#         identicalPercent = numbersIdentical * 100 / numberPixelsApparitions
#         arithmeticalValue += identicalPercent
#     arithmeticalValue /= numberPixelsApparitions
#     print(arithmeticalValue)


# Now we prepare train_data and test_data.
# train = x[:, :50].reshape(-1, 400).astype(np.float32)  # Size = (2500,400)
# test = x[:, 50:100].reshape(-1, 400).astype(np.float32)  # Size = (2500,400)
#
# # Create labels for train and test data
# k = np.arange(10)
# train_labels = np.repeat(k, 250)[:, np.newaxis]
# test_labels = train_labels.copy()
#
# # Initiate kNN, train the data, then test it with test data for k=1
# knn = cv.ml.KNearest_create()
# knn.train(train, cv.ml.ROW_SAMPLE, train_labels)
# ret, result, neighbours, dist = knn.findNearest(test, k=5)
#
# # Now we check the accuracy of classification
# # For that, compare the result with test_labels and check which are wrong
# matches = result == test_labels
# correct = np.count_nonzero(matches)
# accuracy = correct * 100.0 / result.size
# print(accuracy)