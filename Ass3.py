import numpy as np
import pandas as pd
import matplotlib as plt
from collections import Counter

with np.load('mnist.npz') as data:
    training_images = data["x_train"]
    training_labels = data["y_train"]
    testing_images = data["x_test"]
    testing_labels = data["y_test"]
# PCA
# print(training_images.shape)
reqClasses = [0, 1, 2]
reqData = []
reqLabels = []
testData = []
testLabels = []
for i in range(len(training_images)):
    if (training_labels[i] in reqClasses):
        reqData.append(training_images[i])
        reqLabels.append(training_labels[i])

for i in range(len(testing_images)):
    if (testing_labels[i] in reqClasses):
        testData.append(testing_images[i])
        testLabels.append(testing_labels[i])

reqData = np.array(reqData)
testData = np.array(testData)
# print(reqData.shape)
reqDataNum = reqData.shape[0]
testDataNum = testData.shape[0]
# print(reqDataNum)
reqLabels = np.array(reqLabels)
testLabels = np.array(testLabels)
reqData = reqData.reshape(reqDataNum, 784).T
testData = testData.reshape(testDataNum, 784).T


# print(reqData.shape)

def pca(samples_matrix, num_components):
    X_mean = np.mean(samples_matrix, axis=1, keepdims=True)

    # Center the data
    X_centered = samples_matrix - X_mean

    # Calculate the covariance matrix
    covariance_matrix = np.dot(X_centered, X_centered.T) / (reqDataNum - 1)

    # Eigen decomposition of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    U = eigenvectors[:, sorted_indices]

    # Project the centered data onto the principal components
    Y = np.dot(U.T, X_centered)

    # Reconstruct the data using the principal components
    X_reconstructed = np.dot(U, Y) + X_mean
    # print(U.shape)

    Up = U[:, :num_components]
    print(X_reconstructed)

    print(X_reconstructed.shape)
    print("////////////////////////////////")
    print(Up)
    return Up, X_reconstructed
    # X_final = np.dot(Up.T, X_reconstructed)
    # print(X_final.shape)


def countInList(element, list):
    count = 0
    for i in list:
        if (element == i):
            count += 1
    return count


def calculateGini(zeroes, ones, twos):
    total = zeroes + ones + twos
    gini = 0
    array = [zeroes, ones, twos]

    for i in range(3):
        gini += (array[i] / total) * (1 - (array[i] / total))
    return gini


def getGini(data_matrix, data_labels):
    vals = set(data_labels)
    print(type(data_labels[0]), vals)
    numFeatures = data_matrix.shape[0]
    numData = data_matrix.shape[1]
    splits = []
    gini = []

    for i in range(numFeatures):
        splits.append(np.mean(data_matrix[i]))
        leftLabels = []
        rightLabels = []

        for j in range(numData):
            if data_matrix[i][j] < splits[i]:
                leftLabels.append(data_labels[j])
            else:
                rightLabels.append(data_labels[j])

        zeroLeft = countInList(0, leftLabels)
        zeroRight = countInList(0, rightLabels)
        oneLeft = countInList(1, leftLabels)
        oneRight = countInList(1, rightLabels)
        twoLeft = countInList(2, leftLabels)
        twoRight = countInList(2, rightLabels)
        leftGini = calculateGini(zeroLeft, oneLeft, twoLeft)
        rightGini = calculateGini(zeroRight, oneRight, twoRight)

        hehe = (leftGini * (zeroLeft + oneLeft + twoLeft) + rightGini * (zeroRight + oneRight + twoRight)) / (
                zeroLeft + oneLeft + twoLeft + twoRight + oneRight + zeroRight)
        print(f'i = {i}, gini = {hehe}')
        print(f'zeroLeft = {zeroLeft}, oneLeft = {oneLeft}, twoLeft = {twoLeft}')
        print(f'zeroRight = {zeroRight}, oneRight = {oneRight}, twoRight = {twoRight}')
        print()
        gini.append(hehe)
    chosenGini = np.argmin(gini)
    print("*")
    print(gini)
    print(chosenGini)
    print(np.min(gini))
    print("*")

    return chosenGini


def splitUsingGiniValue(data_matrix, data_labels, giniIndex):
    numFeatures = data_matrix.shape[0]
    numData = data_matrix.shape[1]
    print("shape of data_matrix: ", data_matrix.shape)
    splitFeature = giniIndex

    splittingValue = np.mean(data_matrix[splitFeature])
    print("feature : ", splitFeature)
    left = []
    right = []
    leftLabels = []
    rightLabels = []
    for i in range(numData):
        if (data_matrix[splitFeature][i] < splittingValue):
            left.append(data_matrix[:, i])
            leftLabels.append(data_labels[i])
        else:
            right.append(data_matrix[:, i])
            rightLabels.append(data_labels[i])

    left = np.array(left).T
    leftLabels = np.array(leftLabels)
    right = np.array(right).T
    rightLabels = np.array(rightLabels)
    print(len(leftLabels), len(rightLabels))
    print("leftShape ", left.shape, "rightShape ", right.shape)
    return left, leftLabels, right, rightLabels, splittingValue


def getClass(array):
    zeroes = countInList(0, array)
    ones = countInList(1, array)
    twos = countInList(2, array)
    ans = [zeroes, ones, twos]
    return np.argmax(ans)


def predictClasses(data_matrix, data_labels, testData):
    Up, X_r = pca(data_matrix, 10)
    data = np.dot(Up.T, X_r)
    print(data.shape)
    print("//////////////////////////////////////////////")
    testData = np.dot(Up.T, testData)
    gini = getGini(data, data_labels)
    left, leftLabels, right, rightLabels, splitValue1 = splitUsingGiniValue(data, data_labels, gini)
    # for current leaf nodes
    gini1 = getGini(left, leftLabels)
    gini2 = getGini(right, rightLabels)
    ans = []
    left1 = []
    leftLabels1 = []
    right1 = []
    rightLabels1 = []
    origNode = []
    origLabels = []
    isSecondSplitOnLeft = False
    if (gini1 > gini2):
        # left_pred = predictClasses(left, leftLabels)
        origNode = right
        origLabels = rightLabels
        left1, leftLabels1, right1, rightLabels1, splitValue2 = splitUsingGiniValue(left, leftLabels, gini2)
        isSecondSplitOnLeft = True
    else:
        origNode = left
        origLabels = leftLabels
        left1, leftLabels1, right1, rightLabels1, splitValue2 = splitUsingGiniValue(right, rightLabels, gini2)

    print(f'gini1: {gini1}, gini2: {gini2}, splitValue1: {splitValue1}, splitValue2: {splitValue2}')

    print("lShape : ", left.shape)
    print("rShape : ", right.shape)
    classA = getClass(origLabels)
    classB = getClass(leftLabels1)
    classC = getClass(rightLabels1)

    print(classA, classB, classC, isSecondSplitOnLeft)
    print(testData.shape[1])

    predictions = []

    for i in range(testData.shape[1]):
        if (isSecondSplitOnLeft):
            if (testData[gini][i] < splitValue1):
                if (testData[gini2][i] < splitValue2):
                    predictions.append(classB)
                else:
                    predictions.append(classC)
            else:
                predictions.append(classA)
        else:
            if (testData[gini][i] < splitValue1):
                predictions.append(classA)
            else:
                if (testData[gini2][i] < splitValue2):
                    predictions.append(classB)
                else:
                    predictions.append(classC)

    return predictions


def calculate_classwise_accuracy(predictions, actual_labels, classes):
    classwise_correct = np.zeros(len(classes))
    classwise_total = np.zeros(len(classes))
    correct = 0
    total = len(predictions)
    for i in range(len(predictions)):
        if predictions[i] == actual_labels[i]:
            classwise_correct[actual_labels[i]] += 1
            correct += 1
        classwise_total[actual_labels[i]] += 1
    classwise_accuracy = classwise_correct / classwise_total
    print("Total accuracy:", correct / total)
    print("Class-wise correct:", classwise_correct)
    print("Class-wise total:", classwise_total)

    return classwise_accuracy

def generate_random_array(original_data, original_labels):
    num_cols = original_data.shape[1]

    # Choose random column indices with possible repetitions
    random_indices = np.random.choice(num_cols, size=num_cols, replace=True)

    # Select columns using the random indices
    random_array = original_data[:, random_indices]
    random_labels = original_labels[random_indices]
    return random_array, random_labels


def bagging(data_matrix, data_labels, test_data):
    data_matrix, data_labels = generate_random_array(data_matrix, data_labels)
    return predictClasses(data_matrix, data_labels, test_data)
# Usage:
predictions = predictClasses(reqData, reqLabels, testData)
print(set(predictions))
print(set(testLabels))
classwise_accuracy = calculate_classwise_accuracy(predictions, testLabels, [0, 1, 2])
print("Class-wise accuracy:", classwise_accuracy)

print("////////////////////////////////bagging part ///////////////////////////////////////////")
def tryBagging():
    allPredictions = []
    for i in range(5):
        allPredictions.append(bagging(reqData, reqLabels, testData))
    print(allPredictions)
    finalPredictions = []
    size = len(allPredictions[0])
    for i in range(size):
        currVotes = []
        for j in range(5):
            currVotes.append(allPredictions[j][i])
        finalPredictions.append(getClass(currVotes))
    classwise_accuracy = calculate_classwise_accuracy(finalPredictions, testLabels, [0, 1, 2])
    print("Class-wise accuracy:", classwise_accuracy)


tryBagging()


