from datetime import datetime
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import time

with np.load('mnist.npz') as data:
    training_images = data["x_train"]
    training_labels = data["y_train"]
    testing_images = data["x_test"]
    testing_labels = data["y_test"]

training_data = []
train_labels = []
size = len(training_images)
countZero = 0
countOne = 0
val_images = []
val_labels = []
for i in range(size):
    if (training_labels[i] == 0):
        if (countZero < 1000):
            val_images.append(training_images[i])
            val_labels.append(-1)
            countZero += 1
        else:
            training_data.append(training_images[i])
            train_labels.append(-1)
    elif (training_labels[i] == 1):
        if (countOne < 1000):
            val_images.append(training_images[i])
            val_labels.append(1)
            countOne += 1
        else:
            training_data.append(training_images[i])
            train_labels.append(1)

training_images = training_data
training_labels = train_labels
testingSize = len(testing_labels)
testing_data = []
test_labels = []
for i in range(testingSize):
    if (testing_labels[i] == 0 or testing_labels[i] == 1):
        testing_data.append(testing_images[i])
        if (testing_labels[i] == 0):
            test_labels.append(-1)
        else:
            test_labels.append(1)

testing_images = np.array(testing_data)
testing_labels = np.array(test_labels)
val_images = np.array(val_images)
training_images = np.array(training_images)
numData = training_images.shape[0]
numTests = testing_images.shape[0]
numVal = val_images.shape[0]
training_images = training_images.reshape(numData, 784).T
testing_images = testing_images.reshape(numTests, 784).T
val_images = val_images.reshape(numVal, 784).T


def pca(samples_matrix, num_components):
    X_mean = np.mean(samples_matrix, axis=1, keepdims=True)
    X_centered = samples_matrix - X_mean
    numData = samples_matrix.shape[1]
    covariance_matrix = np.dot(X_centered, X_centered.T) / (numData - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    U = eigenvectors[:, sorted_indices]
    chosenOnes = U[:, :num_components]
    return X_mean, chosenOnes


X_mean, U = pca(training_images, 5)
training_images = np.dot(U.T, training_images - X_mean)
testing_images = np.dot(U.T, testing_images - X_mean)
print(testing_images.shape,  "TEST SHAPE")
val_images = np.dot(U.T, val_images - X_mean)

def getAllPossibleSplitsInDimension(data):
    size = len(data)
    splits = []
    for i in range(size - 1):
        splits.append((data[i] + data[i + 1])/2)
    return splits


def findBestSplit(data, labels, weights):
    # data = np.array(data)
    # labels = np.array(labels)
    # weights = np.array(weights)
    numFeatures = data.shape[0]
    numData = data.shape[1]
    bestSplitValues = [1 for i in range(numFeatures)]
    bestSplitErrors = [2 for i in range(numFeatures)]
    for i in range(numFeatures):
        start = datetime.now()
        minErrorInDimension = 2
        unique_values = np.unique(data[i, :])

        unique_values = np.sort(unique_values)
        splits = getAllPossibleSplitsInDimension(unique_values)
        splits = random.sample(splits, 1000) #donig this because otherwise it took a very very long time
        print("Number of splits : ", len(splits))
        print("done")
        for currSplit in splits:
            currPred = np.sign(data[i, :] - currSplit)
            wrongClassifications = np.zeros(numData, dtype=bool)
            wrong = 0
            for k in range(numData):
                if(currPred[k] != labels[k]):
                    wrongClassifications[k] = True
            incorrectWeight = 0
            totalWeight = 0
            for k in range(numData):
                if(wrongClassifications[k]):
                    incorrectWeight += weights[k]
                totalWeight += weights[k]
            currError = incorrectWeight / totalWeight
            if(currError < minErrorInDimension):
                bestSplitValues[i] = currSplit
                bestSplitErrors[i] = currError
                minErrorInDimension = currError
        end = datetime.now()
        print("Time taken for dimension: ", end - start)
    overallMinError = 2
    bestDimension = 0
    for i in range(numFeatures):
        if(bestSplitErrors[i] < overallMinError):
            overallMinError = bestSplitErrors[i]
            bestDimension = i
    # print("Time taken for dimension: ", end - start)
    print(bestDimension, bestSplitValues[bestDimension], overallMinError)
    return bestDimension, bestSplitValues[bestDimension], overallMinError

def adaboost(data_matrix, data_labels):
    numData = data_matrix.shape[1]
    weights = []
    for i in range(numData):
        weights.append(1/numData)
    decisionStumps = []
    allAlphas = []
    for i in range(300):
        start = datetime.now()
        print("here")
        bestDimension, bestSplit, bestError = findBestSplit(data_matrix, data_labels, weights)
        alpha = np.log((1 - bestError) / bestError)
        allAlphas.append(alpha)
        predictions = np.sign(data_matrix[bestDimension, :] - bestSplit)
        wrongClassifications = np.zeros(numData, dtype=bool)
        wrong = 0
        for k in range(numData):
            if(predictions[k] != data_labels[k]):
                wrongClassifications[k] = True
                wrong += 1
        print(f"Accuracy for h{i+1}(x) : " , 1 - wrong/numData)

        factor = np.exp(alpha * wrongClassifications)
        weights = np.multiply(weights, factor)
        decisionStumps.append((bestDimension, bestSplit))
        end = datetime.now()
        print(f"Time taken for iteration {i+1} : ", end - start)
    return allAlphas, decisionStumps

allAlphas, decisionStumps = adaboost(training_images, training_labels)

print(len(allAlphas), len(decisionStumps))
print(allAlphas, decisionStumps)
def predictClass(data_matrix, alphas, decisionStumps):
    numStumps = len(alphas)
    print(len(alphas), len(decisionStumps), "hehehee")
    numData = data_matrix.shape[1]
    predictions = np.zeros(numData)
    for i in range(numStumps):
        feature, split = decisionStumps[i]
        alpha = alphas[i]
        predictions += (alpha)*(np.sign(data_matrix[feature, :] - split))
    finalPredictions = np.sign(predictions)
    return finalPredictions

def calculateAccuracy(predictions, true_labels):
    numData = len(predictions)
    print(len(predictions), len(true_labels))
    print(predictions)
    print(true_labels)
    correct = 0
    for i in range(numData):
        if(predictions[i] == true_labels[i]):
            correct += 1
    return correct / numData

bestValAccuracy = 0
bestNumStumps = 0
val_labels = np.array(val_labels)
valAccuracies = []
size = len(allAlphas) + 1
for i in range(1, size):
    print("CURR : ", i)
    valCopy = val_labels.copy()
    copyAlphas = allAlphas.copy()
    copyStumps = decisionStumps.copy()
    valPred = predictClass(val_images, copyAlphas[:i], copyStumps[:i])
    valAccuracy = calculateAccuracy(valPred, val_labels)
    valAccuracies.append(valAccuracy)
    print(f"Val accuracy for h{i}(x): {valAccuracy}")
    if(valAccuracy > bestValAccuracy):
        bestValAccuracy = valAccuracy
        bestNumStumps = i

#Evaluate on testset

testPred = predictClass(testing_images, allAlphas[:bestNumStumps], decisionStumps[:bestNumStumps])
testAccuracy = calculateAccuracy(testPred, testing_labels)
print(f"Test accuracy for h{bestNumStumps}(x): {testAccuracy}")


plt.figure(figsize = (10, 6))

plt.plot(range(1, size), valAccuracies, label = "Validation Accuracy")
plt.xlabel("Number of Stumps")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy vs Number of Stumps")
plt.grid(True)
plt.legend()
plt.show()



def regressionTree(data_matrix, data_labels):
    numFeatures = data_matrix.shape[0]
    numData = data_matrix.shape[1]
    bestDimension = 0
    bestSplitValue = 0
    bestLeftMean = 0
    bestRightMean = 0
    bestNetSsr = float('inf')
    for feature in range(numFeatures):
        unique_values = np.unique(data_matrix[feature, :])
        unique_values = np.sort(unique_values)
        splits = getAllPossibleSplitsInDimension(unique_values)
        splits = random.sample(splits, 1000) #again, to reduce running time
        for currSplit in splits:
            leftLabels = []
            rightLabels = []
            size = len(data_labels)
            for i in range(size):
                if(data_matrix[feature][i] < currSplit):
                    leftLabels.append(data_labels[i])
                else:
                    rightLabels.append(data_labels[i])
            leftMean = np.mean(leftLabels)
            rightMean = np.mean(rightLabels)
            srLeft = 0
            srRight = 0
            currLeft = 0
            currRight = 0
            for i in range(size):
                if(data_matrix[feature][i] < currSplit):
                    srLeft += (leftLabels[currLeft] - leftMean)**2
                    currLeft += 1
                else:
                    srRight += (rightLabels[currRight] - rightMean)**2
                    currRight += 1
            netSsr = srLeft + srRight
            if(netSsr < bestNetSsr):
                bestNetSsr = netSsr
                bestDimension = feature
                bestSplitValue = currSplit
                bestLeftMean = leftMean
                bestRightMean = rightMean
    return bestDimension, bestSplitValue, bestLeftMean, bestRightMean

def gradient_boost(data_matrix, data_labels, valData, valLabels):
    decisionStumps = []
    MSEList = []
    data_labels = np.array(data_labels)
    valLabels = np.array(valLabels)
    data_labels = data_labels.astype(float)
    valLabels = valLabels.astype(float)
    for i in range(300):
        bestDimension, bestSplit, bestLeftMean, bestRightMean = regressionTree(data_matrix, data_labels)
        decisionStumps.append((bestDimension, bestSplit, bestLeftMean, bestRightMean))
        size = len(data_labels)
        for j in range(size):
            if(data_matrix[bestDimension][j] < bestSplit):
                data_labels[j] -= bestLeftMean * 0.01
            else:
                data_labels[j] -= bestRightMean * 0.01
        size = len(valLabels)
        for j in range(size):
            if (valData[bestDimension][j] < bestSplit):
                valLabels[j] -= bestLeftMean * 0.01
            else:
                valLabels[j] -= bestRightMean * 0.01
        valMSE = np.mean(valLabels**2)
        MSEList.append(valMSE)
        print(f"Validation MSE for h{i}(x): {valMSE}")
    return MSEList, decisionStumps


MseList, decisionStumps = gradient_boost(training_images, training_labels, val_images, val_labels)
plt.figure()
plt.plot(range(1, len(MseList) + 1), MseList)
plt.xlabel("Number of Stumps")
plt.ylabel("Mean Squared Error")
plt.title("Validation MSE vs Number of Stumps")
plt.grid(True)
plt.show()

bestStump = np.argmin(MseList)
lowestError = MseList[bestStump]
print(f"Best number of stumps: {bestStump + 1}")
print(f"Lowest validation MSE: {lowestError}")

testing_labels = testing_labels.astype(float)

bestDecisionStump = decisionStumps[bestStump]

def predictRegression(data_matrix, decision_stumps, num_stumps):
    num_data = data_matrix.shape[1]
    predictions = np.zeros(num_data)
    for i in range(num_stumps):
        dimension, split, left_mean, right_mean = decision_stumps[i]
        for j in range(num_data):
            if data_matrix[dimension][j] < split:
                predictions[j] += left_mean
            else:
                predictions[j] += right_mean
    predictions = np.sign(predictions)
    return predictions


test_predictions = predictRegression(testing_images, decisionStumps[:bestStump+1], bestStump + 1)

test_MSE = np.mean((test_predictions - testing_labels) ** 2)
print(f"Test MSE for best stump: {test_MSE}")