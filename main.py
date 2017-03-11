import numpy as np
import math
import csv
import random

TRAINING_FILE_NAME = "Data Set/trainingData.csv"
VERIFICATION_FILE_NAME = "Data Set/verificationData.csv"

def calculate_sigmoid_output(weights, features):
    sumScore = np.dot(weights, features)
    return float(1) / (1 + math.e**sumScore)


def getExamplesFromFile(fileName):
    with open(fileName, 'rU') as csvfile:
        trainingExamples = []
        reader = csv.reader(csvfile)
        for row in reader:
            example = (map(int, row[1:]),row[0])
            trainingExamples.append(example)
        return trainingExamples


def forwardPropagate(hiddenWeights, outputWeights, features):
    hiddenOutputs = []
    for hiddenWeight in hiddenWeights:
        hiddenOutputs.append(calculate_sigmoid_output(hiddenWeight,features))
    outputs = []
    for outputWeight in outputWeights:
        outputs.append(calculate_sigmoid_output(outputWeight,hiddenOutputs))
    return hiddenOutputs, outputs


def backpropagation(trainingExamples, alpha, n_hidden):
    '''
    Backpropagation algorithm using stochastic gradient descent.

    :param trainingExamples:
    :param alpha:
    :param n_hidden:
    :return:
    '''
    # initialize weights for each sigmoid randomly with values between -0.05 and 0.05
    output_weights = [[],[]]
    for output_weight in output_weights:
        for i in range(n_hidden):
            output_weight.append((random.random()-0.5)/10)
    hidden_weights = []
    numInputs = len(trainingExamples[0][0])
    for i in range(n_hidden):
        newWeight = []
        for i in range(numInputs):
            newWeight.append((random.random()-0.5)/10)
        hidden_weights.append(newWeight)

    # stochastic gradient descent
    minimumAlpha = 0.0001
    alphaDecrementInterval = (alpha - minimumAlpha) / len(trainingExamples)
    for trainingExample in trainingExamples:
        features = trainingExample[0]
        hiddenOutputs, outputs = forwardPropagate(hidden_weights, output_weights, features)

        # outputs[0] = edible, outputs[1] = poisonous
        # Calculate output errors based on target value
        outputErrors = []
        if trainingExample[1] == 'e':
            errorForEdibleOutput = outputs[0]*(1-outputs[0])*(1-outputs[0])
            errorForPoisonousOutput = outputs[1]*(1-outputs[1])*(0-outputs[1])
            outputErrors.append(errorForEdibleOutput)
            outputErrors.append(errorForPoisonousOutput)
        elif trainingExample[1] == 'p':
            errorForEdibleOutput = outputs[0] * (1 - outputs[0]) * (0 - outputs[0])
            errorForPoisonousOutput = outputs[1] * (1 - outputs[1]) * (1 - outputs[1])
            outputErrors.append(errorForEdibleOutput)
            outputErrors.append(errorForPoisonousOutput)

        #  Calculate error for hidden nodes
        hiddenErrors = []
        for i in range(n_hidden):
            sumOutputError = float(0)
            for j in range(len(outputs)):
                sumOutputError += output_weights[j][i] * outputErrors[j]
            hiddenError = hiddenOutputs[i] * (1 - hiddenOutputs[i]) * sumOutputError
            hiddenErrors.append(hiddenError)

        # Update weights for output nodes
        for j in range(len(output_weights)):
            for i in range(n_hidden):
                delta = alpha * outputErrors[j] * hiddenOutputs[i]
                output_weights[j][i] += delta

        # Update weights for hidden nodes
        for j in range(n_hidden):
            for i in range(len(features)):
                delta = alpha * hiddenErrors[j] * features[i]
                hidden_weights[j][i] += delta

        alpha -= alphaDecrementInterval

    return hidden_weights, output_weights




def main():
    trainingExamples = getExamplesFromFile(TRAINING_FILE_NAME)
    hidden_weights, output_weights = backpropagation(trainingExamples, 0.02, 30)
    numEdible = 0
    numPoisonous = 0
    totalEdibleScore = 0
    totalPoisionousScore = 0

    verificationExamples = getExamplesFromFile(VERIFICATION_FILE_NAME)
    for example in verificationExamples:
        print(example[1])
        hiddenO, output = forwardPropagate(hidden_weights, output_weights, example[0])
        score = output[0]-output[1]
        if example[1] == 'e':
            numEdible += 1
            totalEdibleScore += score
        elif example[1] == 'p':
            numPoisonous += 1
            totalPoisionousScore += score

    print "mean edible score:", totalEdibleScore/numEdible
    print "mean poisonous score:", totalPoisionousScore/numPoisonous


main()
