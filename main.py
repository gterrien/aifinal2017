import numpy as np
import math
import csv
import random

TRAINING_FILE_NAME = "Data Set/trainingData.csv"
VERIFICATION_FILE_NAME = "Data Set/verificationData.csv"
TEST_FILE_NAME = "Data Set/testData.csv"

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


def stochastic_backpropagation(trainingExamples, alpha, n_hidden):
    '''
    Backpropagation algorithm using stochastic gradient descent.

    :param trainingExamples:
    :param alpha:
    :param n_hidden:
    :return:
    '''
    # initialize weights for each sigmoid randomly with values between -0.05 and 0.05
    output_weights = [[]]
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
        '''if trainingExample[1] == 'e':
            errorForEdibleOutput = outputs[0]*(1-outputs[0])*(1-outputs[0])
            errorForPoisonousOutput = outputs[1]*(1-outputs[1])*(0-outputs[1])
            outputErrors.append(errorForEdibleOutput)
            outputErrors.append(errorForPoisonousOutput)
        elif trainingExample[1] == 'p':
            errorForEdibleOutput = outputs[0] * (1 - outputs[0]) * (0 - outputs[0])
            errorForPoisonousOutput = outputs[1] * (1 - outputs[1]) * (1 - outputs[1])
            outputErrors.append(errorForEdibleOutput)
            outputErrors.append(errorForPoisonousOutput)'''
        target_value = 0.0
        if trainingExample[1] == 'e':
            target_value = 1.0
        errorForEdibleOutput = outputs[0] * (target_value - outputs[0]) * (target_value - outputs[0])
        outputErrors.append(errorForEdibleOutput)

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

def batch_backpropagation(trainingExamples, alpha, n_hidden):
    '''
    Backpropagation algorithm using gradient descent.

    :param trainingExamples:
    :param alpha:
    :param n_hidden:
    :return:
    '''
    # initialize weights for each sigmoid randomly with values between -0.05 and 0.05
    output_weights = [[], []]
    for output_weight in output_weights:
        for i in range(n_hidden):
            output_weight.append((random.random() - 0.5) / 10)
    hidden_weights = []
    numInputs = len(trainingExamples[0][0])
    hidden_weights = []
    numInputs = len(trainingExamples[0][0])
    for i in range(n_hidden):
        newWeight = []
        for i in range(numInputs):
            newWeight.append((random.random() - 0.5) / 10)
        hidden_weights.append(newWeight)

    minimumAlpha = 0.0001
    while alpha > minimumAlpha:
        hidden_delta = hidden_weights[:]
        for hidden_node in hidden_delta:
            for delta in hidden_node:
                delta = 0.0
        output_delta = output_weights[:]
        for output_node in output_delta:
            for delta in output_node:
                delta = 0.0
        for trainingExample in trainingExamples:
            features = trainingExample[0]
            hiddenOutputs, outputs = forwardPropagate(hidden_weights, output_weights, features)
            for i in range(n_hidden):
                for j in range(len(features)):
                    pass
                    #hidden_delta[i] += alpha * ()



def main():
    trainingExamples = getExamplesFromFile(TRAINING_FILE_NAME)
    hidden_weights, output_weights = stochastic_backpropagation(trainingExamples, 0.1, 14)
    numEdible = 0
    numPoisonous = 0
    totalEdibleScore = 0.0
    totalPoisionousScore = 0.0

    verificationExamples = getExamplesFromFile(VERIFICATION_FILE_NAME)
    num_correct = 0
    num = 0
    for example in verificationExamples:
        hiddenO, output = forwardPropagate(hidden_weights, output_weights, example[0])
        if example[1] == 'e':
            totalEdibleScore += output[0]
            numEdible += 1
        elif example[1] == 'p':
            totalPoisionousScore += output[0]
            numPoisonous += 1
    meanEdibleScore = totalEdibleScore / numEdible
    meanPoisonousScore = totalPoisionousScore / numPoisonous
    midpoint = (meanEdibleScore + meanPoisonousScore) / 2.0
    numPoisonous = 0
    numEdible = 0
    for example in verificationExamples:
        hiddenO, output = forwardPropagate(hidden_weights, output_weights, example[0])
        '''score = output[0]-output[1]
        if example[1] == 'e':
            numEdible += 1
            totalEdibleScore += score
        elif example[1] == 'p':
            numPoisonous += 1
            totalPoisionousScore += score

    print "mean edible score:", totalEdibleScore/numEdible
    print "mean poisonous score:", totalPoisionousScore/numPoisonous'''
        print "Output " + str(output)
        print "Type " + example[1]

        classification = 0
        if output[0] >= midpoint:
            classification = 1
        if example[1] == 'e':
            if classification == 1:
                num_correct += 1
        elif example[1] == 'p':
            if classification == 0:
                num_correct += 1
        num += 1
    print str(float(num_correct) / num)
main()
