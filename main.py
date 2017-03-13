import numpy as np
import math
import csv
import random
import operator

TRAINING_FILE_NAME = "Data Set/trainingDataGeneric.csv"
VERIFICATION_FILE_NAME = "Data Set/verificationDataGeneric.csv"
TEST_FILE_NAME = "Data Set/testDataGeneric.csv"
ALPHA = .05
N_HIDDEN = 14

def calculate_sigmoid_output(weights, features):
    sumScore = np.dot(weights, features)
    return float(1) / (1 + math.e**sumScore)

def calculate_perceptron_output(weights, features):
    sumScore = np.dot(weights, features)
    output = 1.0
    if sumScore < 0.0:
        output = -1.0
    return output

def update_perceptron_weights(oldWeights, trainingExample):
    '''targetValue = 1.0
    if trainingExample[0] == 'p':
        targetValue = -1.0'''
    targetValue = trainingExample[1]
    newWeights = map(operator.add, oldWeights, np.dot(targetValue, trainingExample[0]))
    return newWeights


def getExamplesFromFile(fileName):
    with open(fileName, 'rU') as csvfile:
        trainingExamples = []
        reader = csv.reader(csvfile)
        for row in reader:
            example = (map(int, row[1:]),float(row[0]))
            trainingExamples.append(example)
        return trainingExamples


def forwardPropagate(hiddenWeights, outputWeights, features):
    hiddenOutputs = []
    for hiddenWeight in hiddenWeights:
        hiddenOutputs.append(calculate_sigmoid_output(hiddenWeight,features))
    #outputs = []
    '''for outputWeight in outputWeights:
        outputs.append(calculate_perceptron_output(outputWeight,hiddenOutputs))'''
    output = calculate_perceptron_output(outputWeights, hiddenOutputs)
    return hiddenOutputs, output


def stochastic_backpropagation(trainingExamples, alpha, n_hidden):
    '''
    Backpropagation algorithm using stochastic gradient descent.

    :param trainingExamples:
    :param alpha:
    :param n_hidden:
    :return:
    '''
    # initialize weights for each sigmoid randomly with values between -0.05 and 0.05
    '''output_weights = []
    for output_weight in output_weights:
        for i in range(n_hidden):
            output_weight.append(0.0)'''
    output_weights = []
    for i in range(n_hidden):
        output_weights.append(0.0)
    hidden_weights = []
    numInputs = len(trainingExamples[0][0])
    for i in range(n_hidden):
        newWeight = []
        for j in range(numInputs):
            newWeight.append((random.random()-0.5)/10)
        hidden_weights.append(newWeight)

    # stochastic gradient descent
    minimumAlpha = 0.0001
    alphaDecrementInterval = (alpha - minimumAlpha) / len(trainingExamples)
    for trainingExample in trainingExamples:
        features = trainingExample[0]
        hiddenOutputs, output = forwardPropagate(hidden_weights, output_weights, features)

        # outputs[0] = edible, outputs[1] = poisonous
        # Calculate output errors based on target value
        #outputErrors = []
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
        '''target_value = 1.0
        if trainingExample[1] == 'p':
            target_value = -1.0'''
        target_value = trainingExample[1]
        '''errorForEdibleOutput = outputs[0] * (target_value - outputs[0]) * (target_value - outputs[0])
        outputErrors.append(errorForEdibleOutput)'''
        output_error = output * (target_value - output) * (target_value - output)

        #  Calculate error for hidden nodes
        hiddenErrors = []
        for i in range(n_hidden):
            sumOutputError = float(0)
            '''for j in range(len(outputs)):
                sumOutputError += output_weights[j][i] * outputErrors[j]
            hiddenError = hiddenOutputs[i] * (1 - hiddenOutputs[i]) * sumOutputError
            hiddenErrors.append(hiddenError)'''
            sumOutputError = output_weights[i] * output_error
            hiddenError = hiddenOutputs[i] * (1 - hiddenOutputs[i]) * sumOutputError
            hiddenErrors.append(hiddenError)

        '''for j in range(len(outputs)):
            if outputs[j] != target_value:
                output_weights[j] = map(operator.add, output_weights[j], np.dot(target_value, hiddenOutputs))
                #output_weights = update_perceptron_weights(output_weights, trainingExample)'''
        # Update weights for output node if necessary
        if output != target_value:
            output_weights = map(operator.add, output_weights, np.dot(float(target_value), hiddenOutputs))

        # Update weights for hidden nodes
        for j in range(n_hidden):
            for i in range(len(features)):
                delta = alpha * hiddenErrors[j] * features[i]
                hidden_weights[j][i] += delta

        alpha -= alphaDecrementInterval

    return hidden_weights, output_weights

def calculate_perceptron_weights(trainingExamples):
    weights = [0.0] * len(trainingExamples[0][0])
    for example in trainingExamples:
        '''target_value = 1.0
        if example[1] == 'p':
            target_value = -1.0'''
        target_value = float(example[1])
        #print example[0]
        output = calculate_perceptron_output(weights, example[0])
        if output != target_value:
            weights = update_perceptron_weights(weights, example)
    return weights


def perceptron_classify(weights, testExamples):
    num = 0
    num_correct = 0
    for example in testExamples:
        output = calculate_perceptron_output(weights, example[0])
        targetValue = 1.0
        if example[1] == 'p':
            targetValue = -1.0
        if targetValue == output:
            num_correct += 1
        num += 1
    percent_correct = float(num_correct) / num
    return percent_correct

def neural_net_classify(hidden_weights, output_weights, testExamples):
    num_correct = 0
    num = 0
    for example in testExamples:
        hiddenO, output = forwardPropagate(hidden_weights, output_weights, example[0])
        classification = output
        '''if classification == 1.0:
            if example[1] == 'e':
                num_correct += 1
        elif classification == -1.0:
            if example[1] == 'p':
                num_correct += 1'''
        if classification == example[1]:
            num += 1
    percent_correct = float(num_correct) / num
    return percent_correct


def tune():
    max_correct = 0.0
    max_alpha = 0.0
    max_hidden = 0
    trainingExamples = getExamplesFromFile(TRAINING_FILE_NAME)
    verificationExamples = getExamplesFromFile(VERIFICATION_FILE_NAME)
    resultsFile = open("results4.txt", 'w')
    for n_hidden in range(25, 30):
        for alpha in np.arange(.01, .20, .01):
            hidden_weights, output_weights = stochastic_backpropagation(trainingExamples, alpha, n_hidden)
            num = 0
            num_correct = 0
            for example in verificationExamples:
                hiddenO, output = forwardPropagate(hidden_weights, output_weights, example[0])
                classification = output[0]
                if classification == 1.0:
                    if example[1] == 'e':
                        num_correct += 1
                elif classification == -1.0:
                    if example[1] == 'p':
                        num_correct += 1
                num += 1
            percent_correct = float(num_correct) / num
            resultString = "alpha: " + str(alpha) + " n_hidden: " + str(n_hidden) + " percent correct: " + str(percent_correct)
            resultsFile.write(resultString + '\n')
            if percent_correct > max_correct:
                max_alpha = alpha
                max_hidden = n_hidden
                max_correct = percent_correct
            print "Finished alpha: " + str(alpha) + " n_hidden: " + str(n_hidden)
    maxString = "Best: alpha: " + str(max_alpha) + " n_hidden: " + str(max_hidden) + " percent correct: " + str(max_correct)
    resultsFile.write(maxString + '\n')
    resultsFile.close()

def main():
    trainingExamples = getExamplesFromFile(TRAINING_FILE_NAME)
    verificationExamples = getExamplesFromFile(VERIFICATION_FILE_NAME)
    testExamples = getExamplesFromFile(TEST_FILE_NAME)
    perceptron_weights = calculate_perceptron_weights(trainingExamples)
    percent_correct = perceptron_classify(perceptron_weights, verificationExamples)
    print "Perceptron:"
    print "Validation Set: " + str(percent_correct)
    percent_correct = perceptron_classify(perceptron_weights, testExamples)
    print "Test Set: " + str(percent_correct)
    print("Two-layer Neural Net:")
    hidden_weights, output_weights = stochastic_backpropagation(trainingExamples, ALPHA, N_HIDDEN)
    percent_correct = neural_net_classify(hidden_weights, output_weights, verificationExamples)
    print "Validation Set: " + str(percent_correct)
    percent_correct = neural_net_classify(hidden_weights, output_weights, testExamples)
    print "Test Set: " + str(percent_correct)

main()
