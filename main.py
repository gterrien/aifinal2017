import numpy as np
import math
import csv
import random
import operator

#These constants can be changed to switch the data sets used to train, tune, and
#test a perceptron or two-layer neural network.
TRAINING_FILE_NAME = "Data Set/trainingDataGeneric.csv"
VERIFICATION_FILE_NAME = "Data Set/verificationDataGeneric.csv"
TEST_FILE_NAME = "Data Set/testDataGeneric.csv"

#These constants can be changed to alter the learning rate
#and number of hidden nodes of a two-layer neural network.
ALPHA = 0.05
N_HIDDEN = 14


#Calculates and returns the output of the sigmoid specified by
#weights, with features as the input.
def calculate_sigmoid_output(weights, features):
    sumScore = np.dot(weights, features)
    return float(1) / (1 + math.e**sumScore)

#Calculates and returns the output of the perceptron defined by
#weights, with features as the input.
def calculate_perceptron_output(weights, features):
    sumScore = np.dot(weights, features)
    output = 1.0
    if sumScore < 0.0:
        output = -1.0
    return output

#Given a single training example and starting perceptron weights,
#update and return the weights.
def update_perceptron_weights(oldWeights, trainingExample):
    targetValue = trainingExample[1]
    newWeights = map(operator.add, oldWeights, np.dot(targetValue, trainingExample[0]))
    return newWeights


#Reads filename and returns all training or test instances from that file.
def getExamplesFromFile(fileName):
    with open(fileName, 'rU') as csvfile:
        trainingExamples = []
        reader = csv.reader(csvfile)
        for row in reader:
            example = (map(int, row[1:]),float(row[0]))
            trainingExamples.append(example)
        return trainingExamples

#Propagates the feature values through the two-layer neural network
#specified by hiddenWeights and outputWeights, and returns
#both the outputs of the hidden layer and the outputs of the perceptron.
def forwardPropagate(hiddenWeights, outputWeights, features):
    hiddenOutputs = []
    for hiddenWeight in hiddenWeights:
        hiddenOutputs.append(calculate_sigmoid_output(hiddenWeight,features))
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

        target_value = trainingExample[1]
        output_error = output * (target_value - output) * (target_value - output)

        #  Calculate error for hidden nodes
        hiddenErrors = []
        for i in range(n_hidden):
            sumOutputError = float(0)
            sumOutputError = output_weights[i] * output_error
            hiddenError = hiddenOutputs[i] * (1 - hiddenOutputs[i]) * sumOutputError
            hiddenErrors.append(hiddenError)

        # Update weights for output node if necessary
        if output != target_value:
            output_weights = map(operator.add, output_weights, np.dot(target_value, hiddenOutputs))

        # Update weights for hidden nodes
        for j in range(n_hidden):
            for i in range(len(features)):
                delta = alpha * hiddenErrors[j] * features[i]
                hidden_weights[j][i] += delta

        alpha -= alphaDecrementInterval

    return hidden_weights, output_weights

#Calculates and returns the weights for a perceptron
#trained on trainingExamples.
def calculate_perceptron_weights(trainingExamples):
    weights = [0.0] * len(trainingExamples[0][0])
    for example in trainingExamples:
        target_value = example[1]
        output = calculate_perceptron_output(weights, example[0])
        if output != target_value:
            weights = np.array(weights) + np.dot(target_value, example[0])
    return weights

#Classifies all instances withing testExamples using a perceptron specified
#by weights, and returns the percent of instances correctly classified.
def perceptron_classify(weights, testExamples):
    num = 0
    num_correct = 0
    for example in testExamples:
        output = calculate_perceptron_output(weights, example[0])
        targetValue = example[1]
        if targetValue == output:
            num_correct += 1
        num += 1
    percent_correct = float(num_correct) / num
    return percent_correct

#Classfies all instances within testExamples using the two-layer neural network specified
#by hidden_weights and output_weights, and returns the percent of instances correctly classified.
def neural_net_classify(hidden_weights, output_weights, testExamples):
    num_correct = 0
    num = 0
    for example in testExamples:
        hiddenO, output = forwardPropagate(hidden_weights, output_weights, example[0])
        targetValue = example[1]
        if targetValue == output:
            num_correct += 1
        num += 1
    percent_correct = float(num_correct) / num
    return percent_correct

#Iterates through a range of values for both the number of hidden states and the learning rate
#of a two-layer neural network, and prints the results of applying each of these neural networks
#to the validation set along with their hyper-parameters to a text file. One of the top performing
#neural networks will also be printed to the bottom of the file. Depending on the range of values allowed
#for both the learning rate and the number of hidden nodes, this function require a significant amount
#of time(i.e. hours) to finish running.
def tune():
    max_correct = 0.0
    max_alpha = 0.0
    max_hidden = 0
    trainingExamples = getExamplesFromFile(TRAINING_FILE_NAME)
    verificationExamples = getExamplesFromFile(VERIFICATION_FILE_NAME)
    resultsFile = open("results.txt", 'w')
    for n_hidden in range(3, 30):
        for alpha in np.arange(.01, .20, .01):
            hidden_weights, output_weights = stochastic_backpropagation(trainingExamples, alpha, n_hidden)
            num = 0
            num_correct = 0
            for example in verificationExamples:
                hiddenO, output = forwardPropagate(hidden_weights, output_weights, example[0])
                targetValue = example[1]
                if targetValue == output:
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

#Trains a perceptron and a two-layer neural network specified by
#N_HIDDEN and ALPHA on trainingExamples before appyling both neural networks
#to both the validation set and the test set, and printing the results.
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
