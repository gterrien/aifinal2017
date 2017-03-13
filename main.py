import numpy as np
import math
import csv
import random
import operator

TRAINING_FILE_NAME = "Data Set/trainingData.csv"
VERIFICATION_FILE_NAME = "Data Set/verificationData.csv"
TEST_FILE_NAME = "Data Set/testData.csv"

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
    targetValue = 1.0
    if trainingExample[0] == 'p':
        targetValue = -1.0
    newWeights = map(operator.add, oldWeights, np.dot(targetValue, trainingExample[0]))
    return newWeights


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
        outputs.append(calculate_perceptron_output(outputWeight,hiddenOutputs))
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
            output_weight.append(0.0)
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
        target_value = 1.0
        if trainingExample[1] == 'p':
            target_value = -1.0
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
        '''for j in range(len(output_weights)):
            for i in range(n_hidden):
                delta = alpha * outputErrors[j] * hiddenOutputs[i]
                output_weights[j][i] += delta'''
        for j in range(len(outputs)):
            if outputs[j] != target_value:
                output_weights[j] = map(operator.add, output_weights[j], np.dot(target_value, hiddenOutputs))
                #output_weights = update_perceptron_weights(output_weights, trainingExample)

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

def perceptron_classify(trainingExamples, validationExamples, testExamples):
    print "Perceptron classification:"
    weights = [0.0] * len(trainingExamples[0][0])
    for trainingExample in trainingExamples:
        targetValue = 1.0
        if trainingExample[1] == 'p':
            targetValue = -1.0
        output = calculate_perceptron_output(weights, trainingExample[0])
        if output != targetValue:
            weights = np.array(weights) + np.dot(targetValue, trainingExample[0])
    num = 0
    num_correct = 0
    for example in validationExamples:
        output = calculate_perceptron_output(weights, example[0])
        targetValue = 1.0
        if example[1] == 'p':
            targetValue = -1.0
        if targetValue == output:
            num_correct += 1
        num += 1
        #print str(output) + " , " + example[1]
    percent_correct = float(num_correct) / num
    print "Validation Set: " + str(percent_correct)
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
        #print str(output) + " , " + example[1]
    percent_correct = float(num_correct) / num
    print "Test Set: " + str(percent_correct)

def neural_net_classify(alpha, n_hidden, trainingExamples, verificationExamples, testExamples):
    print "Two-layer neural network classification:"
    hidden_weights, output_weights = stochastic_backpropagation(trainingExamples, alpha, n_hidden)
    num_correct = 0
    num = 0
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
    percent_correct =  str(float(num_correct) / num)
    print "Validation set: " + percent_correct

def main():
    trainingExamples = getExamplesFromFile(TRAINING_FILE_NAME)
    verificationExamples = getExamplesFromFile(VERIFICATION_FILE_NAME)
    testExamples = getExamplesFromFile(TEST_FILE_NAME)
    #hidden_weights, output_weights = stochastic_backpropagation(trainingExamples, .05, 1)

    perceptron_classify(trainingExamples, verificationExamples, testExamples)
    neural_net_classify(.05, 14, trainingExamples, verificationExamples, testExamples)
    '''num_correct = 0
    num = 0
    for example in verificationExamples:
        hiddenO, output = forwardPropagate(hidden_weights, output_weights, example[0])
        print "Output " + str(output)
        print "Type " + example[1]

        classification = output[0]
        if classification == 1.0:
            if example[1] == 'e':
                num_correct += 1
        elif classification == -1.0:
            if example[1] == 'p':
                num_correct += 1
        num += 1
    print str(float(num_correct) / num)'''
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
main()
#main()
