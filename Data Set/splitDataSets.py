# splitDataSets.py
#
# Creates files for the data set with the following proportions, randomly assigned:
# 70% - trainingData.csv
# 15% - verificationData.csv
# 15% - testData.csv

import csv
import random

dataItems = []
with open('expandedData.csv', 'rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        dataItems.append(row)

random.shuffle(dataItems) # randomize ordering to ensure good distribution

upperBoundTraining = int(len(dataItems)*0.7)
totalLeft = len(dataItems)-upperBoundTraining
upperBoundVerification = upperBoundTraining + totalLeft/2

trainingItems = []
verificationItems = []
testItems = []
for i in range(upperBoundTraining):
    trainingItems.append(dataItems[i])

for i in range(upperBoundTraining, upperBoundVerification):
    verificationItems.append(dataItems[i])

for i in range(upperBoundVerification, len(dataItems)):
    testItems.append(dataItems[i])

trainingString = ""
for item in trainingItems:
    itemString = ""
    for attribute in item:
       itemString += attribute + ','
    itemString = itemString[:-1] + '\n'
    trainingString += itemString

trainingFile = open('trainingData.csv', 'w')
trainingFile.write(trainingString)
trainingFile.close()

verificationString = ""
for item in verificationItems:
    itemString = ""
    for attribute in item:
       itemString += attribute + ','
    itemString = itemString[:-1] + '\n'
    verificationString += itemString

verificationFile = open('verificationData.csv', 'w')
verificationFile.write(verificationString)
verificationFile.close()

testString = ""
for item in testItems:
    itemString = ""
    for attribute in item:
       itemString += attribute + ','
    itemString = itemString[:-1] + '\n'
    testString += itemString

testFile = open('testData.csv', 'w')
testFile.write(testString)
testFile.close()
