# Change 'e' and 'p' to 1 and -1 in our data sets, respectively. This makes it a generic format so that our classification
# methods can be used on other data sets.

import csv
files = ['testData.csv', 'trainingData.csv', 'verificationData.csv']

for file in files:
    with open(file, 'rU') as csvfile:
        reader = csv.reader(csvfile)
        newfileString = ""
        for row in reader:
            rowString = ""
            for item in row:
                if item == 'e':
                    rowString += '1' + ','
                elif item == 'p':
                    rowString += '-1' + ','
                else:
                    rowString += item + ','
            rowString = rowString[:-1] + '\n'
            newfileString += rowString

        newfile = open(file[:-4] + 'Generic.csv', 'w')
        newfile.write(newfileString)
        newfile.close()
