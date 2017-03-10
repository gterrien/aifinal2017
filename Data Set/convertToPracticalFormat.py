# The data is in a format with multiple-value discrete domains in the attributes. We expand this so that all attributes
# are either 0 or 1 so that we can use all of the information as an input for the neural network easily.

import csv
with open('agaricus-lepiota.data.txt', 'rU') as csvfile:
        reader = csv.reader(csvfile)
        newCSVString = ""
        for row in reader:
            binaryString = ""
            binaryString += row[0] + ','
            # ************
            capShape = row[1]
            capShapeList = [0]*6 # 6 different values for this attribute
            if capShape == 'b':
                capShapeList[0] = 1
            if capShape == 'c':
                capShapeList[1] = 1
            if capShape == 'x':
                capShapeList[2] = 1
            if capShape == 'f':
                capShapeList[3] = 1
            if capShape == 'k':
                capShapeList[4] = 1
            if capShape == 's':
                capShapeList[5] = 1
            capShapeString = ""
            for num in capShapeList:
                capShapeString += str(num) + ','
            binaryString += capShapeString
            # **************
            capSurface = row[2]
            capSurfaceList = [0]*4
            if capSurface == 'f':
                capSurfaceList[0] = 1
            if capSurface == 'g':
                capSurfaceList[1] = 1
            if capSurface == 'y':
                capSurfaceList[2] = 1
            if capSurface == 's':
                capSurfaceList[3] = 1
            capSurfaceString = ""
            for num in capSurfaceList:
                capSurfaceString += str(num) + ','
            binaryString += capSurfaceString
            # *****************
            capColor = row[3]
            capColorList = [0]*10
            if capColor == 'n':
                capColorList[0] = 1
            if capColor == 'b':
                capColorList[1] = 1
            if capColor == 'c':
                capColorList[2] = 1
            if capColor == 'g':
                capColorList[3] = 1
            if capColor == 'r':
                capColorList[4] = 1
            if capColor == 'p':
                capColorList[5] = 1
            if capColor == 'u':
                capColorList[6] = 1
            if capColor == 'e':
                capColorList[7] = 1
            if capColor == 'w':
                capColorList[8] = 1
            if capColor == 'y':
                capColorList[9] = 1
            capColorString = ""
            for num in capColorList:
                capColorString += str(num) + ','
            binaryString += capColorString
            # ***********************
            bruises = row[4]
            bruisesString = "0,"
            if bruises == 't':
                bruisesString = "1,"
            binaryString += bruisesString
            # *************************
            odor = row[5]
            odorList = [0]*9
            if odor == 'a':
                odorList[0] = 1
            if odor == 'l':
                odorList[1] = 1
            if odor == 'c':
                odorList[2] = 1
            if odor == 'y':
                odorList[3] = 1
            if odor == 'f':
                odorList[4] = 1
            if odor == 'm':
                odorList[5] = 1
            if odor == 'n':
                odorList[6] = 1
            if odor == 'p':
                odorList[7] = 1
            if odor == 's':
                odorList[8] = 1
            odorString = ""
            for num in odorList:
                odorString += str(num) + ','
            binaryString += odorString
            # *************************
            gillAttachment = row[6]
            gillAttachmentList = [0]*4
            if gillAttachment == 'a':
                gillAttachmentList[0] = 1
            if gillAttachment == 'd':
                gillAttachmentList[1] = 1
            if gillAttachment == 'f':
                gillAttachmentList[2] = 1
            if gillAttachment == 'n':
                gillAttachmentList[3] = 1
            gillAttachmentString = ""
            for num in gillAttachmentList:
                gillAttachmentString += str(num) + ','
            binaryString += gillAttachmentString
            # **************************
            gillSpacing = row[7]
            gillSpacingList = [0]*3
            if gillSpacing == 'c':
                gillSpacingList[0] = 1
            if gillSpacing == 'w':
                gillSpacingList[1] = 1
            if gillSpacing == 'd':
                gillSpacingList[2] = 1
            gillSpacingString = ""
            for num in gillSpacingList:
                gillSpacingString += str(num) + ','
            binaryString += gillSpacingString
            # ***********************8
            gillSize = row[8]
            gillSizeList = [0]*2
            if gillSize == 'b':
                gillSizeList[0] = 1
            if gillSize == 'n':
                gillSizeList[1] = 1
            gillSizeString = ""
            for num in gillSizeList:
                gillSizeString += str(num) + ','
            binaryString += gillSizeString
            # ************************
            gillColor = row[9]
            gillColorList = [0]*12
            if gillColor == 'k':
                gillColorList[0] = 1
            if gillColor == 'n':
                gillColorList[1] = 1
            if gillColor == 'b':
                gillColorList[2] = 1
            if gillColor == 'h':
                gillColorList[3] = 1
            if gillColor == 'g':
                gillColorList[4] = 1
            if gillColor == 'r':
                gillColorList[5] = 1
            if gillColor == 'o':
                gillColorList[6] = 1
            if gillColor == 'p':
                gillColorList[7] = 1
            if gillColor == 'u':
                gillColorList[8] = 1
            if gillColor == 'e':
                gillColorList[9] = 1
            if gillColor == 'w':
                gillColorList[10] = 1
            if gillColor == 'y':
                gillColorList[11] = 1
            gillColorString = ""
            for num in gillColorList:
                gillColorString += str(num) + ','
            binaryString += gillColorString
            # **********************************
            stalkShape = row[10]
            stalkShapeList = [0]*2
            if stalkShape == 'e':
                stalkShapeList[0] = 1
            if stalkShape == 't':
                stalkShapeList[1] = 1
            stalkShapeString = ""
            for num in stalkShapeList:
                stalkShapeString += str(num) + ','
            binaryString += stalkShapeString
            # *****************************
            stalkRoot = row[11]
            stalkRootList = [0]*7
            if stalkRoot == 'b':
                stalkRootList[0] = 1
            if stalkRoot == 'c':
                stalkRootList[1] = 1
            if stalkRoot == 'u':
                stalkRootList[2] = 1
            if stalkRoot == 'e':
                stalkRootList[3] = 1
            if stalkRoot == 'z':
                stalkRootList[4] = 1
            if stalkRoot == 'r':
                stalkRootList[5] = 1
            if stalkRoot == '?':
                stalkRootList[6] = 1
            stalkRootString = ""
            for num in stalkRootList:
                stalkRootString = str(num) + ','
            binaryString += stalkRootString
            # **********************
            stalkSurfaceAR = row[12]
            stalkSurfaceARList = [0]*4
            if stalkSurfaceAR == 'f':
                stalkSurfaceARList[0] = 1
            if stalkSurfaceAR == 'y':
                stalkSurfaceARList[1] = 1
            if stalkSurfaceAR == 'k':
                stalkSurfaceARList[2] = 1
            if stalkSurfaceAR == 's':
                stalkSurfaceARList[3] = 1
            stalkSurfaceARString = ""
            for num in stalkSurfaceARList:
                stalkSurfaceARString += str(num) + ','
            binaryString += stalkSurfaceARString
            # ************************
            stalkSurfaceBR = row[13]
            stalkSurfaceBRList = [0]*4
            if stalkSurfaceBR == 'f':
                stalkSurfaceBRList[0] = 1
            if stalkSurfaceBR == 'y':
                stalkSurfaceBRList[1] = 1
            if stalkSurfaceBR == 'k':
                stalkSurfaceBRList[2] = 1
            if stalkSurfaceBR == 's':
                stalkSurfaceBRList[3] = 1
            stalkSurfaceBRString = ""
            for num in stalkSurfaceBRList:
                stalkSurfaceBRString += str(num) + ','
            binaryString += stalkSurfaceBRString
            # ***********************************
            stalkColorAR = row[14]
            stalkColorArList = [0]*9
            if stalkColorAR == 'n':
                stalkColorArList[0] = 1
            if stalkColorAR == 'b':
                stalkColorArList[1] = 1
            if stalkColorAR == 'c':
                stalkColorArList[2] = 1
            if stalkColorAR == 'g':
                stalkColorArList[3] = 1
            if stalkColorAR == 'o':
                stalkColorArList[4] = 1
            if stalkColorAR == 'p':
                stalkColorArList[5] = 1
            if stalkColorAR == 'e':
                stalkColorArList[6] = 1
            if stalkColorAR == 'w':
                stalkColorArList[7] = 1
            if stalkColorAR == 'y':
                stalkColorArList[8] = 1
            stalkColorARString = ""
            for num in stalkColorArList:
                stalkColorARString += str(num) + ','
            binaryString += stalkColorARString
            # ****************************
            stalkColorBR = row[15]
            stalkColorBrList = [0]*9
            if stalkColorBR == 'n':
                stalkColorBrList[0] = 1
            if stalkColorBR == 'b':
                stalkColorBrList[1] = 1
            if stalkColorBR == 'c':
                stalkColorBrList[2] = 1
            if stalkColorBR == 'g':
                stalkColorBrList[3] = 1
            if stalkColorBR == 'o':
                stalkColorBrList[4] = 1
            if stalkColorBR == 'p':
                stalkColorBrList[5] = 1
            if stalkColorBR == 'e':
                stalkColorBrList[6] = 1
            if stalkColorBR == 'w':
                stalkColorBrList[7] = 1
            if stalkColorBR == 'y':
                stalkColorBrList[8] = 1
            stalkColorBRString = ""
            for num in stalkColorBrList:
                stalkColorBRString += str(num) + ','
            binaryString += stalkColorBRString
            # ****************************
            veilType = row[16]
            veilTypeList = [0]*2
            if veilType == 'p':
                veilTypeList[0] = 1
            if veilType == 'u':
                veilTypeList[1] = 1
            veilTypeString = ""
            for num in veilTypeList:
                veilTypeString += str(num) + ','
            binaryString += veilTypeString
            # ****************************
            veilColor = row[17]
            veilColorList = [0]*4
            if veilColor == 'n':
                veilColorList[0] = 1
            if veilColor == 'o':
                veilColorList[1] = 1
            if veilColor == 'w':
                veilColorList[2] = 1
            if veilColor == 'y':
                veilColorList[3] = 1
            veilColorString = ""
            for num in veilColorList:
                veilColorString += str(num) + ','
            binaryString += veilColorString
            # ***********************
            ringNumber = row[18]
            ringNumberList = [0]*3
            if ringNumber == 'n':
                ringNumberList[0] = 1
            if ringNumber == 'o':
                ringNumberList[1] = 1
            if ringNumber == 't':
                ringNumberList[2] = 1
            ringNumberString = ""
            for num in ringNumberList:
                ringNumberString += str(num) + ','
            binaryString += ringNumberString
            # **********************
            ringType = row[19]
            ringTypeList = [0]*8
            if ringType == 'c':
                ringTypeList[0] = 1
            if ringType == 'e':
                ringTypeList[1] = 1
            if ringType == 'f':
                ringTypeList[2] = 1
            if ringType == 'l':
                ringTypeList[3] = 1
            if ringType == 'n':
                ringTypeList[4] = 1
            if ringType == 'p':
                ringTypeList[5] = 1
            if ringType == 's':
                ringTypeList[6] = 1
            if ringType == 'z':
                ringTypeList[7] = 1
            ringTypeString = ""
            for num in ringTypeList:
                ringTypeString += str(num) + ','
            binaryString += ringTypeString
            # **********************
            sporePrintColor = row[20]
            sporePrintColorList = [0]*9
            if sporePrintColor == 'k':
                sporePrintColorList[0] = 1
            if sporePrintColor == 'n':
                sporePrintColorList[1] = 1
            if sporePrintColor == 'b':
                sporePrintColorList[2] = 1
            if sporePrintColor == 'h':
                sporePrintColorList[3] = 1
            if sporePrintColor == 'r':
                sporePrintColorList[4] = 1
            if sporePrintColor == 'o':
                sporePrintColorList[5] = 1
            if sporePrintColor == 'u':
                sporePrintColorList[6] = 1
            if sporePrintColor == 'w':
                sporePrintColorList[7] = 1
            if sporePrintColor == 'y':
                sporePrintColorList[8] = 1
            sporePrintColorString = ""
            for num in sporePrintColorList:
                sporePrintColorString += str(num) + ','
            binaryString += sporePrintColorString
            # ***********************
            population = row[21]
            populationList = [0]*6
            if population == 'a':
                populationList[0] = 1
            if population == 'c':
                populationList[1] = 1
            if population == 'n':
                populationList[2] = 1
            if population == 's':
                populationList[3] = 1
            if population == 'v':
                populationList[4] = 1
            if population == 'y':
                populationList[5] = 1
            populationString = ""
            for num in populationList:
                populationString += str(num) + ','
            binaryString += populationString
            # ***********************
            habitat = row[22]
            habitatList = [0]*7
            if habitat == 'g':
                habitatList[0] = 1
            if habitat == 'l':
                habitatList[1] = 1
            if habitat == 'm':
                habitatList[2] = 1
            if habitat == 'p':
                habitatList[3] = 1
            if habitat == 'u':
                habitatList[4] = 1
            if habitat == 'w':
                habitatList[5] = 1
            if habitat == 'd':
                habitatList[6] = 1
            habitatString = ""
            for num in habitatList:
                habitatString += str(num) + ','
            binaryString += habitatString
            # *******************
            # remove last comma, add new line
            binaryString = binaryString[:-1] + '\n'
            newCSVString += binaryString
        newFile = open("expandedData.csv", 'w')
        newFile.write(newCSVString)
        newFile.close()


