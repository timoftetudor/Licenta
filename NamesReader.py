from random import randint

import cv2
import cv2 as cv
from fuzzywuzzy import fuzz
from matplotlib import pyplot as plt
import numpy as np
import time
from difflib import SequenceMatcher


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


start_time = time.time()

np.set_printoptions(threshold=np.inf)

file_names = 'test.png'

img = cv.imread(file_names)
# Change the picture to a certain color space, in this case to the grey color space
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# Now we split the image to 400 cells, each 50x50 size
cells = [np.hsplit(row, 20) for row in np.vsplit(gray, 20)]
numberPixelsApparitions = 50
# Make it into a Numpy array. It size will be (20,20,50,50)
x = np.array(cells)


def obtain_np_data(fileName):
    img = cv.imread(fileName)
    # Change the picture to a certain color space, in this case to the grey color space
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Now we split the image to 400 cells, each 50x50 size
    cells = [np.hsplit(row, 20) for row in np.vsplit(gray, 20)]
    # Make it into a Numpy array. It size will be (20,20,50,50)
    return np.array(cells)


def change_values(fileName):
    for indexI in range(0, 20):
        for indexJ in range(0, 20):
            letter = fileName[indexI][indexJ]
            for i in range(0, 50):
                for j in range(0, 50):
                    if letter[i][j] > 150:
                        letter[i][j] = 1
                    else:
                        letter[i][j] = 0


change_values(x)


def print_letter(letter):
    for i in range(0, 50):
        temp_array = []
        for j in range(0, 50):
            temp_array.append(letter[i][j])
        print temp_array


def compare(value1, value2):
    numbersIdentical = 0
    for idx1 in range(0, 50):
        for idx2 in range(0, 50):
            if value1[idx1][idx2] == value2[idx1][idx2] and value1[idx1][idx2] == 1 and value2[idx1][idx2] == 1:
                numbersIdentical += 1

    if np.count_nonzero(value1) == numbersIdentical:
        return 100
    else:
        if np.count_nonzero(value1) > numbersIdentical:
            arithmeticalValue = (numbersIdentical * 100) / np.count_nonzero(value1)
        else:
            arithmeticalValue = (numbersIdentical * 100) / np.count_nonzero(value2)
        return arithmeticalValue


namePositions = []


def read_names():
    lastWasEmpty = False
    for i in range(0, 20):
        # change_values(x[i][0])
        numberOfZero = 0
        if i < 19:
            numberOfZero = 2500 - np.count_nonzero(x[i][0]) + 99
        else:
            if i == 19:
                numberOfZero = 2500 - np.count_nonzero(x[i][0]) + 148

        if numberOfZero < 2500:
            namePositions.append(0)
            lastWasEmpty = False
        else:
            break

        for j in range(1, 20):
            # change_values(x[i][j])
            numberOfZero = 0

            if i < 19 and j < 19:
                numberOfZero = 2500 - np.count_nonzero(x[i][j]) + 99
            else:
                if i == 19 and j < 19 or i < 19 and j == 19:
                    numberOfZero = 2500 - np.count_nonzero(x[i][j]) + 148
                else:
                    if i == 19 and j == 19:
                        numberOfZero = 2500 - np.count_nonzero(x[i][j]) + 196

            if lastWasEmpty and numberOfZero < 2500:
                namePositions.append(j)
                lastWasEmpty = False

            if numberOfZero == 2500:
                if not lastWasEmpty:
                    namePositions.append(j - 1)
                    lastWasEmpty = True
                else:
                    break
            if j == 20:
                namePositions.append(j)
    print 'namePositions =', namePositions


read_names()
word_lengths = []


def get_word_lengths(namePositions):
    lineCount = -1
    for index in range(0, len(namePositions), 2):
        i = namePositions[index]
        j = namePositions[index + 1]

        if namePositions[index] == 0:
            lineCount += 1

        word_lengths.append(j - i + 1)
        # for letterPosition in range(i, j + 1):
        #     print_letter(x[lineCount][letterPosition])
    print 'word-length =', word_lengths


get_word_lengths(namePositions)

testLetters = []
letters = []


def get_letters(namePositions, word_length):
    global possibleLetterLine, possibleLetterCol

    def letter_range(start, stop, step=1):
        """Yield a range of lowercase letters."""
        for ord_ in range(ord(start.lower()), ord(stop.lower()), step):
            yield chr(ord_)

    alphabet = list(letter_range("A", "Z"))

    abcd = obtain_np_data('ABCD.png')
    efgh = obtain_np_data('EFGH.png')
    ijkl = obtain_np_data('IJKL.png')
    mnop = obtain_np_data('MNOP.png')
    qrst = obtain_np_data('QRST.png')
    uvwx = obtain_np_data('UVWX.png')
    yz = obtain_np_data('YZ.png')
    change_values(abcd)
    change_values(efgh)
    change_values(ijkl)
    change_values(mnop)
    change_values(qrst)
    change_values(uvwx)
    # change_values(yz)

    lineCount = -1
    for index in range(0, len(namePositions), 2):
        i = namePositions[index]
        j = namePositions[index + 1]
        # testLetters.append('')
        if namePositions[index] == 0:
            lineCount += 1

        changedValuesABCD = False
        changedValuesEFGH = False
        changedValuesIJKL = False
        changedValuesMNOP = False
        changedValuesQRST = False
        changedValuesUVWX = False
        # changedValuesYZ = False

        for letterPosition in range(i, j + 1):
            maxSimilarity = 0
            letterValues = [0] * 26
            for line in range(0, 20):
                for col in range(0, 5):
                    letterToCompare = abcd[line][col]
                    resultCompare = compare(letterToCompare, x[lineCount][letterPosition])
                    if maxSimilarity < resultCompare:
                        maxSimilarity = resultCompare
                        possibleLetterLine = line
            letterValues[possibleLetterLine / 5] = maxSimilarity
            maxSimilarity = 0
            for line in range(0, 20):
                for col in range(0, 5):
                    letterToCompare = efgh[line][col]
                    resultCompare = compare(letterToCompare, x[lineCount][letterPosition])
                    if maxSimilarity < resultCompare:
                        maxSimilarity = resultCompare
                        possibleLetterLine = line
            letterValues[possibleLetterLine / 5 + 4] = maxSimilarity
            maxSimilarity = 0
            for line in range(0, 20):
                for col in range(0, 5):
                    letterToCompare = ijkl[line][col]
                    resultCompare = compare(letterToCompare, x[lineCount][letterPosition])
                    if maxSimilarity < resultCompare:
                        maxSimilarity = resultCompare
                        possibleLetterLine = line
            letterValues[possibleLetterLine / 5 + 8] = maxSimilarity
            maxSimilarity = 0
            for line in range(0, 20):
                for col in range(0, 5):
                    letterToCompare = mnop[line][col]
                    resultCompare = compare(letterToCompare, x[lineCount][letterPosition])
                    if maxSimilarity < resultCompare:
                        maxSimilarity = resultCompare
                        possibleLetterLine = line
            letterValues[possibleLetterLine / 5 + 12] = maxSimilarity
            maxSimilarity = 0
            for line in range(0, 20):
                for col in range(0, 5):
                    letterToCompare = qrst[line][col]
                    resultCompare = compare(letterToCompare, x[lineCount][letterPosition])
                    if maxSimilarity < resultCompare:
                        maxSimilarity = resultCompare
                        possibleLetterLine = line
            letterValues[possibleLetterLine / 5 + 16] = maxSimilarity
            maxSimilarity = 0
            for line in range(0, 10):
                for col in range(0, 5):
                    letterToCompare = uvwx[line][col]
                    resultCompare = compare(letterToCompare, x[lineCount][letterPosition])
                    if maxSimilarity < resultCompare:
                        maxSimilarity = resultCompare
                        possibleLetterLine = line
            letterValues[possibleLetterLine / 5 + 20] = maxSimilarity
            maxSimilarity = 0
            for line in range(0, 10):
                for col in range(0, 5):
                    letterToCompare = uvwx[line][col] # his will be YZ, the file is not saved correctly
                    resultCompare = compare(letterToCompare, x[lineCount][letterPosition])
                    if maxSimilarity < resultCompare:
                        maxSimilarity = resultCompare
                        possibleLetterLine = line
            letterValues[possibleLetterLine / 5 + 20] = maxSimilarity

            maxSimilarity = 0
            for pos in range(0, 26):
                if letterValues[pos] == 100:
                    letterValues[pos] = randint(93, 99)
                if maxSimilarity < letterValues[pos]:
                    maxSimilarity = letterValues[pos]
                    posMaxSimilarity = pos
            letters.append(chr(posMaxSimilarity + 65))
            print letterValues
        print letters


get_letters(namePositions, word_lengths)
print("--- %s seconds ---" % (time.time() - start_time))

firstNames = []


def read_names(fileName):
    with open(fileName, "r") as fileNames:
        names = fileNames.readlines()
    names = [line.rstrip('\n') for line in open(fileName)]
    for i in range(0, len(names)):
        names[i] = names[i].upper()
    return names


firstNames = read_names("FirstNames.txt")

found_names = []


def create_names(letters, word_lengths):
    position = 0
    for i in range(0, len(word_lengths)):
        word = ''
        for j in range(position, position + word_lengths[i]):
            word += letters[j]
        found_names.append(word)
        position += word_lengths[i]


create_names(letters, word_lengths)

# found_names = ['IUDOR', 'IOAUA']


def search_names(found_names, firstNames):
    text_file = open("Output.txt", "w")
    for i in range(0, len(found_names)):
        possibleName = ''
        maxRatio = 0
        for j in range(0, len(firstNames)):
            countSameLetters = 0
            if len(found_names[i]) == len(firstNames[j]):
                for letter in range(0, len(found_names[i])):
                    if found_names[i][letter] == firstNames[j][letter]:
                        countSameLetters += 1
                if countSameLetters > maxRatio:
                    maxRatio = countSameLetters
                    possibleName = firstNames[j]
        print possibleName
        text_file.write(possibleName)
        text_file.write('\n')
    text_file.close()


search_names(found_names, firstNames)
