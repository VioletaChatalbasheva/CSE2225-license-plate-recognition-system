import math
import cv2
import os
import numpy as np
from collections import Counter

from reduce_noise import clean_noise


def countWhitePixels(mask):
    whiteCount = 0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] == 255:
                whiteCount = whiteCount + 1
    return whiteCount


# find the column where the amount of black pixel values is high, meaning this is the end of the character
def findEnd(start, black, black_max, width):
    end = start + 1
    for m in range(start + 1, width - 1):
        if black[m] > 0.9 * black_max:
            end = m
            break
    return end


def segmentCharacters(cropped, frameNumber, timestamp, writer, allPlates, frameList, timeList):
    number = [frameNumber]
    time = [timestamp]

    if cropped is None:
        return

    cropped = cv2.resize(cropped, (400, 100))
    img_gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)

    threshPlateImage = cv2.bitwise_not(threshold(otsuThreshold(img_gray), img_gray))

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    mask = cv2.morphologyEx(threshPlateImage, cv2.MORPH_DILATE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel, iterations=1)

    whiteCount = countWhitePixels(mask)

    # white hinders separating characters correctly
    if 14000 < whiteCount < 16000:
        for i in range(5):
            for j in range(mask.shape[1]):
                mask[i, j] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # too much white noise because of bad illumination
    if 16000 < whiteCount < 20000:
        for i in range(7):
            for j in range(mask.shape[1]):
                mask[i, j] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Record the sum of white pixels in each column
    white = []
    # Record the sum of black pixels in each column
    black = []

    height = mask.shape[0]
    width = mask.shape[1]

    for i in range(width):
        countWhite = 0
        countBlack = 0
        for j in range(height):
            if mask[j][i] == 255:
                countWhite = countWhite + 1
            else:
                countBlack = countBlack + 1

        white.append(countWhite)
        black.append(countBlack)

    white_max = max(white)
    black_max = max(black)

    licensePlate = []
    n = 1
    while n < width - 1:
        n = n + 1
        # start from a column with little white
        if white[n] > 0.1 * white_max:
            start = n
            end = findEnd(start, black, black_max, width)
            n = end
            # if the columns with black values are not consecutive
            if end - start > 5:
                character = mask[1:height, start:end]
                if character.size < 1100:
                    continue
                character = clean_noise(character)
                if character is None:
                    continue
                (char, bestMatch) = bitWiseAllLetters(character)
                licensePlate.append(char)

    i = 0
    length = len(licensePlate)
    while length != 0 and i != length - 1:
        if licensePlate[i].isdigit() and licensePlate[i + 1].isalpha():
            licensePlate.insert(i + 1, '-')
            length = length + 1
        elif licensePlate[i].isalpha() and licensePlate[i + 1].isdigit():
            licensePlate.insert(i + 1, '-')
            length = length + 1
        i = i + 1

    if len(licensePlate) < 8:
        licensePlate = []
    elif len(licensePlate) == 8 and (licensePlate.count('-') > 2 or licensePlate[7] == '-' or licensePlate[0] == '-'):
        licensePlate = []
    elif len(licensePlate) > 8 and licensePlate.count('-') > 2:
        licensePlate = []
    # noise in the beginning
    elif licensePlate[0] == '-':
        licensePlate.remove(licensePlate[0])
    # noise in the end
    elif len(licensePlate) > 8 and (licensePlate[0] != '-' or licensePlate.count('-') > 2):
        licensePlate = licensePlate[:8]

    if len(licensePlate) == 8 and licensePlate[7] == '-':
        licensePlate = []

    digit = 0
    letter = 0
    for i in range(len(licensePlate)):
        if licensePlate[i].isdigit():
            digit = digit + 1
        if licensePlate[i].isalpha():
            letter = letter + 1

    if not ((digit == 3 and letter == 3) or (digit == 2 and letter == 4) or (digit == 4 and letter == 2)):
        licensePlate = []

    # print(''.join(licensePlate))
    if len(licensePlate) != 0:
        if len(allPlates) == 0:
            allPlates.append(''.join(licensePlate))
            frameList.append(number)
            timeList.append(time)
        else:
            licensePlate = ''.join(licensePlate)
            if diff_letters(licensePlate, allPlates[len(allPlates) - 1]) > 3:
                finalPlate = most_frequent(allPlates)
                array = np.array(allPlates)
                index = np.where(array == finalPlate)[0]
                fr = frameList[index[0]]
                ts = timeList[index[0]]
                writer.writerow({'License plate': "'" + str(finalPlate) + "'", 'Frame no.': fr[0],
                                 'Timestamp(seconds)': ts[0]})
                allPlates.clear()
                frameList.clear()
                timeList.clear()
            allPlates.append(licensePlate)
            frameList.append(number)
            timeList.append(time)


def comparisonViaSSIM(image1, image2):
    image2 = cv2.resize(image2, (30, 60))
    nu_x = np.average(image1)
    nu_y = np.average(image2)
    var_x = np.var(image1)
    var_y = np.var(image2)
    covar = np.sum((image1 - np.mean(image1)) * (image2 - np.mean(image2))) / (len(image1) - 1)
    c1 = np.square(0.01 * 255)
    c2 = np.square(0.03 * 255)
    SSIM = ((2 * nu_x * nu_y + c1) * (2 * covar + c2)) / ((np.square(nu_x) + np.square(nu_y) + c1) * (var_x + var_y + c2))
    return SSIM


def calculate_psnr(img1, img2):
    img2 = cv2.resize(img2, (30, 60))
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def most_frequent(List):
    occurrence_count = Counter(List)
    return occurrence_count.most_common(1)[0][0]


def diff_letters(a, b):
    return sum(a[i] != b[i] for i in range(len(a)))


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), 0)
        # if img is not None:
        images.append(img)
    return images


def bitwiseComparison(image1, image2):
    image2 = cv2.resize(image2, (30, 60))
    matched = 0
    for i in range(0, len(image1)):
        for j in range(0, len(image1[i])):
            if image1[i][j] == image2[i][j]:
                matched = matched + 1
    return matched


def checkChar(image, path, bestMatch, char):
    images = load_images_from_folder('./newData/' + str(path))
    for img in images:
        matched = calculate_psnr(image, img)
        # matched = mse(image, img)
        if matched > bestMatch:
            char = path
            bestMatch = matched
    return bestMatch, char


def bitWiseAllLetters(image):
    bestMatch = 0
    char = ''
    chars = ['-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'B',
             'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'X', 'Z']
    for character in chars:
        (bestMatch, char) = checkChar(image, character, bestMatch, char)
    return char, bestMatch


def threshold(otsu, image):
    result = np.zeros(image.shape, dtype=np.uint8)

    strong_row, strong_col = np.where(image >= otsu)
    weak_row, weak_col = np.where(image < otsu)

    result[strong_row, strong_col] = 255
    result[weak_row, weak_col] = 0

    return result


def otsuThreshold(grayPlate):
    # Set total number of bins in the histogram
    bins_num = 256

    # Get the image histogram
    hist, bin_edges = np.histogram(grayPlate, bins=bins_num)

    # Get normalized histogram if it is required
    # if is_normalized:
    #     hist = np.divide(hist.ravel(), hist.max())

    # Calculate centers of bins
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.

    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    # Get the class means mu0(t)
    mean1 = np.cumsum(hist * bin_mids) / weight1
    # Get the class means mu1(t)
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]
    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    # Maximize the inter_class_variance function val
    index_of_max_val = np.argmax(inter_class_variance)
    threshold = bin_mids[:-1][index_of_max_val]

    return int(threshold)
