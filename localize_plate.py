import cv2
import numpy as np

from localize2 import localize2


def localize(inputImage):
    if inputImage.shape[2] != 3:
        return None

    hsv = cv2.cvtColor(inputImage, cv2.COLOR_BGR2HSV)

    # define range of yellow color in HSV
    low_yellow = np.array([15, 100, 100])
    high_yellow = np.array([36, 255, 255])
    yellow_mask = cv2.inRange(hsv, low_yellow, high_yellow)

    kernel_noise = np.ones((4, 4), np.uint8)  # to delete small noises
    kernel_dilate = np.ones((35, 35), np.uint8)  # bigger kernel to fill holes after ropes
    kernel_erode = np.ones((20, 20),
                           np.uint8)  # bigger kernel to delete pixels on edge that was add after dilate function

    imgErode = cv2.erode(yellow_mask, kernel_noise, 1)
    imgDilate = cv2.dilate(imgErode, kernel_dilate, 1)
    imgErode = cv2.erode(imgDilate, kernel_erode, 1)

    yellow = cv2.bitwise_and(inputImage, inputImage, mask=imgErode)
    plate = np.where(np.logical_and(yellow >= (15, 100, 100), yellow <= (36, 255, 255)))

    # and -> or
    if plate[0].size == 0 and plate[1].size == 0:
        return None

    x = np.unique(plate[0])

    count = 0
    maxX = 0
    minX = 0
    for i in range(0, x.size - 1):
        if x[i] + 1 == x[i + 1]:
            count = count + 1
        elif maxX - minX < count:
            maxX = x[i]
            minX = x[i] - count
            count = 0
        else:
            count = 0

    if minX == 0 and maxX == 0:
        minX = x[0]
        maxX = x[x.size - 1]

    croppedX = yellow[minX:maxX, :]

    plate = np.where(np.logical_and(croppedX >= (15, 100, 100), croppedX <= (36, 255, 255)))
    y = np.unique(plate[1])

    count = 0
    maxY = 0
    minY = 0
    for i in range(0, y.size - 1):
        if y[i] + 1 == y[i + 1]:
            count = count + 1
        elif maxY - minY < count:
            maxY = y[i]
            minY = y[i] - count
            count = 0
        else:
            count = 0

    if minY == 0 and maxY == 0:
        minY = y[0]
        maxY = y[y.size - 1]

    croppedXY = inputImage[minX:maxX, minY:maxY]

    return localize2(croppedXY)


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
