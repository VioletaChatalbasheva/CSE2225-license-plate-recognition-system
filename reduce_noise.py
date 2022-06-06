import numpy as np
import cv2


def clean_noise(character):

    whiteCount = 0
    for i in range(character[:, 0].size):
        for j in range(character[0].size):
            if character[i, j] == 255:
                whiteCount = whiteCount + 1

    marker = character.copy()
    marker[1:-1, :] = 0

    kernel = np.ones((3, 3), np.uint8)

    while True:
        tmp = marker.copy()
        marker = cv2.dilate(marker, kernel)
        marker = cv2.min(character, marker)
        difference = cv2.subtract(marker, tmp)
        if cv2.countNonZero(difference) == 0:
            break

    mask1 = cv2.bitwise_not(marker)
    mask2 = cv2.bitwise_and(character, mask1)

    white = np.where(mask2 == 255)

    if white[0].size == 0 and white[1].size == 0:
        return None

    x = np.unique(white[0])

    if x.size == 0:
        return None

    count = 0
    maxX = 0
    minX = 0
    index = 0
    for i in range(0, x.size - 1):
        if x[i] + 1 == x[i + 1]:
            count = count + 1
        elif maxX - minX < count:
            maxX = x[i]
            minX = x[i] - count
            count = 0
        else:
            index = i + 1
            count = 0

    if minX == 0 and maxX == 0:
        minX = x[index]
        maxX = x[x.size - 1]

    if maxX - minX < count:
        maxX = x[x.size - 1]
        minX = x[x.size - 1] - count

    for k in range(mask2[:, 0].size):
        if k < minX or k > maxX:
            mask2[k, :] = 0

    if whiteCount > 700:
        mask2 = mask2[minX:maxX, :]

    white = np.where(mask2 == 255)
    y = np.unique(white[1])

    if y.size == 0:
        return None

    count = 0
    maxY = 0
    minY = 0
    index = 0
    for i in range(0, y.size - 1):
        if y[i] + 1 == y[i + 1]:
            count = count + 1
        elif maxY - minY < count:
            maxY = y[i]
            minY = y[i] - count
            count = 0
        else:
            index = i + 1
            count = 0

    if minY == 0 and maxY == 0:
        minY = y[index]
        maxY = y[y.size - 1]

    if maxY - minY < count:
        maxY = y[y.size - 1]
        minY = y[y.size - 1] - count

    for j in range(mask2[0].size):
        if j < minY or j > maxY:
            mask2[:, j] = 0

    if whiteCount > 700:
        mask2 = mask2[:, minY:maxY]

    if mask2 is None or mask2.shape[0] == 0 or mask2.shape[1] == 0:
        return None

    return cv2.resize(mask2, (30, 60))
