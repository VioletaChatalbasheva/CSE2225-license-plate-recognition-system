import numpy as np
import cv2


def inside_rect(rect, num_cols, num_rows):
    rect_center = rect[0]
    rect_center_x = rect_center[0]
    rect_center_y = rect_center[1]

    if (rect_center_x < 0) or (rect_center_x > num_cols):
        return False
    if (rect_center_y < 0) or (rect_center_y > num_rows):
        return False

    # https://docs.opencv.org/3.0-beta/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html
    box = cv2.boxPoints(rect)

    x_max = int(np.max(box[:, 0]))
    x_min = int(np.min(box[:, 0]))
    y_max = int(np.max(box[:, 1]))
    y_min = int(np.min(box[:, 1]))

    if (x_max <= num_cols) and (x_min >= 0) and (y_max <= num_rows) and (y_min >= 0):
        return True
    else:
        return False


def rect_bbx(rect):
    box = cv2.boxPoints(rect)
    x_max = int(np.max(box[:, 0]))
    x_min = int(np.min(box[:, 0]))
    y_max = int(np.max(box[:, 1]))
    y_min = int(np.min(box[:, 1]))

    center = (int((x_min + x_max) // 2), int((y_min + y_max) // 2))
    width = int(x_max - x_min)
    height = int(y_max - y_min)
    angle = 0
    return (center, (width, height), angle)


def crop_rectangle(image, rect):
    num_rows = image.shape[0]
    num_cols = image.shape[1]

    if not inside_rect(rect=rect, num_cols=num_cols, num_rows=num_rows):
        # print("Proposed rectangle is not fully in the image.")
        return None
    rect_center = rect[0]
    rect_center_x = rect_center[0]
    rect_center_y = rect_center[1]
    rect_width = rect[1][0]
    rect_height = rect[1][1]
    return image[rect_center_y - rect_height // 2:rect_center_y + rect_height - rect_height // 2, rect_center_x - rect_width // 2:rect_center_x + rect_width - rect_width // 2]


def image_rotate_without_crop(mat, angle):
    # https://stackoverflow.com/questions/22041699/rotate-an-image-without-cropping-in-opencv-in-c
    # angle in degrees

    height, width = mat.shape[:2]
    image_center = (width / 2, height / 2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)

    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))

    return rotated_mat


def crop_rotated_rectangle(image, rect):
    # Crop a rotated rectangle from a image

    num_rows = image.shape[0]
    num_cols = image.shape[1]

    if not inside_rect(rect=rect, num_cols=num_cols, num_rows=num_rows):
        # print("Proposed rectangle is not fully in the image.")
        return image

    rotated_angle = rect[2]

    rect_bbx_upright = rect_bbx(rect=rect)
    rect_bbx_upright_image = crop_rectangle(image=image, rect=rect_bbx_upright)
    if rect_bbx_upright_image is None:
        return image
    rotated_rect_bbx_upright_image = image_rotate_without_crop(mat=rect_bbx_upright_image, angle=rotated_angle)
    rect_width = int(rect[1][0])
    rect_height = int(rect[1][1])

    crop_center = (rotated_rect_bbx_upright_image.shape[1] // 2, rotated_rect_bbx_upright_image.shape[0] // 2)
    result = rotated_rect_bbx_upright_image[
             crop_center[1] - rect_height // 2: crop_center[1] + (rect_height - rect_height // 2),
             crop_center[0] - rect_width // 2: crop_center[0] + (rect_width - rect_width // 2)]
    if result.shape[0] <= result.shape[1]:
        return result
    else:
        return image_rotate_without_crop(result, 270)


def localize2(cropped):
    # if cropped is None or cropped.shape[0] == 0 or cropped.shape[1] == 0:
    #     return None
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

    # define range of yellow color in HSV
    low_yellow = np.array([15, 100, 100])
    high_yellow = np.array([36, 255, 255])
    yellow_mask = cv2.inRange(hsv, low_yellow, high_yellow)

    yellow = cv2.bitwise_and(cropped, cropped, mask=yellow_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    yellow = cv2.dilate(yellow, kernel, iterations=1)

    height = yellow.shape[0]
    width = yellow.shape[1]

    listMinY = []
    listMaxY = []
    for i in range(0, height):
        yellowArray = np.array(np.where(np.logical_and(yellow[i, :] >= (15, 100, 100), yellow[i, :] <= (36, 255, 255))))
        yellowRow = yellowArray[0]
        if yellowRow.size != 0:
            min1 = np.min(yellowRow)
            max1 = np.max(yellowRow)
            listMinY.append([i, min1])
            listMaxY.append([i, max1])

    if len(listMinY) == 0 or len(listMaxY) == 0:
        return None

    minArray = np.array(listMinY)

    minArray = minArray[minArray[:, 0].argsort()]
    min = np.min(minArray[:, 1])
    index = np.where(minArray[:, 1] == min)
    minX = minArray[index]
    topLeft = minArray[np.where(minArray[:, 1] == min) and np.where(minArray[:, 0] == np.min(minX[:, 0]))]
    topLeft = [[topLeft[0, 1], topLeft[0, 0]]]
    topLeft = np.array(topLeft)

    maxArray = np.array(listMaxY)
    maxArray = maxArray[maxArray[:, 0].argsort()]
    max = np.max(maxArray[:, 1])
    index = np.where(maxArray[:, 1] == max)
    maxX = maxArray[index]
    bottomRight = maxArray[np.where(maxArray[:, 1] == max) and np.where(maxArray[:, 0] == np.max(maxX[:, 0]))]
    bottomRight = [[bottomRight[0, 1], bottomRight[0, 0]]]
    bottomRight = np.array(bottomRight)

    listMinX = []
    listMaxX = []
    for i in range(0, width):
        yellowArray = np.array(np.where(np.logical_and(yellow[:, i] >= (15, 100, 100), yellow[:, i] <= (36, 255, 255))))
        yellowColumn = yellowArray[0]
        if yellowColumn.size != 0:
            min1 = np.min(yellowColumn)
            max1 = np.max(yellowColumn)
            listMinX.append([min1, i])
            listMaxX.append([max1, i])

    # and -> or
    if len(listMinX) == 0 and len(listMaxX) == 0:
        return None

    minArray = np.array(listMinX)
    minArray = minArray[minArray[:, 1].argsort()]
    min = np.min(minArray[:, 0])
    index = np.where(minArray[:, 0] == min)
    minX = minArray[index]
    topRight = minArray[np.where(minArray[:, 0] == min) and np.where(minArray[:, 1] == np.max(minX[:, 1]))]
    topRight = [[topRight[0, 1], topRight[0, 0]]]
    topRight = np.array(topRight)

    maxArray = np.array(listMaxX)
    maxArray = maxArray[maxArray[:, 1].argsort()]
    max = np.max(maxArray[:, 0])
    position = np.where(maxArray[:, 0] == max)
    maxX = maxArray[position]
    bottomLeft = maxArray[np.where(maxArray[:, 0] == max) and np.where(maxArray[:, 1] == np.min(maxX[:, 1]))]
    bottomLeft = [[bottomLeft[0, 1], bottomLeft[0, 0]]]
    bottomLeft = np.array(bottomLeft)

    cnt = np.array([topRight, topLeft, bottomLeft, bottomRight])

    listContours = []
    listContours.extend(listMinX)
    listContours.extend(listMaxX)
    listContours.extend(listMinY)
    listContours.extend(listMaxY)

    contours = np.array(listContours)
    contours = np.unique(contours, axis=0)

    contoursXY = contours[contours[:, 0].argsort()]
    contoursYX = contoursXY[contoursXY[:, 1].argsort()]

    min5 = np.min(contoursXY[:, 0])
    max5 = np.max(contoursXY[:, 0])
    indexMin = np.where(contoursXY[:, 0] == min5)
    indexMax = np.where(contoursXY[:, 0] == max5)
    minXNew = contoursXY[indexMin]
    maxXNew = contoursXY[indexMax]
    topL = [min5, np.min(minXNew[:, 1])]
    bottomL = [max5, np.max(maxXNew[:, 1])]

    min6 = np.min(contoursYX[:, 1])
    max6 = np.max(contoursYX[:, 1])
    indexMin1 = np.where(contoursYX[:, 1] == min6)
    indexMax1 = np.where(contoursYX[:, 1] == max6)
    minXNew1 = contoursYX[indexMin1]
    maxXNew1 = contoursYX[indexMax1]
    top1 = [np.max(minXNew1[:, 0]), min6]
    bottom1 = [np.min(maxXNew1[:, 0]), max6]

    topLeft1 = [[topL[1], topL[0]]]
    topRight1 = [[top1[1], top1[0]]]
    bottomLeft1 = [[bottom1[1], bottom1[0]]]
    bottomRight1 = [[bottomL[1], bottomL[0]]]
    cnt1 = np.array([topRight1, topLeft1, bottomLeft1, bottomRight1])

    if cv2.contourArea(cnt) > cv2.contourArea(cnt1):
        cnt = cnt
    else:
        cnt = cnt1

    rect = cv2.minAreaRect(cnt)
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
    # cv2.drawContours(cropped, [box], 0, (0, 0, 255), 2)
    # return cropped
    return crop_rotated_rectangle(cropped, rect)
