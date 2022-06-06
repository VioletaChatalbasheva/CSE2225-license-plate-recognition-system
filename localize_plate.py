import cv2
import numpy as np


def localize(inputImage):
    # Ensure that the current frame is an rgb image.
    if inputImage.shape[2] != 3:
        return None

    # Get the masked image of the yellow license plate.
    yellow_mask = convert_to_HSV(inputImage)

    # Create a small kernel to remove some of the noise.
    kernel_noise = np.ones((4, 4), np.uint8)
    # Create a bigger kernel to fill the holes after reducing the noise.
    kernel_dilate = np.ones((35, 35), np.uint8)
    # Create a bigger kernel to delete pixels on edge that were added after the dilation.
    kernel_erode = np.ones((20, 20), np.uint8)

    # Reduce the noise
    img_erode = cv2.erode(yellow_mask, kernel_noise, 1)
    # Fill in the holes after the erosion.
    img_dilate = cv2.dilate(img_erode, kernel_dilate, 1)
    # Delete the newly introduced pixels on the edges.
    img_erode = cv2.erode(img_dilate, kernel_erode, 1)

    # Bitwise-AND the input frame and the mask image.
    yellow = cv2.bitwise_and(inputImage, inputImage, mask=img_erode)

    # Get the pixels with yellow-ish intensities.
    plate = np.where(np.logical_and(yellow >= (15, 100, 100), yellow <= (36, 255, 255)))

    # and -> or
    if plate[0].size == 0 and plate[1].size == 0:
        return None

    # Get the unique pixel values of the plate on the x axis and use them for cropping the plate.
    x = np.unique(plate[0])

    # Find the coordinates for cropping on the x axis.
    max_x, min_x = calculate_crop_coordinates(x)

    # Crop the plate on the x axis.
    cropped_x = yellow[min_x:max_x, :]

    # Get the pixel values with yellow intensities from the already cropped image.
    plate = np.where(np.logical_and(cropped_x >= (15, 100, 100), cropped_x <= (36, 255, 255)))

    # Get the unique pixel values of the plate on the y axis and use them for cropping the plate.
    y = np.unique(plate[1])

    # Find the coordinates for cropping on the y axis.
    max_y, min_y = calculate_crop_coordinates(y)

    # Crop the plate on the y axis.
    cropped_xy = inputImage[min_x:max_x, min_y:max_y]

    return localize2(cropped_xy)


def localize2(cropped):
    # if cropped is None or cropped.shape[0] == 0 or cropped.shape[1] == 0:
    #     return None

    # Get the masked image of the yellow license plate.
    yellow_mask = convert_to_HSV(cropped)

    # Bitwise-AND the cropped license plate and the mask image.
    yellow = cv2.bitwise_and(cropped, cropped, mask=yellow_mask)

    # Create a kernel to connect the disconnected components.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    yellow = cv2.dilate(yellow, kernel, iterations=1)

    height = yellow.shape[0]
    width = yellow.shape[1]

    list_min_y = []
    list_max_y = []
    for i in range(0, height):
        yellow_array = np.array(np.where(np.logical_and(yellow[i, :] >= (15, 100, 100), yellow[i, :] <= (36, 255, 255))))
        print(yellow_array)
        yellow_row = yellow_array[0]
        if yellow_row.size != 0:
            min1 = np.min(yellow_row)
            max1 = np.max(yellow_row)
            list_min_y.append([i, min1])
            list_max_y.append([i, max1])

    if len(list_min_y) == 0 or len(list_max_y) == 0:
        return None

    minArray = np.array(list_min_y)

    minArray = minArray[minArray[:, 0].argsort()]
    min = np.min(minArray[:, 1])
    index = np.where(minArray[:, 1] == min)
    minX = minArray[index]
    topLeft = minArray[np.where(minArray[:, 1] == min) and np.where(minArray[:, 0] == np.min(minX[:, 0]))]
    topLeft = [[topLeft[0, 1], topLeft[0, 0]]]
    topLeft = np.array(topLeft)

    maxArray = np.array(list_max_y)

    maxArray = maxArray[maxArray[:, 0].argsort()]
    max = np.max(maxArray[:, 1])
    index = np.where(maxArray[:, 1] == max)
    maxX = maxArray[index]
    bottomRight = maxArray[np.where(maxArray[:, 1] == max) and np.where(maxArray[:, 0] == np.max(maxX[:, 0]))]
    bottomRight = [[bottomRight[0, 1], bottomRight[0, 0]]]
    bottomRight = np.array(bottomRight)

    list_min_x = []
    list_max_x = []
    for i in range(0, width):
        yellow_array = np.array(np.where(np.logical_and(yellow[:, i] >= (15, 100, 100), yellow[:, i] <= (36, 255, 255))))
        yellow_column = yellow_array[0]
        if yellow_column.size != 0:
            min1 = np.min(yellow_column)
            max1 = np.max(yellow_column)
            list_min_x.append([min1, i])
            list_max_x.append([max1, i])

    # and -> or
    if len(list_min_x) == 0 and len(list_max_x) == 0:
        return None

    minArray = np.array(list_min_x)
    minArray = minArray[minArray[:, 1].argsort()]
    min = np.min(minArray[:, 0])
    index = np.where(minArray[:, 0] == min)
    minX = minArray[index]
    topRight = minArray[np.where(minArray[:, 0] == min) and np.where(minArray[:, 1] == np.max(minX[:, 1]))]
    topRight = [[topRight[0, 1], topRight[0, 0]]]
    topRight = np.array(topRight)

    maxArray = np.array(list_max_x)
    maxArray = maxArray[maxArray[:, 1].argsort()]
    max = np.max(maxArray[:, 0])
    position = np.where(maxArray[:, 0] == max)
    maxX = maxArray[position]
    bottomLeft = maxArray[np.where(maxArray[:, 0] == max) and np.where(maxArray[:, 1] == np.min(maxX[:, 1]))]
    bottomLeft = [[bottomLeft[0, 1], bottomLeft[0, 0]]]
    bottomLeft = np.array(bottomLeft)

    cnt = np.array([topRight, topLeft, bottomLeft, bottomRight])

    listContours = []
    listContours.extend(list_min_x)
    listContours.extend(list_max_x)
    listContours.extend(list_min_y)
    listContours.extend(list_max_y)

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


def convert_to_HSV(image):
    # Convert the image to the HSV color-space to extract a colored object.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define ranges of the yellow color in HSV - the color of the Dutch license plate.
    low_yellow = np.array([15, 100, 100])
    high_yellow = np.array([36, 255, 255])

    # Threshold the HSV image to get only the yellow colors.
    yellow_mask = cv2.inRange(hsv, low_yellow, high_yellow)

    return yellow_mask


def calculate_crop_coordinates(coordinates):
    count = 0
    max_coordinate = 0
    min_coordinate = 0

    # Find the min and max pixel values.
    for i in range(0, coordinates.size - 1):
        # Check if the pixel values are consecutive.
        if coordinates[i] + 1 == coordinates[i + 1]:
            count = count + 1
        # Assign the minimum and maximum pixel for cropping.
        elif max_coordinate - min_coordinate < count:
            max_coordinate = coordinates[i]
            min_coordinate = coordinates[i] - count
            count = 0
        else:
            count = 0

    # In case all pixel values are consecutive.
    if min_coordinate == 0 and max_coordinate == 0:
        min_coordinate = coordinates[0]
        max_coordinate = coordinates[coordinates.size - 1]

    return max_coordinate, min_coordinate


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
    return center, (width, height), angle


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
