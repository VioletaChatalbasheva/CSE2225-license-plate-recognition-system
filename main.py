import cv2
import csv

from segment import segmentCharacters
from localize_plate import localize

# image = cv2.imread('car1.jpg')
# cropped = localize1(image)
# segmentCharacters(cropped)
# cv2.imshow('frame', cropped)
# cv2.waitKey(0)

cap = cv2.VideoCapture('video/trainingsvideo.avi')

with open('output.csv', 'w', newline='') as csvfile:
    fieldnames = ['License plate', 'Frame no.', 'Timestamp(seconds)']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    frameList = []
    timeList = []

    # contains only plates 8 characters long
    allPlates = []

    frameNumber = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if frame is not None:
            frameNumber = frameNumber + 1
            cropped = localize(frame)
            segmentCharacters(cropped, frameNumber, cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, writer, allPlates, frameList,
                              timeList)
            if cropped is not None:
                cv2.imshow('frame', cropped)

            if cv2.waitKey(50) & 0xFF == ord('q'):
                break
        else:
            break

cap.release()
cv2.destroyAllWindows()
