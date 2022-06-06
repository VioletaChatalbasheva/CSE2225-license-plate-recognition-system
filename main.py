import cv2
import csv

from segment import segment_characters
from localize_plate import localize

# image = cv2.imread('car.jpg')
# cropped = localize(image)
# # segment_characters(cropped)
# cv2.imshow('frame', cropped)
# cv2.waitKey(0)

# Create a VideoCapture object to read the video file.
cap = cv2.VideoCapture('video/trainingsvideo.avi')

with open('output.csv', 'w', newline='') as csvfile:
    # Create the columns with regards to the information that will be saved as a final result of the recognition.
    fieldnames = ['License plate', 'Frame no.', 'Timestamp(seconds)']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    frameList = []
    timeList = []

    # Contain only valid plates, 8 characters long.
    allPlates = []

    # Keep track of the frame number.
    frameNumber = 0

    # Start processing the video.
    while cap.isOpened():
        # Read the current frame.
        ret, frame = cap.read()

        # Start processing the frame.
        if frame is not None:
            # Increase the frame number.
            frameNumber = frameNumber + 1

            # Localize and crop the license plate.
            cropped = localize(frame)

            # Segment the characters of the plate and recognize them.
            segment_characters(cropped, frameNumber, cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, writer, allPlates,
                               frameList, timeList)
            if cropped is not None:
                cv2.imshow('frame', cropped)

            if cv2.waitKey(50) & 0xFF == ord('q'):
                break

        # Break the loop.
        else:
            break

# When the video is processed, release the VideoCapture object.
cap.release()

# Close all frames.
cv2.destroyAllWindows()
