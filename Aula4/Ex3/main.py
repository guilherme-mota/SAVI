#!/usr/bin/env python3

from copy import deepcopy
from sys import flags
import cv2
import csv
import numpy as np

def main():
    # ------------------------------------------
    # Initialization
    # ------------------------------------------
    capture = cv2.VideoCapture("/home/guilherme/savi_22-23/Parte04/docs/OxfordTownCentre/TownCentreXVID.mp4")
    if capture.isOpened() == False:
        print("Error opening video stream or file!")

    ret, frame = capture.read()
    window_name = 'Tracking'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 500)

    # Count number of persons in the data set
    number_of_persons = 0
    file = open('/home/guilherme/savi_22-23/Parte04/docs/OxfordTownCentre/TownCentre-groundtruth.top')
    csv_reader = csv.reader(file)  # Read CSV file with ground truth bboxes

    for row in csv_reader:

        if len(row) != 12:  # skip badly formated rows
            continue

        person_number, frame_number, _, _, _, _, _, _, body_left, body_top, body_right, body_bottom = row
        person_number = int(person_number)
        if person_number >= number_of_persons:
            number_of_persons = person_number + 1

    # Create the colors for each person
    colors = np.random.randint(0, high=255, size=(number_of_persons, 3), dtype=int)

    person_detector = cv2.CascadeClassifier('haarcascade_fullbody.xml')

    # ------------------------------------------
    # Execution
    # ------------------------------------------
    frame_counter = 0
    while capture.isOpened():  # loop through all frames whil
        ret, image_original = capture.read()  # get a frame, ret will be true or false if getting succeeds
        image_gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
        image_gui = deepcopy(image_original)  # image for graphical user interface

        if ret == False:
            break

        # ------------------------------------------
        # Draw ground truth bboxes
        # ------------------------------------------
        # file = open('/home/guilherme/savi_22-23/Parte04/docs/OxfordTownCentre/TownCentre-groundtruth.top')
        # csv_reader = csv.reader(file)  # Read CSV file with ground truth bboxes
        # for row in csv_reader:
            
        #     if len(row) != 12:  # skip badly formated rows
        #         continue

        #     person_number, frame_number, _, _, _, _, _, _, body_left, body_top, body_right, body_bottom = row
        #     person_number = int(person_number) # convert to number format (integer)
        #     frame_number = int(frame_number)
        #     body_left = int(float(body_left))
        #     body_right = int(float(body_right))
        #     body_top = int(float(body_top))
        #     body_bottom = int(float(body_bottom))

        #     if frame_counter != frame_number: # do not draw bbox of other frames
        #         continue

        #     x1 = body_left
        #     y1 = body_top
        #     x2 = body_right
        #     y2 = body_bottom
        #     color = colors[person_number,:]

        #     cv2.rectangle(image_gui, (x1, y1), (x2, y2), (int(color[0]), int(color[1]), int(color[2])), 3)

        #     print('person ' + str(person_number) + ' frame ' + str(frame_number))

        # ------------------------------------------
        # Detection of persons 
        # ------------------------------------------
        bboxes = person_detector.detectMultiScale(image_gray, scaleFactor=1.2, minNeighbors=4, minSize=(20,40))
        print(bboxes)

        # Display Image Capture with BBoxes
        cv2.imshow(window_name, image_gui)

        if cv2.waitKey(25) == ord('q'):
            break

        frame_counter += 1

    # ------------------------------------------
    # Termination
    # ------------------------------------------
    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()