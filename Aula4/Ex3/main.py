#!/usr/bin/env python3

from copy import deepcopy
from sys import flags
import cv2
import csv
import numpy as np
from functions import Detection, Tracker

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

    person_detector = cv2.CascadeClassifier('/home/guilherme/workingcopy/opencv-4.5.4/data/haarcascades/haarcascade_fullbody.xml')

    # ------------------------------------------
    # Initialize variables
    # ------------------------------------------
    frame_counter = 0
    detection_counter = 0
    traker_counter = 0
    trackers = []
    iou_threshold = 0.8

    # ------------------------------------------
    # Execution
    # ------------------------------------------
    while capture.isOpened():  # loop through all frames
        ret, image_original = capture.read()  # get a frame, ret will be true or false if getting succeeds
        image_gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
        image_gui = deepcopy(image_original)  # image for graphical user interface

        if ret == False:
            break

        # Create time stamp
        stamp = float(capture.get(cv2.CAP_PROP_POS_MSEC))/1000

        # ------------------------------------------
        # Detection of persons 
        # ------------------------------------------
        bboxes = person_detector.detectMultiScale(image_gray, scaleFactor=1.2, minNeighbors=4, minSize=(20,40))

        # ------------------------------------------
        # Create detections per haard cascade bbox
        # ------------------------------------------
        detections = []  # list is new for each frame of the video
        for bbox in bboxes:  # cycle all bboxes
            x1, y1, w, h = bbox
            detection = Detection(x1, y1, w, h, image_gray, detection_counter, stamp)
            detection_counter += 1
            detection.draw(image_gui)
            detections.append(detection)

        # ------------------------------------------------------------------------------------
        # For each detection, see if there is a tracker to wich it should be associated
        # ------------------------------------------------------------------------------------
        for detection in detections:  # cycle all detections
            for tracker in trackers:  # cycle all trackers
                tracker_bbox = tracker.detections[-1]
                iou = detection.computeIOU(tracker_bbox)
                # print('IOU( T' + str(tracker.id) + ' D' + str(detection.id) + ' ) = ' + str(iou))

                if iou > iou_threshold:  # associate detection with tracker
                    tracker.addDetection(detection, image_gray)

        # ------------------------------
        # Track using Template Matching
        # ------------------------------
        for tracker in trackers:  # cycle all trackers
            last_detection_id = tracker.detections[-1].id
            detections_ids = [d.id for d in detections]
            if not last_detection_id in detections_ids:
                print('Tracker ' + str(tracker.id) + ' doing some tracking')
                tracker.track(image_gray)

        # ------------------------------------------------------------
        # Deactivate Tracker if no detection for more then t secs
        # ------------------------------------------------------------
        for tracker in trackers:  # cycle all trackers
            tracker.updateTime(stamp)

        # -------------------------------------------------
        # Create tracker foreach detection
        # -------------------------------------------------
        for detection in detections:  # cycle all detections
            if not detection.assigned_to_tracker:
                tracker = Tracker(detection, traker_counter, image_gray)
                traker_counter += 1
                trackers.append(tracker)

        # ------------------------------------------
        # Draw stuff
        # ------------------------------------------
        # Draw Trackers
        for tracker in trackers:
            tracker.draw(image_gui)
            # print(tracker)

        # Display Image Capture with BBoxes
        cv2.imshow(window_name, image_gui)

        if cv2.waitKey(100) == ord('q'):
            break

        frame_counter += 1

    # ------------------------------------------
    # Termination
    # ------------------------------------------
    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()