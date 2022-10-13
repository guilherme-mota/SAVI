#!/usr/bin/env python3

from copy import deepcopy
import cv2
import csv
import numpy as np

def main():
    # ------------------------------------------
    # Initialization
    # ------------------------------------------
    # Get the video file and read it
    video = cv2.VideoCapture("/home/guilherme/savi_22-23/Parte04/docs/OxfordTownCentre/TownCentreXVID.mp4")
    ret, frame = video.read()
    frame_height, frame_width = frame.shape[:2]
    
    num_of_persons = 0
    # Read CSV file
    file = open('/home/guilherme/savi_22-23/Parte04/docs/OxfordTownCentre/TownCentre-groundtruth.top') 
    csv_reader = csv.reader(file)
    # for row in csv_reader:
    #         # skip badly formated rows
    #         if len(row) != 12:
    #             continue

    #         personNumber, frameNumber, _, _, _, _, _, _, bodyLeft, bodyTop, bodyRight, bodyBottom = row
    #         personNumber = int(personNumber)
    #         if personNumber >= num_of_persons:
    #             num_of_persons = personNumber + 1

    # ------------------------------------------
    # Execution
    # ------------------------------------------
    frame_counter = 0
    # Start tracking
    while True:
        ret, frame = video.read()
        frame = cv2.resize(frame, (frame_width//2, frame_height//2))
        img_gui = deepcopy(frame)

        file = open('/home/guilherme/savi_22-23/Parte04/docs/OxfordTownCentre/TownCentre-groundtruth.top')
        csv_reader = csv.reader(file)
        
        for row in csv_reader:
            # skip badly formated rows
            if len(row) != 12:
                continue

            personNumber, frameNumber, _, _, _, _, _, _, bodyLeft, bodyTop, bodyRight, bodyBottom = row
            personNumber = int(personNumber)
            frameNumber = int(frameNumber)
            bodyLeft = int(float(bodyLeft))
            bodyTop = int(float(bodyTop))
            bodyRight = int(float(bodyRight))
            bodyBottom = int(float(bodyBottom))

            if frame_counter != frameNumber:
                continue

            cv2.rectangle(img_gui,(bodyLeft, bodyTop),(bodyRight, bodyBottom),(0,255,0),3)

        cv2.imshow("Tracking", img_gui)

        if cv2.waitKey(1) == ord('q'):
            break

        frame_counter += 1

    # ------------------------------------------
    # Termination
    # ------------------------------------------
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()