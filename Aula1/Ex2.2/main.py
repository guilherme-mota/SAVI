#!/usr/bin/env python3

# OpenCV Tutorial - Playing Video from file

import numpy as np
import cv2 as cv

def main():
    cap = cv.VideoCapture('/home/guilherme/SAVI/Aula1/Ex2.3/output.avi')

    while cap.isOpened():
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cv.imshow('frame', gray)

        if cv.waitKey(1) == ord('q'):
            break
        
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
     main()