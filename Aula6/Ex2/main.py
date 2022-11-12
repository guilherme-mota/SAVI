#!/usr/bin/env python3

#-----------------
# Imports
#-----------------
import numpy as np
import cv2
import matplotlib.pyplot as plt
from random import randint


def main():

    #-----------------
    # Initialization
    #-----------------
    image1 = cv2.imread("../../../savi_22-23/Parte06/images/santorini/1.png")  # /home/guilherme/savi_22-23/Parte06/images/santorini/1.png

    #-----------------
    # Execution
    #-----------------
    # Convert image BGR to Gray
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    # Detect Key Points
    sift = cv2.SIFT_create(nfeatures=500)
    key_points, des = sift.detectAndCompute(gray1, None)

    # Draw key points using opencv's option
    # image1 = cv2.drawKeypoints(image1, key_points, image1)

    # Draw Key Points
    for idx, key_point in enumerate(key_points):
        x = int(key_point.pt[0])
        y = int(key_point.pt[1])
        color = (randint(0, 255), randint(0, 255), randint(0, 255))
        cv2.circle(image1, (x, y), 80, color, 3)

    # Visualize
    cv2.namedWindow('Image1', cv2.WINDOW_NORMAL)
    cv2.imshow('Image1', image1)

    #-----------------
    # Termination
    #-----------------
    cv2.waitKey(0)


if __name__ == "__main__":
    main()