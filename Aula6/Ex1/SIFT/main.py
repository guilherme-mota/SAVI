#!/usr/bin/env python3

#-----------------
# Imports
#-----------------
import numpy as np
import cv2 as cv


def main():
    #-----------------
    # Execution
    #-----------------
    img = cv.imread('home.jpg')
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    #kp = sift.detect(gray,None)
    kp, des = sift.detectAndCompute(gray,None)

    img = cv.drawKeypoints(gray,kp,img)

    cv.imwrite('sift_keypoints.jpg',img)

    img = cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imwrite('sift_keypoints.jpg',img)
        

if __name__ == "__main__":
    main()