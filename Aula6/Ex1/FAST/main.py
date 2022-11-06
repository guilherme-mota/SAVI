#!/usr/bin/env python3

#-----------------
# Imports
#-----------------
import numpy as np
import cv2 as cv
from matplotlib import pyplot as pl


def main():
    #-----------------
    # Execution
    #-----------------
    img = cv.imread('/home/guilherme/workingcopy/opencv-4.5.4/samples/data/blox.jpg',0)

    # Initiate FAST object with default values
    fast = cv.FastFeatureDetector_create()

    # find and draw the keypoints
    kp = fast.detect(img,None)
    img2 = cv.drawKeypoints(img, kp, None, color=(255,0,0))

    # Print all default params
    print( "Threshold: {}".format(fast.getThreshold()) )
    print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
    print( "neighborhood: {}".format(fast.getType()) )
    print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )

    cv.imwrite('fast_true.png', img2)

    # Disable nonmaxSuppression
    fast.setNonmaxSuppression(0)
    kp = fast.detect(img, None)
    print( "Total Keypoints without nonmaxSuppression: {}".format(len(kp)) )
    img3 = cv.drawKeypoints(img, kp, None, color=(255,0,0))
    cv.imwrite('fast_false.png', img3)
    


if __name__ == "__main__":
    main()