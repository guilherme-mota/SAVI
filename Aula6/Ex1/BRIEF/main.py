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
    
    # Initiate FAST detector
    star = cv.xfeatures2d.StarDetector_create()

    # Initiate BRIEF extractor
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()

    # find the keypoints with STAR
    kp = star.detect(img,None)

    # compute the descriptors with BRIEF
    kp, des = brief.compute(img, kp)

    print( brief.descriptorSize() )
    print( des.shape )


if __name__ == "__main__":
    main()