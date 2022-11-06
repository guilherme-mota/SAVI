#!/usr/bin/env python3

#-----------------
# Imports
#-----------------
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def main():
    #-----------------
    # Execution
    #-----------------
    img = cv.imread('fly.png',0)

    # Create SURF object. You can specify params here or later.
    # Here I set Hessian Threshold to 400
    surf = cv.xfeatures2d.SURF_create(400)

    # Find keypoints and descriptors directly
    kp, des = surf.detectAndCompute(img,None)
    print( surf.getHessianThreshold() )

    surf.setHessianThreshold(50000)
    print(len(kp))

    kp, des = surf.detectAndCompute(img,None)

    img2 = cv.drawKeypoints(img,kp,None,(255,0,0),4)

    plt.imshow(img2),plt.show()
 

if __name__ == "__main__":
    main()