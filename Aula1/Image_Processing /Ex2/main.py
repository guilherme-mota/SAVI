#!/usr/bin/env python3

# OpenCV Tutorial - Geometric Transformations of Images 

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def main():
    # Scaling
    img = cv.imread('/home/guilherme/SAVI/Aula1/Ex6/opencv-logo.png')
    res = cv.resize(img,None,fx=2, fy=2, interpolation = cv.INTER_CUBIC)
    #OR
    height, width = img.shape[:2]
    res = cv.resize(img,(2*width, 2*height), interpolation = cv.INTER_CUBIC)

    # Translation
    rows,cols,_ = img.shape
    M = np.float32([[1,0,100],[0,1,50]])
    dst = cv.warpAffine(img,M,(cols,rows))
    cv.imshow('img',dst)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Rotation
    rows,cols = img.shape
    # cols-1 and rows-1 are the coordinate limits.
    M = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),90,1)
    dst = cv.warpAffine(img,M,(cols,rows))

    # Affine Transformation
    rows,cols,ch = img.shape
    pts1 = np.float32([[50,50],[200,50],[50,200]])
    pts2 = np.float32([[10,100],[200,50],[100,250]])
    M = cv.getAffineTransform(pts1,pts2)
    dst = cv.warpAffine(img,M,(cols,rows))

    plt.subplot(121),plt.imshow(img),plt.title('Input')
    plt.subplot(122),plt.imshow(dst),plt.title('Output')
    plt.show()

    # Perspective Transformation
    rows,cols,ch = img.shape
    pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
    pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
    M = cv.getPerspectiveTransform(pts1,pts2)
    dst = cv.warpPerspective(img,M,(300,300))
    plt.subplot(121),plt.imshow(img),plt.title('Input')
    plt.subplot(122),plt.imshow(dst),plt.title('Output')
    plt.show()

if __name__ == "__main__":
     main()