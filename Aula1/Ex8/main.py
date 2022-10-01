#!/usr/bin/env python3

# OpenCV Tutorial - Performance Measurement and Improvement Techniques

import numpy as np
import cv2 as cv

def main():
    # check if optimization is enabled
    print(cv.useOptimized())

    # Disable it
    cv.setUseOptimized(False)
    print(cv.useOptimized())

    # Measuring Performance with OpenCV
    img1 = cv.imread('/home/guilherme/SAVI/Aula1/Ex1/UA.jpg.png')

    e1 = cv.getTickCount()

    # your code execution
    for i in range(5,49,2):
        img1 = cv.medianBlur(img1,i)
    
    e2 = cv.getTickCount()

    time = (e2 - e1)/ cv.getTickFrequency()
    print(time)

if __name__ == "__main__":
     main()