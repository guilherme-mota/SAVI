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
    img = cv.imread('../Corner_with_SubPixel_Accuracy/xadrez2.jpg')  #/home/guilherme/SAVI/Aula6/Ex1
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    corners = cv.goodFeaturesToTrack(gray,25,0.01,10)
    corners = np.int0(corners)

    for i in corners:
        x,y = i.ravel()
        cv.circle(img,(x,y),3,255,-1)

    plt.imshow(img),plt.show()
    

if __name__ == "__main__":
    main()