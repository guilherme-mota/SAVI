#!/usr/bin/env python3

# OpenCV Tutorial - Arithmetic Operations on Images

import numpy as np
import cv2 as cv


def main():
    x = np.uint8([250])
    y = np.uint8([10])

    print( cv.add(x,y) ) # 250+10 = 260 => 255

    print( x+y ) # 250+10 = 260 % 256 = 4

    # Image Blending
    img1 = cv.imread('/home/guilherme/SAVI/Aula1/Ex1/UA.jpg.png')
    print( img1.shape )
    img1_resized = cv.resize(img1, (176, 124))
    print( img1_resized.shape )

    img2 = cv.imread('/home/guilherme/SAVI/Aula1/Ex6/opencv-logo.png')
    print( img2.shape )

    dst = cv.addWeighted(img1_resized,0.7,img2,0.3,0)
    cv.imshow('dst',dst)

    cv.waitKey(0)
    cv.destroyAllWindows()

    # Bitwise Operations
    # I want to put logo on top-left corner, So I create a ROI
    rows,cols,channels = img2.shape
    roi = img1_resized[0:rows, 0:cols]

    # Now create a mask of logo and create its inverse mask also
    img2gray = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
    ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)
    mask_inv = cv.bitwise_not(mask)

    # Now black-out the area of logo in ROI
    img1_bg = cv.bitwise_and(roi,roi,mask = mask_inv)

    # Take only region of logo from logo image.
    img2_fg = cv.bitwise_and(img2,img2,mask = mask)

    # Put logo in ROI and modify the main image
    dst = cv.add(img1_bg,img2_fg)
    img1_resized[0:rows, 0:cols ] = dst

    cv.imshow('res',img1_resized)

    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
     main()