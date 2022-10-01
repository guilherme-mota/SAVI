#!/usr/bin/env python3

import cv2 as cv
import sys

def main():
    img = cv.imread("UA.jpg")

    if img is None:
        sys.exit("Could not read the image.")

    cv.imshow("Display window", img)

    k = cv.waitKey(0)

    if k == ord("s"):
        cv.imwrite("UA.jpg.png", img)

if __name__ == "__main__":
     main()