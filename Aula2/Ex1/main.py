#!/usr/bin/env python3

from time import sleep
import numpy as np
import cv2

def main():
    img = cv2.imread("../../savi_22-23/Parte02/images/lake.jpg")

    # img_dark = (img*0.1).astype(np.uint8)

    img2 = img

    h,w,nc = img2.shape

    for i in np.arange(1, 0, -0.1):
        img2[:, int(w/2) :] = (img2[:, int(w/2) :]*i).astype(np.uint8)

        cv2.imshow("Display window", img2)

        cv2.waitKey(10)

        sleep(1)

    k = cv2.waitKey(0)

if __name__ == "__main__":
     main()