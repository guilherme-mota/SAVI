#!/usr/bin/env python3

import numpy as np
import cv2

def main():
    print("Find Wally!")

    img = cv2.imread("../../savi_22-23/Parte02/images/scene.jpg")
    template = cv2.imread("../../savi_22-23/Parte02/images/wally.png")
    h,w,nc = template.shape

    # Apply template Matching
    res = cv2.matchTemplate(img, template, cv2.TM_SQDIFF)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    top_left = min_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img,top_left, bottom_right, 255, 2)

    cv2.imshow("Display window", img)

    cv2.waitKey(0)


if __name__ == "__main__":
     main()