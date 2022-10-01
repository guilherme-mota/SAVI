#!/usr/bin/env python3

import numpy as np
import cv2


def main():
    print("Find Wally!")

    img = cv2.imread("../../savi_22-23/Parte02/images/scene.jpg")
    H,W,_ = img.shape
    img_grey =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template = cv2.imread("../../savi_22-23/Parte02/images/wally.png")
    h,w,nc = template.shape

    # Apply template Matching
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    img_gui = img*0

    top_left = (max_loc[0], max_loc[1])
    bottom_right = (top_left[0] + w, top_left[1] + h)

    mask = np.zeros((H,W)).astype(np.uint8)

    # color = (0, 0, 255) #BGR format
    cv2.rectangle(mask, top_left, bottom_right, 255, -1)

    mask = mask.astype(bool)
    img_gui[mask] = img[mask]

    negated_mask = ~mask # np.logical

    merged = cv2.merge([img_grey, img_grey, img_grey])

    # cv2.imshow("Image RGB", img)
    cv2.imshow("Image Grey", img_gui)
    cv2.waitKey(0)


if __name__ == "__main__":
     main()