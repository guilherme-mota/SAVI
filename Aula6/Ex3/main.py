#!/usr/bin/env python3

#-----------------
# Imports
#-----------------
import numpy as np
import cv2
import matplotlib.pyplot as plt
from random import randint
from copy import deepcopy


def main():

    #-----------------
    # Initialization
    #-----------------
    # Two images, query (q) and target (t)
    q_path = "../../../savi_22-23/Parte06/images/castle/1.png"
    q_image = cv2.imread(q_path)
    q_gui = deepcopy(q_image)
    q_gray = cv2.cvtColor(q_image, cv2.COLOR_BGR2GRAY)
    q_win_name = 'Query Image'

    t_path = "../../../savi_22-23/Parte06/images/castle/2.png"
    t_image = cv2.imread(t_path)
    t_gui = deepcopy(t_image)
    t_gray = cv2.cvtColor(t_image, cv2.COLOR_BGR2GRAY)
    t_win_name = 'Target Image'

    #-----------------
    # Execution
    #-----------------
    # Craete Sift detection object
    sift = cv2.SIFT_create(nfeatures=500)

    # Detect Key Points in both images
    q_key_points, q_des = sift.detectAndCompute(q_gray, None)
    t_key_points, t_des = sift.detectAndCompute(t_gray, None)

    # Match Features (FLANN based matching)
    idx_params = dict(algorithm=1, trees=15)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(idx_params, search_params)
    best_two_matches = flann.knnMatch(q_des, t_des, k=2)

    # Create a list containing only the best matches
    matches = []
    for best_two_match in best_two_matches:
        matches.append(best_two_match[0])

    #  Visualize -------------------------------------------------

    # Draw Query Key Points
    for idx, key_point in enumerate(q_key_points):
        x = int(key_point.pt[0])
        y = int(key_point.pt[1])
        color = (randint(0, 255), randint(0, 255), randint(0, 255))
        cv2.circle(q_gui, (x, y), 30, color, 3)

     # Draw Target Key Points
    for idx, key_point in enumerate(t_key_points):
        x = int(key_point.pt[0])
        y = int(key_point.pt[1])
        color = (randint(0, 255), randint(0, 255), randint(0, 255))
        cv2.circle(t_gui, (x, y), 30, color, 3)

    # Show the matches image
    matches_img = cv2.drawMatches(q_image, q_key_points, t_image, t_key_points, matches, None)

    # Show Windows with Images
    cv2.namedWindow(q_win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(q_win_name, 600, 400)
    cv2.imshow(q_win_name, q_gui)

    cv2.namedWindow(t_win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(t_win_name, 600, 400)
    cv2.imshow(t_win_name, t_gui)

    cv2.namedWindow('Matches', cv2.WINDOW_NORMAL)
    cv2.imshow('Matches', matches_img)

    #-----------------
    # Termination
    #-----------------
    cv2.waitKey(0)


if __name__ == "__main__":
    main()