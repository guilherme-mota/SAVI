#!/usr/bin/env python3

#-----------------
# Imports
#-----------------
import numpy as np
import cv2
from random import randint
from copy import deepcopy


def main():

    #-----------------
    # Initialization
    #-----------------
    # Two images, query (q) and target (t)
    q_path = "../../../savi_22-23/Parte06/images/machu_pichu/2.png"
    q_image = cv2.imread(q_path)
    q_gui = deepcopy(q_image)
    q_gray = cv2.cvtColor(q_image, cv2.COLOR_BGR2GRAY)
    q_win_name = 'Query Image'

    t_path = "../../../savi_22-23/Parte06/images/machu_pichu/1.png"
    t_image = cv2.imread(t_path)
    t_gui = deepcopy(t_image)
    t_gray = cv2.cvtColor(t_image, cv2.COLOR_BGR2GRAY)
    t_win_name = 'Target Image'


    #-----------------
    # Execution
    #-----------------
    # Craete Sift detection object
    sift = cv2.SIFT_create(nfeatures=200)

    # Detect Key Points in both images
    q_key_points, q_des = sift.detectAndCompute(q_gray, None)  # SIFT features
    t_key_points, t_des = sift.detectAndCompute(t_gray, None)  # SIFT features

    # Match Features (FLANN based matching)
    idx_params = dict(algorithm=1, trees=15)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(idx_params, search_params)
    best_two_matches = flann.knnMatch(q_des, t_des, k=2)

    # Create a list containing only the best matches
    matches = []
    for best_two_match in best_two_matches:
        best_match = best_two_match[0]
        second_best_match = best_two_match[1]

        # David Lowe's Test
        if best_match.distance < 0.3*second_best_match.distance:
            matches.append(best_match)


    # Compute Homography -------------------------------------------------

    # Create np.arrays of points to feed Find Homography function
    num_pts = len(matches)
    q_pts_array = np.ndarray((num_pts, 1, 2), dtype=np.float32)
    t_pts_array = np.ndarray((num_pts, 1, 2), dtype=np.float32)

    # first we need to create an np.array of size (n_pts, 1, 2) to feed into the function
    for match_idx, match in enumerate(matches):
        q_idx = match.queryIdx
        q_x = q_key_points[q_idx].pt[0]
        q_y = q_key_points[q_idx].pt[1]
        q_pts_array[match_idx, 0, 0] = q_x
        q_pts_array[match_idx, 0, 1] = q_y

        t_idx = match.trainIdx
        t_x = t_key_points[t_idx].pt[0]
        t_y = t_key_points[t_idx].pt[1]
        t_pts_array[match_idx, 0, 0] = t_x
        t_pts_array[match_idx, 0, 1] = t_y

    # Find Homography 
    M, mask = cv2.findHomography(q_pts_array, t_pts_array, cv2.RANSAC)

    # Warp q_image to move it to the t_image coordinate frame
    q_h, q_w, _ = q_image.shape
    t_h, t_w, _ = t_image.shape

    # When q_image is inside t_image
    stitched_image_h = t_h
    stitched_image_w = t_w

    q_image_warped = cv2.warpPerspective(q_image, M, (stitched_image_w, stitched_image_h))
    q_image_warped = q_image_warped[:,:,0:3] # remove fourth channel

    # Stich Images
    # Alt 1: Average all the pixels
    # stitched_image = ((t_image.astype(float) + q_image_warped.astype(float))/2).astype(np.uint8)

    # Alt 2: Use q pixels in overlaping region
    # overlap_mask = q_image_warped > 0
    # stitched_image = deepcopy(t_image)
    # stitched_image[overlap_mask] = q_image_warped[overlap_mask]

    # Alt 3: Average only overlaping pixels
    overlap_mask = q_image_warped > 0
    stitched_image = deepcopy(t_image)
    stitched_image[overlap_mask] = ((q_image_warped[overlap_mask].astype(float) + stitched_image[overlap_mask].astype(float))/2).astype(np.uint8)


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

    cv2.namedWindow('Stitched Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Stitched Image', 600, 400)
    cv2.imshow('Stitched Image', stitched_image)


    #-----------------
    # Termination
    #-----------------
    cv2.waitKey(0)


if __name__ == "__main__":
    main()