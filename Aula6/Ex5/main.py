#!/usr/bin/env python3

#-----------------
# Imports
#-----------------
import numpy as np
import cv2
import matplotlib.pyplot as plt


def main():
    #-----------------
    # Initialization
    #-----------------
    MIN_MATCH_COUNT = 10

    #-----------------
    # Execution
    #-----------------
    query_image = cv2.imread('/home/guilherme/savi_22-23/Parte06/images/machu_pichu/1.png', cv2.IMREAD_GRAYSCALE)
    train_image = cv2.imread('/home/guilherme/savi_22-23/Parte06/images/machu_pichu/2.png', cv2.IMREAD_GRAYSCALE)

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    key_points_query_image, des1 = sift.detectAndCompute(query_image, None)
    key_points_train_image, des2 = sift.detectAndCompute(train_image, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ key_points_query_image[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ key_points_train_image[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = query_image.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        
        train_image = cv2.polylines(train_image, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None

    # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
    #                singlePointColor = None,
    #                matchesMask = matchesMask, # draw only inliers
    #                flags = 2)

    # img3 = cv2.drawMatches(query_image, key_points_query_image, train_image, key_points_train_image, good, None, **draw_params)
    #plt.imshow(img3, 'gray'),plt.show()

    dst = cv2.warpPerspective(query_image, M, ((query_image.shape[1] + train_image.shape[1]), train_image.shape[0])) #wraped image

    # now paste them together
    dst[0:train_image.shape[0], 0:train_image.shape[1]] = train_image
    dst[0:query_image.shape[0], 0:query_image.shape[1]] = query_image

    cv2.namedWindow('Query Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Query Image', 600, 400)
    cv2.imshow('Query Image', query_image)
    
    cv2.namedWindow('Train Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Train Image', 600, 400)
    cv2.imshow('Train Image', train_image)

    cv2.waitKey(0)


if __name__ == "__main__":
    main()