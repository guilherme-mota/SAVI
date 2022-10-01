#!/usr/bin/env python3

# OpenCV Tutorial - Mouse as a Paint-Brush

import numpy as np
import cv2 as cv

# List all available events
# events = [i for i in dir(cv) if 'EVENT' in i]
# print( events )

# Create a black image, a window and bind the function to window
img = np.zeros((512,512,3), np.uint8)

# mouse callback function
def draw_circle(event,x,y,flags,param):
    if event == cv.EVENT_LBUTTONDBLCLK:
        cv.circle(img,(x,y),100,(255,0,0),-1)

def main():
    cv.namedWindow('image')
    cv.setMouseCallback('image',draw_circle)

    while(1):
        cv.imshow('image',img)

        if cv.waitKey(20) & 0xFF == 27:
            break

    cv.destroyAllWindows()

if __name__ == "__main__":
     main()