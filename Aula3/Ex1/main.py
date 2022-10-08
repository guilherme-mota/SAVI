#!/usr/bin/env python3

import cv2
import numpy as np

def main():
    # ------------------------------------------
    # Initialization
    # ------------------------------------------
    blackout_time = 0.5  # secs without detecting a car
    threshold_difference = 20  # max diference between average and module

    # List of Rectangles (define only once)
    rects = [{'name':'r1', 'x1':400, 'y1':500, 'x2':600, 'y2':600, 'ncars':0, 'tic_since_car_count': -500},
             {'name':'r2', 'x1':700, 'y1':500, 'x2':900, 'y2':600, 'ncars':0, 'tic_since_car_count': -500}]

    # read video from file
    cap = cv2.VideoCapture("/home/guilherme/savi_22-23/Parte03/docs/traffic.mp4")

    if (cap.isOpened() == False):
        print("Error opening video from stream or file")

    is_first_time = True

    # ------------------------------------------
    # Execution
    # ------------------------------------------
    while cap.isOpened():
        # Step 1: Get Frame
        ret, image_rgb = cap.read()  # get a frame, ret will be true or false if getting succeeds

        if ret == False:
            break

        stamp = float(cap.get(cv2.CAP_PROP_POS_MSEC))/1000

        # Step 2: Convert to Gray
        image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)

        # Step 3: Get Average Color in Rectangle
        for rect in rects:
            total = 0
            number_of_pixels = 0

            for row in range(rect['y1'], rect['y2']):  # run for each row
                for col in range(rect['x1'], rect['x2']):  # run for each column
                    number_of_pixels += 1
                    total += image_gray[row, col]  # add pixel color to the total count

            # after computing the total we should divide to get the average
            rect['avg_color'] = int(total/number_of_pixels)

            # How to get the model average? We know that in the first frame there are no cars in the 
            # rectangles. The first measurement is the model average
            if is_first_time:
                rect['model_avg_color'] = rect['avg_color']

            # Compute the difference in color and make a decision
            diff = abs(rect['avg_color'] - rect['model_avg_color'])

            if diff > 20 and (stamp - rect['tic_since_car_count']) > blackout_time:
                rect['ncars'] = rect['ncars'] + 1
                rect['tic_since_car_count'] = stamp

        is_first_time = False

        # Drawing --------------------------
        for rect in rects:
            # Draw Green Rectangles
            cv2.rectangle(image_rgb, (rect['x1'],rect['y1']), (rect['x2'],rect['y2']), (0,255,0),2)  # BGR

            # Add text with average and model color
            text = 'avg=' + str(rect['avg_color']) + 'm=' + str(rect['model_avg_color'])
            image_rgb = cv2.putText(image_rgb, text, (rect['x1'], rect['y1']-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

            # Add text with number of cars
            text = 'ncars=' + str(rect['ncars'])
            image_rgb = cv2.putText(image_rgb, text, (rect['x1'], rect['y1']-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

            # Add text time since last car count
            text = 'Time since lcc=' + str(round(stamp - rect['tic_since_car_count'],1))  + ' secs'
            image_rgb = cv2.putText(image_rgb, text, (rect['x1'], rect['y1']-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1, cv2.LINE_AA)

        cv2.imshow('image_rgb',image_rgb) # show the image
        #cv2.imshow('image_gray',image_gray) # show the image

        if cv2.waitKey(50) == ord('q'):
            break

    # ------------------------------------------
    # Termination
    # ------------------------------------------
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()