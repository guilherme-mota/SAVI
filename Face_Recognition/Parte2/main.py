#!/usr/bin/env python3

#-----------
# Imports
#-----------
from curses.textpad import rectangle
import os
import cv2
import numpy as np
import face_recognition


#-----------
# Functions
#-----------
def findEncodings(images):
    encode_list = []

    # Cycle through all images in list
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)

    return encode_list

def main():

    # Variables
    path = 'Images'
    images = []
    class_names = []

    # Read files in path
    list_of_files = os.listdir(path)
    print('List of Files: ' + str(list_of_files))

    # Cycle through all files in the directory
    for file in list_of_files:
        current_image = cv2.imread(f'{path}/{file}')
        images.append(current_image)  # add image to the list
        class_names.append(os.path.splitext(file)[0])  # get image name

    # Find images encodings
    encode_list = findEncodings(images)
    print('Endoding Complete!')

    # Start Video Capture
    capture = cv2.VideoCapture(0)

    while True:
        success, image_original = capture.read()
        image_gui = cv2.resize(image_original, (0, 0), None, 0.25, 0.25)
        image_gui = cv2.cvtColor(image_gui, cv2.COLOR_BGR2RGB)

        face_current_frame = face_recognition.face_locations(image_gui)
        encode_current_frame = face_recognition.face_encodings(image_gui, face_current_frame)

        # Cycle through all detections
        for encode_face, face_location in zip(encode_current_frame, face_current_frame):
            # Compare face detected with list of faces known
            matches = face_recognition.compare_faces(encode_list, encode_face)
            face_distances = face_recognition.face_distance(encode_list, encode_face)

            # Get index of lowest distance
            match_index = np.argmin(face_distances)

            if matches[match_index]:
                name = class_names[match_index].upper()
                print(name)

                # Get bounding box location
                y1, x2, y2, x1 = face_location
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

                # Draw rectangle
                cv2.rectangle(image_original, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(image_original, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(image_original, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        # Display Image
        cv2.imshow('Webcam', image_original)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()