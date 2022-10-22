#!/usr/bin/env python3

#-----------
# Imports
#-----------
import cv2
import numpy as np
import face_recognition


def main():

    # Import Images
    img_elon = face_recognition.load_image_file('ImagesBasic/elon_musk.jpg')
    img_test = face_recognition.load_image_file('ImagesBasic/elon_test.jpg')
    img_bill = face_recognition.load_image_file('ImagesBasic/bill_gates.jpeg')

    # Convert Image from BGR to RGB
    img_elon = cv2.cvtColor(img_elon, cv2.COLOR_BGR2RGB)
    img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)
    img_bill = cv2.cvtColor(img_bill, cv2.COLOR_BGR2RGB)

    # Face Localization
    face_localization = face_recognition.face_locations(img_elon)[0]
    face_localization_test = face_recognition.face_locations(img_test)[0]

    # Get Face Encoding
    encode_elon = face_recognition.face_encodings(img_elon)[0]
    encode_elon_test = face_recognition.face_encodings(img_test)[0]
    encode_bill = face_recognition.face_encodings(img_bill)[0]

    # Compare Encodings
    results = face_recognition.compare_faces([encode_elon], encode_bill)
    face_distance = face_recognition.face_distance([encode_elon], encode_bill)

    # Put text in images
    cv2.putText(img_bill, f'{results} {round(face_distance[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    # Prints
    print('Results: ' + str(results))
    print('Face Distance: ' + str(face_distance))

    # Draw rectangles around faces
    cv2.rectangle(img_elon, (face_localization[3], face_localization[0]), (face_localization[1], face_localization[2]), (255, 0, 255), 2)
    cv2.rectangle(img_test, (face_localization_test[3], face_localization_test[0]), (face_localization_test[1], face_localization_test[2]), (255, 0, 255), 2)

    # Show Images
    cv2.imshow('Elon Musk', img_elon)
    cv2.imshow('Elon Musk Test', img_bill)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()