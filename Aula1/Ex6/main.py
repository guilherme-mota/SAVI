#!/usr/bin/env python3

# OpenCV Tutorial - Basic Operations on Images

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def main():
    img = cv.imread('/home/guilherme/SAVI/Aula1/Ex1/UA.jpg')

    px = img[100,100]
    print( px )

    # accessing only blue pixel
    blue = img[100,100,0]
    print( blue )

    img[100,100] = [255,255,255]
    print( img[100,100] )

    # accessing RED value
    print( img.item(10,10,2))

    # modifying RED value
    img.itemset((10,10,2),100)
    print( img.item(10,10,2))

    print( img.shape )
    print( img.size )
    print( img.dtype )

    # Making Borders for Images (Padding) 
    BLUE = [255,0,0]

    img1 = cv.imread('opencv-logo.png')

    replicate = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_REPLICATE)
    reflect = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_REFLECT)
    reflect101 = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_REFLECT_101)
    wrap = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_WRAP)
    constant= cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_CONSTANT,value=BLUE)

    plt.subplot(231),plt.imshow(img1,'gray'),plt.title('ORIGINAL')
    plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
    plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
    plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
    plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
    plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')
    
    plt.show()

if __name__ == "__main__":
     main()