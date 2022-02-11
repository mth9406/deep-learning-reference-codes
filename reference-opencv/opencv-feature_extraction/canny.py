import cv2
import numpy as np

'''
Canny edge detection function
cv2.Canny(image, threshold1, threshold2, edges= None, apertureSize=None,
          L2gradient= None) -> edge

image: original input image
threshold1, threshold2: weak edge, strong edge respectively
                      : naive values - thr1:thr2 = 1:2 or 1:3
                      : recommendation: 50, 150
edges: edge image
apertureSize: kernel size of Sobel operation
L2gradient: default= False (L1 norm) 
'''
src = cv2.imread('./data/building.jpg', cv2.IMREAD_GRAYSCALE)

edges = cv2.Canny(src, 50, 100)

cv2.imshow('src', src)
cv2.imshow('edges', edges)
cv2.waitKey()
cv2.destroyAllWindows()