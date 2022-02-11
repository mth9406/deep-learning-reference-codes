'''
we use sober filter to obtain dx, dy
and find the total derivative.
'''

import cv2
import numpy as np

src = cv2.imread('./data/lenna.bmp', cv2.IMREAD_GRAYSCALE)

dx = cv2.Sobel(src, ddepth = cv2.CV_32F, dx=1, dy=0)
dy = cv2.Sobel(src, ddepth = cv2.CV_32F, dx=0, dy=1)

grad = dx, dy
mag = np.clip(cv2.magnitude(dx, dy), 0, 255).astype(np.uint8)
# cv2.magnitude's dtype is float

# we can use boolean indexing to extract edges 
edge = np.zeros(src.shape[:2], dtype= np.uint8) # initiate the edge in gray scale
thr = 120 # threshold
edge[mag>thr] = mag[mag>thr]

cv2.imshow('src', src)
cv2.imshow('mag', mag)
cv2.imshow('edge', edge)
cv2.waitKey()
cv2.destroyAllWindows()