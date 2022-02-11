import numpy as np
import cv2

'''
cv2.Sobel(src, ddepth, dx, dy, 
        dst=None, ksize= None, scale= None, delta= None, 
        border_Type=None)

src: input image
ddepth: the same datatype if -1, recommends using cv2.CV_32F for more precise computation
dx : differential coef along the x axis
dy : differential coef along the y axis; 
   : recommends using either dx = 1, dy =0 or dx = 0, dy = 1
dst: destination image
ksize: default = 3 (recommended)
scale: value to multiply to the 'raw' Sobal output
delta: bias to add to the 'raw' Sobal output --> to have a better look.
borderType: default= cv2.BORDER_DEFAULT (REFLECT_101)
'''
src = cv2.imread('./data/lenna.bmp', cv2.IMREAD_GRAYSCALE)
sobel = cv2.Sobel(src, ddepth= -1, dx=1, dy=0, delta= 128).astype(np.uint8) 
# we use, delta = 128 to prevent saturation where pixels show gray to black patterns.
# In addition, black curves (lines) in the sobel filtered image indicates
# where the pixel values dramatically shifted.

dx = cv2.Sobel(src, ddepth= cv2.CV_32F, dx=1, dy=0, delta= 0)
dy = cv2.Sobel(src, ddepth= cv2.CV_32F, dx=0, dy=1, delta = 0)

# grad = dx, dy
mag = np.clip(cv2.magnitude(dx, dy), 0, 255).astype(np.uint8)

cv2.imshow('src', src)
cv2.imshow('sobel', sobel) # when dst dtype is the same with src dtype.
cv2.imshow('dx', dx)
cv2.imshow('dy', dy)
cv2.imshow('magnitude of gradient', mag)

cv2.waitKey()
cv2.destroyAllWindows()