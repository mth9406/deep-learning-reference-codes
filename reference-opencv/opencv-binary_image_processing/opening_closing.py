'''Opening and Closing

cv2.morphologyEx(src, op, kernel, 
              dst= None, anchor= None, iterations= None,
              borderType= None, borderValue= None) -> dst

src: input image
op: morphology operation flag
  : (ex) 
  : cv2.MORPH_ERODE: erosion
  : cv2.MORPH_DILATE: dilation
  : cv2.MORPH_OPEN: opening
  : cv2.MORPH_CLOSE: closing
  : cv2.MORPH_GRADIENT: dilation - erosion
kernel: kernel
dst: destination image

'''

import cv2
import sys
import numpy as np
from sklearn.preprocessing import binarize

def localThreshold(src, num_pathes):

    h, w = src.shape[:2]
    dh, dw = h//num_pathes, w//num_pathes
    dst = np.zeros(src.shape[:2], dtype=np.uint8)
    
    for x in range(num_pathes+1):
        for y in range(num_pathes+1):
            startX, endX, startY, endY \
                = dw*x, min(dw*(x+1), w), dh*y, min(dh*(y+1), h)
            cv2.threshold(src[startY:endY, startX:endX], 0, 255, 
                        type= cv2.THRESH_BINARY|cv2.THRESH_OTSU,
                        dst= dst[startY:endY, startX:endX])
    
    return dst

src = cv2.imread('./data/rice.png', cv2.IMREAD_GRAYSCALE)
if src is None:
    print('Image load failed')
    sys.exit()

# binarize before opening the image
# _, binarized = cv2.threshold(src, 128, maxval= 255, type= cv2.THRESH_BINARY)

binarized = localThreshold(src, 7)

opened = cv2.morphologyEx(binarized, cv2.MORPH_OPEN, kernel= None)
# equivalent to:
# dst2 = cv2.erode(binarized, None)
# dst2 = cv2.dilate(dst2, None)

denoised = np.zeros(src.shape[:2], dtype= np.uint8)
denoised[opened>0] = src[opened>0]

cv2.imshow('src', src)
cv2.imshow('binarized',binarized)
cv2.imshow('opened', opened)
while True:
    q = cv2.waitKey()
    if q == 27:
        break
cv2.destroyWindows()