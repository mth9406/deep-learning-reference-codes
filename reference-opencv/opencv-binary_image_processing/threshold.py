'''Threshold
cv2.threshold(src, thresh, maxval, type, dst= None) 
-> retval, dst

src: input image
thresh: user specified threshold
maxval: maxval when either cv2.THRESH_BINARY or cv2.THRESH_BINARY_IN
        is used (recommended value: 255)
type: cv2.THRESH_* 
* cv2.THRESH_BINARY:
dst(x,y) = maxval if src(x,y) > thresh else 0
* cv2.THRESH_BINARY_INV:
dst(x,y) = 0 if src(x,y) > thresh else maxval 

retval: used thresh (usually not used)
dst: dst image

'''

import cv2
import numpy as np
import sys

src = cv2.imread('./data/cells.png', cv2.IMREAD_GRAYSCALE)

if src is None:
    print('Image laod failed')
    sys.exit()

_, dst1 = cv2.threshold(src, 100, 255, cv2.THRESH_BINARY)
# detect white-like colors
_, dst2 = cv2.threshold(src, 100, 255, cv2.THRESH_BINARY_INV)
# detect gray(black)-like colors
cv2.imshow('src', src)
cv2.imshow('binary', dst1)
cv2.imshow('binary_inv', dst2)
cv2.waitKey()
cv2.destroyAllWindows()