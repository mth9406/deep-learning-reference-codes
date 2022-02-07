import numpy as np
import cv2

# ROI:
# Region of Interest 
# Some part of an image where we want to apply 
# some certain operations.

# mask-operation
# OpenCV supports some RoI operations
# which require a mask image

# (ex) cv2.copyTo(), cv2.calcHist(), cv2.matchTemplate(), etc.

# cv2.coptTo(src, mask, dst= None) -> dst
# src: input image
# mask: mask image, copy pixels where the mask value is not 0.
# dst: output image, 
#    : if size and dtype of dst is equal to the src,
#    : does not generate a new image (dst)
#    : else, makes a new image (dst) and applies operation. 

# dtype of a mask image is,
# cv2.CV_8UC1: gray-scale 
# mask-operation is applied where pixel value of a mask image is not 0.
# RoI: (255,255,255) white color in a mask image.

src = cv2.imread('./data/airplane.bmp', cv2.IMREAD_COLOR)
# src = cv2.imread('./data/opencv-logo-white.png', cv2.IMREAD_UNCHANGED)
mask = cv2.imread('./data/mask_plane.bmp', cv2.IMREAD_GRAYSCALE)
dst_original = cv2.imread('./data/field.bmp', cv2.IMREAD_COLOR)
dst = dst_original.copy()

# copy RoI in the src
# (src, mask)
# to the dst
# inplace = True.
cv2.copyTo(src, mask, dst)
# the same as:
# dst[mask>0] = src[mask>0]

background = np.zeros_like(src, dtype= np.uint8)
cv2.copyTo(src, mask, background) # get RoI only

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.imshow('original dst', dst_original)
cv2.imshow('mask', mask)
cv2.imshow('RoI', background)
cv2.waitKey()

cv2.destroyAllWindows()