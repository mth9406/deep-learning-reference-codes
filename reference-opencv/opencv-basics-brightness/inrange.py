from turtle import back
import cv2
import numpy as np

def filterColorRegionHSV(src, lowerb, upperb):
    
    src_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV) 
    mask = cv2.inRange(src_hsv, lowerb, upperb) # mask image
    background = np.full(img.shape, 255, dtype=np.int8)
    mask_img = cv2.copyTo(src, mask, background)
    
    return mask_img, mask


img = cv2.imread('./data/candies.png', cv2.IMREAD_COLOR)
dst1 = cv2.inRange(img, (0,128,0), (100, 255, 100))
background = np.full(img.shape, 255, dtype=np.int8)

mask_img = cv2.copyTo(img, dst1, background)

mask_img_hsv, dst2 = filterColorRegionHSV(img, (50,150,0), (80, 255,255))

cv2.imshow('BGR', dst1)
cv2.imshow('HSV', dst2)

cv2.imshow('BGR IMAGE', mask_img)
cv2.imshow('HSV IMAGE', mask_img_hsv)

while True:
    q = cv2.waitKey()
    if q == 27:
        break
cv2.destroyAllWindows()