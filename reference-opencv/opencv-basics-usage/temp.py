import numpy as np
import cv2

filter = np.full((480,480,3), (0,255,255), dtype= np.uint8)

img1 = np.empty((480,640), dtype= np.uint8) # grayscale image
img2 = np.zeros((480,640,3), dtype= np.uint8) # black image
img3 = np.ones((480, 640), dtype= np.uint8) * 255 # white
img4 = np.full((480,640,3), (0,255,255), dtype= np.uint8) # yellow

img5 = img4.copy() # copy the image

cv2.namedWindow('image')
cv2.imshow('image',mat = filter)
cv2.waitKey()
cv2.destroyAllWindows()