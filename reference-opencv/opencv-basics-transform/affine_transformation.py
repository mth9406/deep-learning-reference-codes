import cv2
import numpy as np

# source image
src = cv2.imread('./data/cat.bmp', cv2.IMREAD_COLOR)
h, w = src.shape[:2]

src_pts = np.array([[0,0],[w,0],[0,h]], dtype= np.float32)
dst_pts = np.array([[0,0],[w,10],[10,h]], dtype= np.float32)

# affine matrix
M = cv2.getAffineTransform(src_pts,dst_pts)
dst = cv2.warpAffine(src, M, dsize= (0,0))

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()