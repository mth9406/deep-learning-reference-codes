import cv2
import numpy as np

# source image
src = cv2.imread('./data/cat.bmp', cv2.IMREAD_COLOR)
h, w = src.shape[:2]

src_pts = np.array([[0,0],[w,0],[0,h], [w,h]], dtype= np.float32)
dst_pts = np.array([[0,0],[w,+50],[+30,h], [w-10,h-20]], dtype= np.float32)

# affine matrix
M = cv2.getPerspectiveTransform(src_pts,dst_pts)
dst = cv2.warpPerspective(src, M, dsize= (0,0))

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()