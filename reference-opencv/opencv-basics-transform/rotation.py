import cv2
import sys
import numpy as np
from sklearn.preprocessing import scale

src = cv2.imread('./data/tekapo.bmp')

if src is None:
    print('Image load failed')
    sys.exit()

center =  np.array(src.shape[:2])/2
angle = -20
sc = 0.5 # scale

rotMatrix = cv2.getRotationMatrix2D(center, angle, sc)
rotated = cv2.warpAffine(src, rotMatrix, (0,0), 
                        borderMode= cv2.BORDER_CONSTANT, borderValue=0)
# borderMode= cv2.BORDER_CONSTANT, borderValue=0; default settings

cv2.imshow('rotated', rotated)
cv2.waitKey()
cv2.destroyAllWindows()