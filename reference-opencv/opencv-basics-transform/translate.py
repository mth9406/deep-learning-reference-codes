import sys
import cv2
import numpy as np
import sys

src = cv2.imread('./data/tekapo.bmp', cv2.IMREAD_COLOR)

if src is None:
    print('Image load failed')
    sys.exit()

shift = np.array([10,20], dtype= np.float32).reshape(-1,1) # (2, 1), 10 right, 20 down: 
aff = np.concatenate([np.eye(2), shift], axis=1) # (2, 3)

translated = cv2.warpAffine(src, aff, (0,0))
# [x, y] <- [[1, 0, 10], [0, 1, 20]]@[ [x], [y]]

cv2.imshow('src', src)
cv2.imshow('translated', translated)
cv2.waitKey()
cv2.destroyAllWindows()