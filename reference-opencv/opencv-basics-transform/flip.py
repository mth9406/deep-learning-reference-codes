import cv2
import numpy as np
import sys

src = cv2.imread('./data/tekapo.bmp', cv2.IMREAD_COLOR)

if src is None:
    print('Image load failed')
    sys.exit()

flipcode= 0 
# flipcode: 
# 0: up-down conversion
# 1: left-right conversion
# -1: left-rigth, up-down conversion
flipped = cv2.flip(src, flipcode)

cv2.imshow('src', src)
cv2.imshow('flipped', flipped)
cv2.waitKey()
cv2.destroyAllWindows()