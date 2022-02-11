from tkinter import W
import cv2
import numpy as np
import sys

src = cv2.imread('./data/tekapo.bmp', cv2.IMREAD_COLOR)
h, w = src.shape[:2]

if src is None:
    print('Image load failed')
    sys.exit()

m = 0.2 # shear transform in width
n = 0.5 # shear transfrom in height
dsize= (w+int(h*m), h+int(w*n))

aff = np.array([[1, m, 0], [n, 1, 0]], dtype=np.float32)
translated = cv2.warpAffine(src, aff, dsize)
# [x, y] <- [[1, 0, 10], [0, 1, 20]]@[ [x], [y]]

cv2.imshow('src', src)
cv2.imshow('translated', translated)
cv2.waitKey()
cv2.destroyAllWindows()