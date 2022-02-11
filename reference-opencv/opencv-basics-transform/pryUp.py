import cv2
import numpy as np

src = cv2.imread('./data/cat.bmp', cv2.IMREAD_COLOR)

rc = np.array([250, 120, 200, 200]) # (x, y, width, height)
cpy= src.copy()
cv2.rectangle(cpy, rc, (0,0,255), 2)
cv2.waitKey()

for i in range(1, 4):
    src = cv2.pyrUp(src)
    cpy = src.copy()
    rc = 2*rc
    cv2.rectangle(cpy, rc, (0,0,255), 2)
    cv2.imshow('src', cpy)
    cv2.waitKey()
    cv2.destroyWindow('src')