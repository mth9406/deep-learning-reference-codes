import cv2

src = cv2.imread('./data/cat.bmp', cv2.IMREAD_COLOR)

rc = (250, 120, 200, 200) # (x, y, width, height)
cpy= src.copy()
cv2.rectangle(cpy, rc, (0,0,255), 2)
cv2.waitKey()

for i in range(1, 4):
    src = cv2.pyrDown(src)
    cpy = src.copy()
    cv2.rectangle(cpy, rc, (0,0,255), 2, shift= i) # rc 좌표를 i배 shrink.
    cv2.imshow('src', cpy)
    cv2.waitKey()
    cv2.destroyWindow('src')
