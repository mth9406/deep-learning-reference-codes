import cv2
import sys

src = cv2.imread('./data/rose.bmp', cv2.IMREAD_COLOR)

if src is None:
    print('--(!)Image load failed')
    sys.exit()

dst1 = cv2.resize(src, (0,0), fx=2, fy=2, interpolation= cv2.INTER_NEAREST)
dst2 = cv2.resize(src, (1920//2, 1280//2)) # cv2.INTER_LINEAR
dst3 = cv2.resize(src, (1920//2, 1280//2), interpolation= cv2.INTER_CUBIC)
dst3 = cv2.resize(src, (1920//2, 1280//2), interpolation= cv2.INTER_LANCZOS4)

cv2.imshow('src', src)
cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)
cv2.imshow('dst3', dst3)
cv2.waitKey()
cv2.destroyAllWindows()
