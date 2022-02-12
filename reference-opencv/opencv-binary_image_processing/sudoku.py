import cv2
import sys

def onChange(pos):

    thr = cv2.getTrackbarPos('threshold', 'dst')
    _, dst = cv2.threshold(src, thr, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('dst', dst)

src = cv2.imread('data/sudoku.jpg', cv2.IMREAD_GRAYSCALE)

if src is None:
    print('Image load failed')
    sys.exit()


cv2.imshow('src', src)

cv2.namedWindow('dst')
cv2.createTrackbar('threshold', 'dst', 100, 255, onChange)

while True:
    query = cv2.waitKey()
    if query == 27:
        break

cv2.destroyAllWindows()