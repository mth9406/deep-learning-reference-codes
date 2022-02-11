import cv2

src = cv2.imread('./data/lenna.bmp', cv2.IMREAD_COLOR)

# adapt bilateral filtering
blurred = cv2.bilateralFilter(src, -1, 4, 2)
# src
# d= -1: automatically determine the filter size (recommended)
# sigmaColor = std. in the color space, recommend 10~20
# sigmaSpace = std. in the pixel space, recommend 2~4, 
# if filter size is too large, the processing time will be too slow.

cv2.imshow('src', src)
cv2.imshow('blurred', blurred)
cv2.waitKey()
cv2.destroyAllWindows()