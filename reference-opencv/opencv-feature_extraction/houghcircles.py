'''Circle detection
cv2.HoughCircles(
    image, method, dp, minDist,
    circles= None, param1= None,
    param2= None, minRadius= None,
    maxRadius = None
) -> circles

image: original image (NOT EDGE)
method: only cv2.HOUGH_GRADIENT (OpenCV <= 4.2)
dp: the size of accumulatio array is the same as the 
  : origianl image if 1
  : elif 2, the array is the half of the image.
minDist: min distance between centers
circles: (cx, cy, r), (1, N, 3), dtype= np.float32
param1, param2: Thr_upper, Thr_lower in the Canny algorithm
minRadius, maxRadius: min, max readius of a circle to detect

'''
import sys
import numpy as np
import cv2


# 입력 이미지 불러오기
src = cv2.imread('./data/dial.jpg')

if src is None:
    print('Image open failed!')
    sys.exit()

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
blr = cv2.GaussianBlur(gray, (0, 0), 1.0)


def on_trackbar(pos):
    rmin = cv2.getTrackbarPos('minRadius', 'img')
    rmax = cv2.getTrackbarPos('maxRadius', 'img')
    th = cv2.getTrackbarPos('threshold', 'img')

    circles = cv2.HoughCircles(blr, cv2.HOUGH_GRADIENT, 1, 50,
                               param1=120, param2=th, minRadius=rmin, maxRadius=rmax)

    dst = src.copy()
    if circles is not None:
        for i in range(circles.shape[1]):
            cx, cy, radius = circles[0][i]
            cv2.circle(dst, (cx, cy), int(radius), (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('img', dst)


# 트랙바 생성
cv2.imshow('img', src)
cv2.createTrackbar('minRadius', 'img', 0, 100, on_trackbar)
cv2.createTrackbar('maxRadius', 'img', 0, 150, on_trackbar)
cv2.createTrackbar('threshold', 'img', 0, 100, on_trackbar)
cv2.setTrackbarPos('minRadius', 'img', 10)
cv2.setTrackbarPos('maxRadius', 'img', 80)
cv2.setTrackbarPos('threshold', 'img', 40)
cv2.waitKey()

cv2.destroyAllWindows()




# import cv2
# import numpy as np
# import sys

# src = cv2.imread('./data/dial.jpg')
# if src is None:
#     print('Image load failed')
#     sys.exit()

# gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
# # (Recommendation) blur the image before using HoughCircles
# blur = cv2.GaussianBlur(src, (0,0), 1.0)

# def on_trackbar(pos):
#     rmin = cv2.getTrackbarPos('minRadius', 'img')
#     rmax = cv2.getTrackbarPos('maxRadius', 'img')
#     th = cv2.getTrackbarPos('threshold', 'img')

#     circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 50,
#                                param1=120, param2=th, 
#                                minRadius=rmin, maxRadius=rmax)
    
#     dst = src.copy()
    
#     if circles is not None:
#         for i in range(circles.shape[1]):
#             cx, cy, r = circles[0][i]
#             cv2.circle(dst, (cx, cy), int(r), (0,0,255), 2, cv2.LINE_AA)

#     cv2.imshow('img', dst)

# # initiate a track bar
# cv2.imshow('img', src)
# cv2.createTrackbar('minRadius', 'img', 0, 100, on_trackbar)
# cv2.createTrackbar('maxRadius', 'img', 0, 150, on_trackbar)
# cv2.createTrackbar('threshold', 'img', 0, 100, on_trackbar)

# cv2.setTrackbarPos('minRadius', 'img', 10)
# cv2.setTrackbarPos('maxRadius', 'img',80)
# cv2.setTrackbarPos('threshold', 'img',40)

# cv2.waitKey()
# cv2.destroyAllWindows()