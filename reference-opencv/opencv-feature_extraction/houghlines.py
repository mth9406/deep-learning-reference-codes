'''Hough transformation --> line detection

(1) cv2.HoughLines(image, rho, theta, threshold, 
                lines= None, srn= None, stn= None,
                min_theta= None, max_theta=None)->lines

image: input edge image
rho: 'rho' interval in the accumulation array
   : 1.0 -> one pixel interval
theta: 'theta' interval in the accumulation array
     : np.pi/180 -> 1 degree interval
threshold: threshold to determine a line
lines: line parameters np.ndarray 
     : shape= (N, '1', 2), dtyep = np.float32
srn, stn: rho resolution in multi scale hough transformation
min_theta, max_theta: min, max theta to detect a line

(2) cv2.HoughLinesP(
    image, rho, theta, threshold,
    lines= None, minLineLength=None,
    maxLineGap=None
) -> lines
* P represents 'Probability'
image: input edge image
rho: 'rho' interval in the accumulation array
   : 1.0 -> one pixel interval
theta: 'theta' interval in the accumulation array
     : np.pi/180 -> 1 degree interval
threshold: threshold to determine a line
lines: starting and end points of a segment.
     : shape= (N, 1, 4), np.int32
minLineLength: min length of a segment
maxLineGap: min gap of a line
'''
import cv2
import sys
import numpy as np

src = cv2.imread('./data/building.jpg', cv2.IMREAD_GRAYSCALE)
if src is None:
    print('Image load failed')
    sys.exit()

# first obtain an edge
edges = cv2.Canny(src, 50, 150)

lines = cv2.HoughLinesP(edges, 1.0, np.pi/180, 160, 
                        minLineLength=50, maxLineGap=5)

dst = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

if lines is not None:
    for line in lines:
        pt1 = line[0][0], line[0][1] # stating point
        pt2 = line[0][2], line[0][3] # end point
        cv2.line(dst, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()