'''An example of local threshold algorithm
-> Adaptive threshold 
cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, 
                     thresholdType, blockSize, C, dst =None) -> dst

src: input image
maxValue: usually 255 (max value of threshold function) - refer to cv2.threshold()
adaptiveMethod: how to obtain average value by block 
              : (ex) cv2.ADAPTIVE_THRESH_MEAN, cv2.ADAPTIVE_THRESH_GAUSSIAN_C
thresholdType: cv2.THRESH_BINARY or cv2.THRESH_BINARY_INV ..
blockSize: the size of a block (odd number >= 3)
C: bias term to subtract every average value of a block
'''

import cv2
import sys

src = cv2.imread('./data/rice.png', cv2.IMREAD_GRAYSCALE)
if src is None:
    print('Image load failed')
    sys.exit()

dst = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                            151, 5)

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()