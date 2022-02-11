import cv2
import sys
import numpy as np

def myEqaulizeHist(src, bins = [256]):
    
    modified = src.copy()
    
    hist = cv2.calcHist(src, [0],None, bins, [0,256])
    cdf = np.array([0] * len(hist))
    cdf[0] = hist[0]
    for i in range(1,len(hist)):
        cdf[i] = cdf[i-1]+hist[i]
    cdf = np.array(cdf, dtype=np.uint8)
    values = np.arange(0, 256)
    
    for i, v in enumerate(values):
        modified[modified == v] = cdf[i]

    return modified


src = cv2.imread('./data/field.bmp', cv2.IMREAD_GRAYSCALE)

if src is None:
    print('--(!)Image load failed')
    sys.exit()

dst = cv2.equalizeHist(src)
# modified = myEqaulizeHist(src)

cv2.imshow('src', src)
cv2.imshow('equalized', dst)
# cv2.imshow('modified', modified)

while True:
    if cv2.waitKey() == 27:
        break

cv2.destroyAllWindows()