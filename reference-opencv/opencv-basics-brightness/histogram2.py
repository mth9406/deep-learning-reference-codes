import cv2  
import sys
import numpy as np

def myImread(path, code):
    src = cv2.imread(path, code)
    if src is None:
        print('--(!)Image load failed')
        sys.exit()
    return src

def getGrayHistImage(hist):
    imgHist = np.full((100, 256), 255, dtype = np.uint8) # white background
    
    histMax = np.max(hist)
    for x in range(256):
        pt1 = (x, 100) # width, height
        pt2 = (x, 100 - int(hist[x, 0]*100 / histMax)) # notice that subtraction yields higher position in height.
        cv2.line(imgHist, pt1, pt2, 0)
    
    return imgHist

src = myImread('./data/lenna.bmp', cv2.IMREAD_GRAYSCALE)

hist = cv2.calcHist([src], [0], None, [256], [0,256])

histImg = getGrayHistImage(hist)

cv2.imshow('src', src)
cv2.imshow('histImg', histImg)
while True:
    query = cv2.waitKey()
    if query == 27:
        break
cv2.destroyAllWindows()