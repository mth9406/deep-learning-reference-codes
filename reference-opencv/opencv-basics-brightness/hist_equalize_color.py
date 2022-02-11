import cv2
import numpy as np

def myEqaulizeColor(src):    
    modified  = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)
    modified[:,:,0] = cv2.equalizeHist(modified[:,:,0]) # apply equalizeHist only to the bright dimension.
    modified = cv2.cvtColor(modified, cv2.COLOR_YCrCb2BGR)
    return modified

src = cv2.imread('./data/field.bmp', cv2.IMREAD_COLOR)
modified = myEqaulizeColor(src)

cv2.imshow('src', src)
cv2.imshow('modified', modified)
cv2.waitKey()
cv2.destroyAllWindows()