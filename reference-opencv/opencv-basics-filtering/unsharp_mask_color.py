import cv2
import argparse
import numpy as np
import sys

def imread(src, readtype):
    img = cv2.imread(src, readtype)
    if img is None:
        print('--(!)Image load failed...')
        sys.exit()
    return img

# def unsharpFilter(src, unsharp_src):
#     # sharpened image
#     # = saturate(2*src - unsharp_src)
#     sharp_src = cv2.addWeighted(src,1,unsharp_src,-1, 2)
#     modified = cv2.addWeighted(src, 2, unsharp_src, -1, 0)
#     # modified = np.clip(2*src-unsharp_src,0,255).astype(np.uint8)
#     return modified, sharp_src

def unsharpFilterColor(src, alpha, radius):
    # use YCrCb to control the contrast.
    # only Y(brightness) should be sharpened. 
    src_ycrcb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(src_ycrcb)
    blurred = cv2.GaussianBlur(y, (0,0), radius)
    y_modified = cv2.addWeighted(y, 1+alpha, blurred, -alpha, 0)
    modified = cv2.merge([y_modified, cr, cb])

    # convert modified to the BGR format.
    modified = cv2.cvtColor(modified, cv2.COLOR_YCrCb2BGR)
    return modified

# the above implementation is equivalent to.
def unsharpFilterColor(arc, alpha, radius):
    src_ycrcb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(src_ycrcb)
    y = y.astype(np.float32) # float32
    blurred = cv2.GaussianBlur(y, (0,0), radius) # float32
    y_modified = np.clip((1+alpha)*y - alpha*blurred, 0, 255).astype(np.uint8)
    modified = cv2.merge([y_modified, cr, cb])
    
    modified = cv2.cvtColor(modified, cv2.COLOR_YCrCb2BGR)
    return modified

src = imread('./data/rose.bmp', cv2.IMREAD_COLOR)

modified = unsharpFilterColor(src, 1., 2.)

cv2.imshow('src', src)
cv2.imshow('modified', modified)

while True:
    query = cv2.waitKey()
    if query == 27:
        break
cv2.destroyAllWindows()

