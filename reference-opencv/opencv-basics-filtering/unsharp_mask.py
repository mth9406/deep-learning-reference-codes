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
#     
#     # modified = np.clip(2*src-unsharp_src,0,255).astype(np.uint8)
#     # if np.clip were tobe used, image dtype should be in float type.
#     return modified, sharp_src

def unsharpFilter(src, alpha, radius):
    # alpha: the larger alpha is, the larger contrast would be.
    #      : weight corresponding to a blurred image.
    # radius: sigmaX of the cv2.GaussianBlur. (smoothing effect controller)
    blurred = cv2.GaussianBlur(src, (0,0), radius)
    sharp_src = cv2.addWeighted(src, 1, blurred, -1, 0)
    modified = cv2.addWeighted(src, 1+alpha, blurred, -alpha, 0)
    return modified, sharp_src

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

src = imread('./data/rose.bmp', cv2.IMREAD_GRAYSCALE)

# unsharp_src
unsharp_src = cv2.GaussianBlur(src, (0,0), 2, borderType=cv2.BORDER_REFLECT_101)
modified, sharp_src = unsharpFilter(src, 1., 2.)

cv2.imshow('src', src)
cv2.imshow('modified', modified)
cv2.imshow('sharp part', sharp_src)

while True:
    query = cv2.waitKey()
    if query == 27:
        break
cv2.destroyAllWindows()

