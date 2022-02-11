import cv2
import numpy as np
import argparse

def argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--src', type=str, default = r'./data/field.bmp')
    config = p.parse_args()
    return config

def main(config):
    src = cv2.imread(config.src, cv2.IMREAD_COLOR)

    x, y, w, h = cv2.selectROI(src) # width, height, width, height
    src_ycrcb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)
    crop = src_ycrcb[y:y+h, x:x+w]

    # backprojection
    # apply histogram stretching in YCrCb format.  
    # get histogram.
    channels = [1,2]
    mask = None
    
    histsize = [128,128] # bins.
    ranges = [0,256,0,256]

    hist = cv2.calcHist([crop], channels, mask, histsize, ranges)
    # calculates histogram of the selected RoI.
    # we only use cr, cb channels.
    # hist_norm = cv2.normalize(hist, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    backproj = cv2.calcBackProject([src_ycrcb], channels, hist, ranges, 1)
    dst = cv2.copyTo(src, backproj)
    
    cv2.imshow('backproj', backproj)
    # cv2.imshow('hist_norm', hist_norm)
    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    config = argparser()
    main(config)