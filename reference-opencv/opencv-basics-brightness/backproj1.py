import cv2
import numpy as np
import sys
import argparse

def argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--src', type= str, default = './data/cropland.png')
    config = p.parse_args()
    return config

def main(config):
    src = cv2.imread(config.src, cv2.IMREAD_COLOR)
    
    if src is None:
        print('--(!)Image load failed...')
        sys.exit()

    x, y, w, h = cv2.selectROI(src) 
    # we can select ROI using mouse drag
    # w: cols
    # h: rows
    # bbox: [y:y+h, x:x+w]
    src_ycrcb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)
    crop = src_ycrcb[y:y+h, x:x+w]

    # calculates histogram.
    channels = [1, 2] # use cr, cb; exclude y-channel (brightness) 
    cr_bins = 128 
    cb_bins = 128
    histSize= [cr_bins, cb_bins]
    cr_range = [0, 256]
    cb_range = [0, 256]
    ranges = cr_range+cb_range

    hist = cv2.calcHist([crop], channels, None, histSize, ranges)
    hist_norm = cv2.normalize(hist, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # back project 
    backproj = cv2.calc


    
