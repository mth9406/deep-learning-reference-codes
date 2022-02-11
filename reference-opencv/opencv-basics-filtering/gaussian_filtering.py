import cv2
import argparse
import numpy as np
import sys

def argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--src', type= str, default= r'./data/rose.bmp')
    p.add_argument('--kernel_size', type= int, default= 0)
    p.add_argument('--imread_type', type= int, default= 0)
    p.add_argument('--sigmaX', type= float, default= 1.0)
    p.add_argument('--sigmaY', type= float, default= 1.0)    
    config = p.parse_args()

    return config

def imread(src, readtype):
    if readtype == 0:
        img = cv2.imread(src, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print('--(!)Image load failed...')
        sys.exit()
    return img

def main(config):
    # read the src image
    src = imread(config.src, config.imread_type)
    
    # apply mean filtering.
    dst = cv2.GaussianBlur(src, (config.kernel_size, config.kernel_size),
                            sigmaX= config.sigmaX, sigmaY= config.sigmaY,
                            borderType= cv2.BORDER_REFLECT_101)
    # if kernel-size = (0,0) --> it uses automatic kernel size.
    
    # show the image
    cv2.imshow('src', src)
    cv2.imshow('dst', dst)

    while True:
        if cv2.waitKey() == 27:
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    config = argparser()
    main(config)


    