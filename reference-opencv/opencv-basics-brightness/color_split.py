import cv2
import numpy as np
import argparse
import sys

def argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--src_path', type= str, default= './data/candies.png')
    config = p.parse_args()
    return config

def main(config):
    # read the image
    src = cv2.imread(config.src_path, cv2.IMREAD_COLOR)

    if src is None:
        print('--(!)Image load failed')
        sys.exit()

    hue = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

    planes = cv2.split(hue) # Each plane: B, G, R
    # check color image property
    print('src.shape', src.shape) # 480, 640, 3
    print('src.dtype', src.dtype) # uint8

    cv2.imshow('src', src)
    for i, color in enumerate(['B','G','R']):
        # all of them are represented in 'Gray-scale'.
        cv2.imshow(f'plane - {color}', planes[i])
    cv2.imshow('hsv', hue)
    cv2.waitKey()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    config = argparser()
    main(config)
