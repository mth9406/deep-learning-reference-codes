import cv2
import sys
import argparse
import numpy as np

def argparser():
    p = argparse.ArgumentParser()
    
    p.add_argument('--src', type= str, default= r'./data/rice.png',
                    help= 'a path to source image')
    p.add_argument('--num_patch', type= int, default = 4,
                    help= 'the number of pathes to make')

    config = p.parse_args()
    return config


def main(config):

    src = cv2.imread(config.src, cv2.IMREAD_GRAYSCALE)

    if src is None:
        print('Image laod failed')
        sys.exit()

    # split the data into 'num_patches' patches
    h, w = src.shape[:2]
    dh, dw = h//config.num_patch, w//config.num_patch

    dst = np.zeros(src.shape, dtype= np.uint8)

    for x in range(config.num_patch+1):
        for y in range(config.num_patch+1):
            starty, endy = dh*y, min(dh*(y+1), h)
            startx, endx = dw*x, min(dw*(x+1), w)

            if endy == starty and endx == startx:
                continue
            _, img = cv2.threshold(src[starty:endy, startx:endx],
                        0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
            dst[starty:endy, startx:endx] = img
    
    cv2.imshow('src', src)
    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    config = argparser()
    main(config)