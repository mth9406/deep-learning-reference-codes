import argparse
import cv2
import numpy as np
import sys

def myImread(path, code):
    src = cv2.imread(path, code)
    if src is None:
        print('--(!)Image load failed')
        sys.exit()
    return src

def myContrast(img, alpha):
    # the larger the alpha is, the larger the contrast effect is.
    # if alpha < 1: contrast effect shrinks
    # else: contrast effect gets larger.
    vals, counts = np.unique(img, return_counts= True) 
    idx = np.argmax(counts)
    # vals[idx] = mode

    modified = (1+alpha)*img - alpha * vals[idx]  # externally dividing
    modified = np.clip(modified, 0, 255).astype(np.uint8)
    # print(f'mode: {vals[idx]}')
    return modified

def myContrast2(img, alpha):
    # the larger the alpha is, the larger the contrast effect is.
    # if alpha < 1: contrast effect shrinks
    # else: contrast effect gets larger.

    modified = (1+alpha)*img - alpha * 128  # externally dividing
    modified = np.clip(modified, 0, 255).astype(np.uint8)
    return modified

def histStretching(img):
    G_max, G_min = np.max(img), np.min(img)
    # stretch the image
    modified = (255/(G_max-G_min + 1e-6) * (img - G_min)).astype(np.uint8)
    # maps G_max --> 255 and G_min --> 0
    # contrast effects..!
    return modified

def argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--src', type = str, default= './data/lenna.bmp')
    config = p.parse_args()
    return config

def getGrayHistImage(src):

    hist = cv2.calcHist([src], [0], None, [256], [0,256])

    imgHist = np.full((100, 256), 255, dtype = np.uint8) # white background
    
    histMax = np.max(hist)
    for x in range(256):
        pt1 = (x, 100) # width, height
        pt2 = (x, 100 - int(hist[x, 0]*100 / histMax)) # notice that subtraction yields higher position in height.
        cv2.line(imgHist, pt1, pt2, 0)
    
    return imgHist
    

def main(config):
    src = cv2.imread(config.src, cv2.IMREAD_GRAYSCALE)
    modified = myContrast(src, 0.8)
    modifiedHist = histStretching(src)

    modified_hist = getGrayHistImage(modified)
    modifiedHist_hist = getGrayHistImage(modifiedHist)

    cv2.imshow('src', src)
    
    cv2.imshow('modified', modified)
    cv2.imshow('modified_hist', modified_hist)

    cv2.imshow('modifiedHist', modifiedHist)
    cv2.imshow('modifiedHist_hist', modifiedHist_hist)

    while True:
        if cv2.waitKey() == 27:
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    config = argparser()    
    main(config)