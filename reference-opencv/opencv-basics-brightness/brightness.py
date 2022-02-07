import cv2
import numpy as np
import sys
import argparse

def arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--src_path', type= str, required= True)
    p.add_argument('--n', type= float, default= 50)
    config = p.parse_args()
    return config

def brighten(img, n):
    img_arr = img.copy()
    dims = len(img_arr.shape)
    if dims < 2:
        img_arr = np.clip(img_arr + n, 0, 255)
    else:
        img_arr = np.clip(img_arr + np.array([n for _ in range(img_arr.shape[-1])]), 0, 255)
    # uint --> larger than 255 --> 0
    #      --> smaller than 0 --> 255
    # img_arr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    return img_arr

def main(config):
    
    img = cv2.imread(config.src_path, cv2.IMREAD_GRAYSCALE)
    
    # notice that config.n is float type.
    img_arr = brighten(img, config.n).astype(np.uint8)
    
    dims = len(img.shape)
    levels = config.n if dims > 2 else tuple([config.n] * 4)
    converted_img = cv2.add(img, levels)
    # if image is in color
    # converted_img = cv2.add(img, (config.n, config.n, config.n, 0))
    # print(img_arr - img)

    # print(img_arr - img)

    cv2.imshow('Image', img)
    cv2.imshow('Converted image', img_arr)
    cv2.imshow('Converted image2', converted_img)
    while True:
        query = cv2.waitKey()
        if query == ord('q') or query == 27:
            break
        elif query == ord('i') or query == ord('I'):
            img = ~img
            cv2.imshow('Image', img)
            img_arr = ~img_arr
            cv2.imshow('Converted image', img_arr)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    config = arg_parser()
    main(config)