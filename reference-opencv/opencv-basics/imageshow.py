import cv2
import argparse
import matplotlib.pyplot as plt
from glob import glob
import os
import numpy as np

def read_image(full_path):
    img_array = np.fromfile(full_path, np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--path', type= str, 
                 default= r'C:\Users\User\OneDrive\바탕 화면\getDrill\reference-opencv\opencv-basics\images')
    config = p.parse_args()
    return config

def main(config):
    img_paths = glob(os.path.join(config.path, '*.jpg'))

    for path in img_paths:
        img = read_image(path)
        if img is None:
            print(f'Image from {path} does not exists')
            continue
        cv2.namedWindow('Image from {path}', cv2.WINDOW_NORMAL)
        cv2.imshow('image', img)

        while True:
            if cv2.waitKey() == ord('q'):
                break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    config = define_argparser()
    main(config)