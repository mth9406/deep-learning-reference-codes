import sys
import cv2
from glob import glob
import argparse
import os
import numpy as np

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--path', type= str, default = r'C:\Users\User\OneDrive\바탕 화면\getDrill\reference-opencv')
    p.add_argument('--flag', type= int, default= 0,
                    help= '1: cv2.IMREAD_COLOR, 0:cv2.IMREAD_GRAYSCALE, -1:cv2.IMREAD_UNCHANGED')
    config = p.parse_args()

    return config

def read_image(full_path, config):
    img_array = np.fromfile(full_path, np.uint8)
    if config.flag == 1:
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    elif config.flag == 0:
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
    return img

def main(config):

    # # prints an image.
    # img_paths = glob(os.path.join(config.path,'*.bmp'))
    # images = {'image':[], 'name':[]}
    # for img_path in img_paths:
    #     # read the image from the path.
    #     image = read_image(img_paths, config)
    #     images['image'].append(image)
    #     winname = f'Image from {img_path}'
    #     images['name'].append(winname)
    #     # show the image
    #     cv2.namedWindow(winname)
    #     cv2.imshow('image', image)
    #     cv2.waitKey()
    #     cv2.destroyWindow(winname)
    
    image_path = os.path.join(config.path,'cat.bmp')
    image = read_image(image_path, config)
    # image
    # uint8, uint8, uint8 
    # value range: 0 ~255
    # if float :
    # multiplies every value by 255
    
    if image is None:
        print('\'cat.bmp\' Image does not exists in the path...')
        sys.exit()
    cv2.namedWindow('Image')
    cv2.imshow('image', image)
    # search 
    # cv2 waitkey escape
    # for the detailed explanation of the control below.
    while True:
        if cv2.waitKey() == ord('q'):
            break
    # cv2.waitKey(delay= 0)
    # delay < 0: wait forever
    # delay = 3000: wait for 3000 ms

    cv2.destroyAllWindows()

    # print('Saving the image...')
    # cv2.imwrite(os.path.join(config.path,'cat2.bmp'), image)

def new_func(image_path):
    print(image_path)

if __name__ == '__main__':
    config = define_argparser()
    main(config)
