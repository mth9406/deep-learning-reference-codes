import cv2
import argparse

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--comments', type = str, default= 'Hello, openCV')
    
    config = p.parse_args()
    return config



def main(config):
    comments = config.comments
    print(comments, cv2.__version__)

if __name__ == '__main__':
    config = define_argparser()
    main(config)