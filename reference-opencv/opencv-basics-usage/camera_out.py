import sys
import cv2

import argparse

# Notice that
# VideoCapture does not write 'sound'

def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--filename', type= str, default = r'C:\Users\User\OneDrive\바탕 화면\getDrill\reference-opencv\opencv-basics-usage\data\output.mp4')
    config = p.parse_args()
    return config


def main(config):
    # take inputs from webcam.
    cap = cv2.VideoCapture(0)

    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'DIVX') # *'DIVX' = 'D', 'I', 'V', 'X'
    out = cv2.VideoWriter(config.filename, fourcc, 30, (w,h)) # declare an writer instance
    fps = cv2.CAP_PROP_FPS
    delay = round(1000/fps) # in ms.

    while True:
        ret, frame = cap.read()
        
        if not ret:
            print('failed to read camera...')
            sys.exit()

        inversed = ~frame # reflect the frame
        out.write(inversed) # write the frame   

        cv2.imshow('frame', frame)
        cv2.imshow('inversed', inversed)

        if cv2.waitKey(delay) == 27:
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    config = define_argparser()
    main(config)