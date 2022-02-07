import sys
import cv2

import argparse

def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--srcpath', type= str, default = r'C:\Users\User\OneDrive\바탕 화면\getDrill\reference-opencv\opencv-basics-usage\data\video1.mp4')
    p.add_argument('--filename', type= str, default = r'C:\Users\User\OneDrive\바탕 화면\getDrill\reference-opencv\opencv-basics-usage\data\modified_video.mp4')
    config = p.parse_args()
    return config


def main(config):
    # take inputs from webcam.
    cap = cv2.VideoCapture(config.srcpath)

    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'DIVX') # *'DIVX' = 'D', 'I', 'V', 'X'
    out = cv2.VideoWriter(config.filename, fourcc, 30, (w,h)) # declare an writer instance

    while True:
        ret, frame = cap.read()
        
        if not ret:
            print('failed to read camera...')
            sys.exit()

        # edge = cv2.Canny(frame) 
        # cv2.Canny - gray scale 
        # to write the video 
        # we need to change the color
        # edge_color = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
        inversed = ~frame # reflect the frame
        out.write(inversed) # write the frame   
        # out.write(edge_color)

        # cv2.imshow('edge', edge_color)
        cv2.imshow('frame', frame)
        cv2.imshow('inversed', inversed)

        if cv2.waitKey(20) == 27:
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    config = define_argparser()
    main(config)