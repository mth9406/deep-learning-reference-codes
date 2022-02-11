import sys
from gevent import config
import numpy as np
import argparse
import cv2

def argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--src', type= str, default= r'./data/woman.mp4')
    p.add_argument('--background', type= str, default = r'./data/raining.mp4')
    p.add_argument('--save_path', type = str, default= r'./data/chroma_key.mp4' )
    config = p.parse_args() 
    return config

def getWidthAndHeight(cap):
    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return w, h

def resizeVideo(cap, w, h):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

def resizeVideo(cap, w, h):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

def selectGreenArea(frame, lowerb= (40,40,40), upperb=(70, 255, 255)):
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(frame_hsv, lowerb, upperb)
    return mask

def main(config):
    cap1 = cv2.VideoCapture(config.src) # green video clip
    cap2 = cv2.VideoCapture(config.background) # back ground 

    # resize each video 
    w1, h1 = getWidthAndHeight(cap1)
    w2, h2 = getWidthAndHeight(cap2)
    w, h = np.min([w1, w2]), np.min([h1, h2])
    resizeVideo(cap1, w, h)
    resizeVideo(cap2, w, h)

    # set the fps and delay
    # delay is the parameter of cv2.waitKey()
    fps = cap1.get(cv2.CAP_PROP_FPS)
    delay = int(1000/fps) # in milli sec.
    
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')

    # to write the new video.
    out = cv2.VideoWriter(config.save_path, fourcc, fps, (w, h)) 


    # the number of frames
    num_frames1 = cap1.get(cv2.CAP_PROP_FRAME_COUNT)
    num_frames2 = cap2.get(cv2.CAP_PROP_FRAME_COUNT)
    num_frames = int(np.min([num_frames1, num_frames2]))

    for i in range(num_frames):
        
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
        
        # synthesize.
        # (1) select green zone.
        mask = selectGreenArea(frame1)

        # (2) insert frame2 where frame1 is green.
        frame1[mask>0,:] = frame2[mask>0,:]

        out.write(frame1)
        cv2.imshow('frame', frame1)
        cv2.waitKey(delay)

    cap1.release()
    cap2.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    config = argparser()
    main(config)