import cv2
import numpy as np
import argparse
import sys

from torch import BenchmarkConfig

def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--video1_path', type= str, required= True)
    p.add_argument('--video2_path', type= str, required= True)
    p.add_argument('--save_path', type= str, required= True)
    # p.add_argument('--video1_prop', type= float, default= 0.7)
    # p.add_argument('--lastN', type= float, default = 0.9)
    # p.add_argument()
    config = p.parse_args()
    return config

def videoOpenedCheck(cap, i):
    if not cap.isOpened():
        print(f'{i}th video open failed!')
        sys.exit()

def getWidthAndHeight(cap):
    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return w, h

def resizeVideo(cap, w, h):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)


# video clip effect
# (1) open the two video clip at the same time
# (2) synthesize the last N frame of the first video and the following one.
# (3) save the new video.
def main(config):
    # (1) open the two video clip at the same time
    cap1 = cv2.VideoCapture(config.video1_path)
    cap2 = cv2.VideoCapture(config.video2_path)

    # to handle exceptions...
    videoOpenedCheck(cap1, 1)
    videoOpenedCheck(cap2, 2)

    # resize the frames so that
    # we can synthesize the two video clips.
    w1, h1 = getWidthAndHeight(cap1)
    w2, h2 = getWidthAndHeight(cap2)
    min_w, min_h = np.min([w1, w2]), np.min([h1, h2])
    resizeVideo(cap1, min_w, min_h)
    resizeVideo(cap2, min_w, min_h)

    # assume that fps's of each video are the same.
    # set the fps's the same if required...
    fps = cap1.get(cv2.CAP_PROP_FPS)
    delay = int(1000/ fps) # in milli second
    effect_frames = int(fps*2)
    
    # total number of frames of each video
    frame_cnt1 = round(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_cnt2 = round(cap2.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(config.save_path, fourcc, 30, (min_w, min_h)) # fps = 30

    for i in range(frame_cnt1 - effect_frames):
        ret1, frame1 = cap1.read()
        if not ret1:
            break
        
        out.write(frame1)

        cv2.imshow('frame', frame1)
        cv2.waitKey(delay)

    # synthesizing the frame
    frame = np.zeros((min_h, min_w, 3), dtype= np.uint8)

    for i in range(effect_frames):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        # # synthesize
        # dx = int(min_w * i / effect_frames)    
        # frame[:, 0:dx] = frame2[:, 0:dx]
        # frame[:, dx:] = frame1[:, dx:]

        # Another effect
        # dissolve
        alpha = 1.0 - (i/effect_frames)
        frame = cv2.addWeighted(frame1, alpha, frame2, 1-alpha, 0)

        out.write(frame)
        cv2.imshow('frame', frame)
        cv2.waitKey(delay)

    for i in range(frame_cnt2-effect_frames):
        ret2, frame2 = cap2.read()

        if not ret2:
            break
        
        cv2.imshow('frame', frame2)
        cv2.waitKey(delay)
        out.write(frame2)

    cap1.release()
    cap2.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    config = define_argparser()
    main(config)