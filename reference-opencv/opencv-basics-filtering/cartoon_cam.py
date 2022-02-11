# (1) sketch filter
# (2) cartoon filter
# press space bar to adapt different kind of
# filter.

import cv2
import argparse
import sys

def argparser():
    p = argparse.ArgumentParser()
    # p.add_argument('--cam_mode', type= bool, default = 0)
    p.add_argument('--save_path', type= str, default = r'./data/output.mp4')
    p.add_argument('--sketch_simgaX', type= int, default= 10)
    p.add_argument('--cartoon_alpha', type= float, default= 2.0)
    
    config = p.parse_args()
    return config

def applySketchFilter(frame, sigmaX = 10):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # img-blurred in bright dimension
    blurred = cv2.GaussianBlur(frame_gray, (0,0), sigmaX)
    edge = ~cv2.addWeighted(frame_gray, 1, blurred, -1, 0)
    cv2.putText(edge, 'Sketch filter', (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, cv2.LINE_AA)
    edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
    return edge

def applyCartoonFilter(frame, alpha=2):
    # increase contrast
    frame_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(frame_ycrcb)
    
    # img-blurred in bright dimension
    y_blurred = cv2.GaussianBlur(y, (0,0), 10)
    y_contrast = cv2.addWeighted(y, (1+alpha), y_blurred, -alpha, 0)
    modified = cv2.merge([y_contrast, cr, cb])
    
    # YCrCb --> BGR
    modified = cv2.cvtColor(modified, cv2.COLOR_YCrCb2BGR)
    cv2.putText(modified, 'Cartoon filter', (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, cv2.LINE_AA)
    return modified


def main(config):  

    cap = cv2.VideoCapture(0) # connect a local camera
    cap.open(0)

    if not cap.isOpened():
        print('--(!)camera open failed')
        sys.exit()

    # initiate a writer
    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000/fps) # milli second
    fourcc= cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(config.save_path, fourcc, fps, (w,h))
    
    status = 0

    while True:
        # read the frame from the cap
        ret, frame = cap.read()

        if not ret:
            break
        
        query = cv2.waitKey(delay)
        if status == 0:
            # original frame
            cv2.imshow('out', frame)  
        elif status == 1:
            frame = applySketchFilter(frame, config.sketch_simgaX)
            cv2.imshow('out', frame)
        else: 
            frame = applyCartoonFilter(frame, config.cartoon_alpha)
            cv2.imshow('out', frame)
        
        out.write(frame)

        if query == 32:
            # update status when you press spacebar.
            status = (status + 1)%3

        elif query == 27:
            # end while loop when you press esc.
            break
        
    cv2.destroyAllWindows()

    cap.release()
    out.release()


if __name__ == '__main__':
    config = argparser()
    main(config)

    
    
    
