import sys
import cv2

cap = cv2.VideoCapture('./data/video1.mp4') # give file path.

if not cap.isOpened():
    print('video open failed!')
    sys.exit()

## set width, height 
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 240)

while True:
    ret, frame = cap.read() # True/ False, frame

    # for a video clip
    # we have ret=False 
    # when the frame is the last one.
    if not ret:
        break

    # obtain edge from a frame
    edge = cv2.Canny(frame, 50, 150)

    cv2.imshow('frame', frame)
    cv2.imshow('edge  frame', edge)
    if cv2.waitKey(20) == 27: #ESC
        break

cap.release() 
# release the used resources.
# if you try to make another instance, say,
# cap2 = cv2.VideoCapture(0)
# before you call cap.release()
# then it raises the error such as:
# "Device or resource busy"

cv2.destroyAllWindows()
    
