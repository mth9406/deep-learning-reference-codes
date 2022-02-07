import sys
import cv2

cap = cv2.VideoCapture()
cap.open(0) # open 0'th camera 

if not cap.isOpened():
    print('camera open failed!')
    sys.exit()

# if we want camera frame
print('FPS: ',cap.get(cv2.CAP_PROP_FPS))
print('Frame width', cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print('Frame height', cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

## set width, height 
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 240)

while True:
    ret, frame = cap.read() # True/ False, frame
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
    
