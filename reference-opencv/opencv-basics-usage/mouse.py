# from functools import partial
import cv2
# import argparse
import numpy as np


# To process a mouse event
# setMouseCallback(windowName, onMouse, param=None)
# -> None
# onMouse: my mouse call back function.
# param: data (video clip, image ...) to convey to the call back function.

# signiture of onMouse
# onMouse(event, x, y, flags, param) -> None
# event: cv2.EVENT_* 
# x, y : coordinate
# flags: status when the mouse event happens.
# param: data to convey to the call back function

# constant to the mouse event
# cv2.EVENT_MOUSEMOVE
# EVENT_LBUTTONDOWN
# EVENT_RBUTTONDOWN
# ...
# https://wiserloner.tistory.com/833
# https://docs.opencv.org/4.x/d0/d90/group__highgui__window__flags.html#ga927593befdddc7e7013602bca9b079b0

# def myImread(path):
#     img_array = np.fromfile(path, dtype= np.uint8)
#     img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
#     return img

oldx = oldy = -1

def onMouse(event:int, x:int, y:int, flags:int, userdata=None)->None:
    global img, oldx, oldy

    if event == cv2.EVENT_LBUTTONDOWN:
        oldx, oldy = x, y
        print(f'EVENT_RBUTTONDOWN: ({x}, {y})')
    elif event == cv2.EVENT_LBUTTONUP:
        print(f'EVENT_RBUTTONDOWN: ({x}, {y})')
    elif event == cv2.EVENT_MOUSEMOVE:
        # '==' operation for the event
        # '&' operation for the flags
        if flags & cv2.EVENT_FLAG_LBUTTON:
            # print(f'EVENT_MOUSEMOVE: ({x}, {y})')
            # cv2.circle(img, (x,y), 5, (0,0,255), -1, cv2.LINE_AA) # too slow.
            cv2.line(img, (oldx, oldy), (x,y), (0,0,255), 3, cv2.LINE_AA)
            cv2.imshow('image', img)
            oldx, oldy = x, y


# img = myImread(config.path)
img = np.ones(shape= (640,640,3), dtype= np.uint8) * 255 # white background.

# cv2.namedWindow('image')
cv2.imshow('image', img)

# Mouse call back function should have
# 'windowName' opened 
# before declaring it.
cv2.setMouseCallback('image', onMouse, img)
while True:
    query = cv2.waitKey()
    if query == 27:
        break
