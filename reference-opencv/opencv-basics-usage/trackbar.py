import cv2
import numpy as np

# Trackbar
# cv2.createTrackbar(trackbarName, windowName, value, count, onChange)-> None
# trackbarName: the name of the track bar
# windowName: window name
# value: initial postion
# count: maximal value of the trackbar
# onChange: call-back function when the position of a track-bar changes
#         : onChange(pos) -> None

def on_level_changed(pos):
    # in this case
    # pos = 0 ~ 15
    # createTrackbar: count = 16, minimum = 0 (by default.)
    global img
    # print(pos)
    level = pos*16
    img[:,:] = np.clip(level, 0, 255)
    # equivalent to...
    # img[:,:] = level if level  < 256 else 255
    # 0, 16, ..., 255
    # if 256 is assigned, force the number 0. 
    # lightens the image
    cv2.imshow('image', img)
    return None

img = np.zeros((480, 640), np.uint8) # grayscale-image

cv2.namedWindow('image')

cv2.createTrackbar('level', 'image', 
                0, 16, on_level_changed
                )

cv2.imshow('image', img)
cv2.waitKey()

cv2.destroyAllWindows()