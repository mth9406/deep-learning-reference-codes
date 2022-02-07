import cv2
import sys
import numpy as np

# cv2.TickMeter() -> tm
# tm.start(): start ticking 
# tm.stop(): end ticking
# tm.reset(): initiate the time  
# tm.start() func1 tm.stop() *tm.reset() tm.start() func2 tm.stop()

# tm.getTimeSec(): returns the time in sec
# tm.getTimeMilli(): returns the time in ms
# tm.getTimeMicro(): return the time in micro sec


img = cv2.imread('./data/hongkong.jpg')

if img is None:
    print('image open failed')
    sys.exit()

tm = cv2.TickMeter()

tm.start()
edge = cv2.Canny(img, 50, 150)
tm.stop()

print(f'Elapsed time: {tm.getTimeMilli()} ms')
