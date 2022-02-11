import cv2
import sys

import matplotlib.pyplot as plt

def myImread(path, code):
    src = cv2.imread(path, code)
    if src is None:
        print('--(!)Image load failed')
        sys.exit()
    return src

src = myImread('./data/lenna.bmp', cv2.IMREAD_COLOR)
# src_color = myImread('./data/lenna.bmp', cv2.IMREAD_COLOR)

colors = ['b','g','r']
bgr_planes = cv2.split(src)

for p, c in zip(bgr_planes, colors):
    hist = cv2.calcHist([p], [0], None, [256], [0,256])
    plt.plot(hist, color = c)

# one dimension matrix

cv2.imshow('src', src)
plt.show()




