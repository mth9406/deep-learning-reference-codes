'''labeling
(1) cv2.connectedComponents(image, 
                        labels= None, connectivity= None, ltype= None)
-> retval, labels

image: 8bit 1channel image
labels: label map 2d array (the same shape as a source image)
connectivity: 4 or 8, default= 8 (recommended: None)
ltype: labels type 
     : cv2.CV_32S(default) or cv2.CV_16S
     : (recommended: None)

retval: the number of objects

(2) cv2.connectedComponentsWithStats(image, ...)
-> retval, labels, stats, centroids

image: source image
ltypes: labels matrix type, cv2.CV_32S (default)

retvals: the number of objects (including a background)
labels: label map
stats: bounding box, the number of pixels of each object; 
     : shape= (N, 5) = (x, y, w, h, numPixels)
     : np.int32
     : (including a background)
centroids: centroids of each object; shape= (N, 2), np.float64
         : (including a background)
'''

import cv2
import sys
import numpy as np

def localThreshold(src, num_pathes):

    h, w = src.shape[:2]
    dh, dw = h//num_pathes, w//num_pathes
    dst = np.zeros(src.shape[:2], dtype=np.uint8)
    
    for x in range(num_pathes+1):
        for y in range(num_pathes+1):
            startX, endX, startY, endY \
                = dw*x, min(dw*(x+1), w), dh*y, min(dh*(y+1), h)
            cv2.threshold(src[startY:endY, startX:endX], 0, 255, 
                        type= cv2.THRESH_BINARY|cv2.THRESH_OTSU,
                        dst= dst[startY:endY, startX:endX])
    
    return dst

src = cv2.imread('./data/rice.png', cv2.IMREAD_GRAYSCALE)
if src is None:
    print('Image load failed')
    sys.exit()

dst = localThreshold(src, 7) # binarize.
dst2 = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

retval, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

# notice that 
# stats[0] is a background.
for x, y, w, h, area in stats[1:]:
    if area >= 20:
        # or denoise 'dst' before using connetedComponentsWithStats()
        cv2.rectangle(dst2, (x,y,w,h), (0,255,0), 2)

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.imshow('dst2', dst2)
cv2.waitKey()
cv2.destroyAllWindows()
