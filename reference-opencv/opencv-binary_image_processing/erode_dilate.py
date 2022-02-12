'''erosion and dilation

Notice that, we erode or dilate 'binarized image'!!

(1) cv2.erode(src, kernel, 
              dst= None, anchor= None, iterations= None,
              borderType= None, borderValue= None)
src: source image
kernel: structural element (the shape of a kernel) 
      : 3X3 kernel (filter) if None is specified
      : you can make a specific kernel by cv2.getStructuringElement()
anchor: default = (-1,-1)
iterations: dafault = 1; the number of iterations of erosion algorithm
borderType, borderValue: 'how to pad'

(2) cv2.dilate(src, kernel, 
              dst= None, anchor= None, iterations= None,
              borderType= None, borderValue= None)
src: source image
kernel: structural element (the shape of a kernel) 
      : 3X3 kernel (filter) if None is specified
      : you can make a specific kernel by cv2.getStructuringElement()
anchor: default = (-1,-1)
iterations: dafault = 1; the number of iterations of dilation algorithm
borderType, borderValue: 'how to pad'


* cv2.getStructuringElement(shape, ksize, anchor= None) -> retval
shape: the shape of structure,
google it for a detailed description...
'''

import cv2
import sys

src = cv2.imread('./data/circuit.bmp', cv2.IMREAD_GRAYSCALE)
# notice that this image is already binarized.

if src is None:
    print('Image load failed')
    sys.exit()

# as a matter of a fact,
# we usually do not specify a specific structure 
# as below.
# we simply use 3X3 kernel instead. 
se = cv2.getStructuringElement(cv2.MORPH_RECT, (5,3))
dst1 = cv2.erode(src, se)
dst2 = cv2.erode(src, None) # kernel= None

dst3 = cv2.dilate(src, None)

cv2.imshow('src', src)
cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)
cv2.imshow('dst3', dst3)

cv2.waitKey()
cv2.destroyAllWindows()

