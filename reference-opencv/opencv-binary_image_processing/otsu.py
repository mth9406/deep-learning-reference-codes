'''How to use Otsu to binarize an image

th, dst = threshold(src, thres, maxval, 
                    type = cv2.THRESH_* | cv2.THRESH_OTSU)

notice that thres will be ignored
if you only ues cv2.ThRESH_OTSU, then cv2.THRESH_BINARY will be
used as a 'backbone' binarization. 

'''


import cv2
import sys

from cv2 import threshold

src = cv2.imread('./data/rice.png', cv2.IMREAD_GRAYSCALE)

if src is None:
    print('Image load failed!')
    sys.exit()

##############################################################################
th, dst = threshold(src, 0, 255, type = cv2.THRESH_BINARY | cv2.THRESH_OTSU)##
print("Otsu's threshold", th)

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()