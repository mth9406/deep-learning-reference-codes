import cv2
import numpy as np

src = cv2.imread('./data/tekapo.bmp', cv2.IMREAD_COLOR)

h, w  = src.shape[:2]
map2, map1 = np.indices((h, w), dtype= np.float32) # row, col --> map2, map1
map2 = map2 + 5*np.sin(map1/32) 

# sin mapping
dst = cv2.remap(src, map1, map2, 
        interpolation= cv2.INTER_LINEAR, borderMode= cv2.BORDER_DEFAULT)

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()