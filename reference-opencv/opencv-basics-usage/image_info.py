import numpy as np
import cv2
import os

def read_image(full_path):
    img_array = np.fromfile(full_path, np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

img_path = os.path.join(os.pardir, 'opencv-basics/cat.bmp')
img = read_image(img_path)

# print(img.shape)
# print(img)
row, col = img.shape[:2]
imgs = img[:row//2,:col//2,:], img[:row//2,col//2:,:]\
    ,img[row//2:,:col//2,:], img[row//2:,col//2:,:]
imgs[0][:,:,:] = np.full(imgs[0].shape, fill_value = (0,255,255), dtype= np.uint8) # Yellow 
cv2.rectangle(imgs[1], (0,0), (100,100), (0,0,0), 2) # inplace = True 

for im in imgs:
    cv2.namedWindow('Cat image')
    cv2.imshow('Cat image',im)
    while True: 
        if cv2.waitKey() == ord('q'):
            break

cv2.destroyAllWindows()