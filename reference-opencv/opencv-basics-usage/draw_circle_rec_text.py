import cv2
import numpy as np
# import os
import argparse

# get data path 
# and draw line, circle, text on the data.
def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--path', type= str,
                    default = r'C:\Users\User\OneDrive\바탕 화면\getDrill\reference-opencv\opencv-basics-usage\data\field.bmp'
    )
    config = p.parse_args()
    return config

def readImage(path):
    image = np.fromfile(path, dtype= np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def main(config):
    # cv2.imread()
    img = readImage(config.path)
    rows, cols = img.shape[:2]
    thrs = rows//2, cols//2
    crops = img[:thrs[0], :thrs[1]], img[:thrs[0], thrs[1]:],\
            img[thrs[0]:, :thrs[1]],img[thrs[0]:, thrs[1]:]
    
    # draw a rectangle
    # draw a circle
    # draw a line
    # write some texts
    for i, crop in enumerate(crops):
        if i == 0:
            cv2.rectangle(crop, (0,0), (thrs[0],thrs[1]),(0,0,0), 2)
        elif i == 1:
            cv2.line(crop, (0,0), (thrs[0],thrs[1]),(0,0,0), 2)
        elif i == 2:
            center = (thrs[1]//2, thrs[0]//2) # col, height
            radius = np.min((thrs[0]//2, thrs[1]//2))
            cv2.circle(crop, center, radius, (0,0,0), 2, cv2.LINE_AA)
            # fill the circle if -1 is given to the param: thickness.
            # cv2.Line_AA: To smooth the boundary of the circle.
        else:
            center = (thrs[1]//2, thrs[0]//2)
            cv2.putText(crop, 'Apple', center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, cv2.LINE_AA)

    # draw a polyline
    pts = np.array([
        [0,0],
        [thrs[0]//2, thrs[1]//2],
        [thrs[0]//4, thrs[1]//2],
        [thrs[1]//8, thrs[0]//4]
    ])
    cv2.polylines(crops[-1], [pts], isClosed= True, color = (0,0,0), thickness= 1)

    for i, crop in enumerate(crops):
        cv2.imshow(f'{i}\'th image', crop)
    cv2.imshow('original image', img)

    while True:
        if cv2.waitKey() == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    config = define_argparser()
    main(config)
