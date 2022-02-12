import cv2

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