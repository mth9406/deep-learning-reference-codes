import cv2
import argparse
import numpy as np
import sys

def argparser():
    p = argparse.ArgumentParser()
    
    p.add_argument('--src', type = str, default= './data/namecard.jpg')
    p.add_argument('--dst', type = str, default = './data/namecard.modified.jpg')

    config = p.parse_args()
    return config

def imread(src, code):
    img = cv2.imread(src, code)
    if img is None:
        print('--(!)Image load failed')
        sys.exit()
    return img

def applyPerspectiveTransform(src, srcQuad, dstQuad, dsize= (0,0)):
    # get an affine matrix
    M = cv2.getPerspectiveTransform(srcQuad, dstQuad)
    dst = cv2.warpPerspective(src, M, dsize= dsize, flags= cv2.INTER_CUBIC)
    return dst

def drawROI(img, corners):

    cpy = img.copy()
    
    c1 = (192, 192, 255) # color of circles
    c2 = (128, 128, 255) # color of edges

    for corner in corners:
        cv2.circle(cpy, tuple(corner.astype(int)), 25, c1, -1, cv2.LINE_AA)
    
    cv2.line(cpy, tuple(corners[0].astype(int)), tuple(corners[1].astype(int)), c2, 2, cv2.LINE_AA)
    cv2.line(cpy, tuple(corners[1].astype(int)), tuple(corners[2].astype(int)), c2, 2, cv2.LINE_AA)
    cv2.line(cpy, tuple(corners[2].astype(int)), tuple(corners[3].astype(int)), c2, 2, cv2.LINE_AA)
    cv2.line(cpy, tuple(corners[3].astype(int)), tuple(corners[0].astype(int)), c2, 2, cv2.LINE_AA)

    # 미쳤다 진짜 이렇게 하면 되는 구나 와 진짜 와...
    # 그냥 cpy를 리턴해도 되지만
    # addWeighted를 사용해서 
    # 선택 영역 뒷부분도 보이게 함.
    disp = cv2.addWeighted(img, 0.3, cpy, 0.7, 0)
    
    return disp

# adjust srcQuad
def onMouse(event, x, y, flags, param):
    
    global srcQuad, dragSrc, ptOld, src
    
    if event == cv2.EVENT_LBUTTONDOWN:
        for i in range(4):
            if cv2.norm(srcQuad[i]-(x, y)) < 25:
                dragSrc[i] = True
                ptOld = (x,y)
                break
    if event == cv2.EVENT_LBUTTONUP:
        for i in range(4):
            dragSrc[i] = False
    
    if event == cv2.EVENT_MOUSEMOVE:
        for i in range(4):
            if dragSrc[i]:
                dx = x - ptOld[0]
                dy = y - ptOld[1]
                
                srcQuad[i] += (dx, dy)

                cpy = drawROI(src, srcQuad)
                cv2.imshow('Image', cpy)
                ptOld = (x, y)
                break

def main(config):
    # To implement:
    # (1) select the corners of documents by dragging 4 corners (flags)
    # (2) do perspective transformation when you press enter key.
    global srcQuad, dragSrc, ptOld, src
    src = imread(config.src, cv2.IMREAD_COLOR)
    
    # image specs 
    # 86:54
    h, w = src.shape[:2]
    dw = 500
    dh = round(dw* 54/86)

    # initiate srcQuad and dstQuad
    srcQuad = np.array([
        [30,30],
        [30, h-30],
        [w-30, h-30],
        [w-30, 30]
    ], dtype= np.float32)
    dstQuad = np.array(
        [[0,0],
        [0, dh-1],
        [dw-1, dh-1],
        [dw-1, 0]]
    , dtype= np.float32)
    dragSrc = [False, False, False, False]

    disp = drawROI(src, srcQuad)
    
    # (1) 
    cv2.imshow('Image', disp)
    cv2.setMouseCallback('Image', onMouse)

    # (2)
    # do perspective transformation when you press enter key.
    while True:
        query = cv2.waitKey()
        if query == 27:
            break
        
        if query == 13:
            dst = applyPerspectiveTransform(src, srcQuad, dstQuad, dsize= (dw, dh))
            cv2.imshow('dst', dst)

        elif query == ord('s') or query == ord('S'):
            cv2.imwrite(config.dst, dst)
            break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    config = argparser()
    main(config)