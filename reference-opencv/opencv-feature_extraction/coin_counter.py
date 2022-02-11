import cv2
import argparse
import numpy as np
import sys

# Coin counter
# to implement:
# (1) detect coins (Kor)
# (2) distinguish coins 
#     assumption: we only have 100, 10 won.

# color detection failed!

def argparser():
    p = argparse.ArgumentParser()
    
    # source image path
    p.add_argument('--src', type= str, default = './data/coins1.jpg')
    
    # HoughCircles' params
    p.add_argument('--minDist', type= int, default= 20)
    p.add_argument('--param1', type= int, default = 150)
    p.add_argument('--param2', type= int, default = 40)
    p.add_argument('--maxRadius', type= int, default= 100)
    p.add_argument('--minRadius', type= int, default= 10)
      
    config = p.parse_args()
    return config

def imread(src, code):
    img = cv2.imread(src, code)
    if img is None:
        print('image load failed')
        sys.exit()
    return img

# (1) coin detection:
# we use HoughCircles to detect coins
def detectCoins(src, hundCoinColor, tenCoinColor, 
                minDist, param1, param2, minRadius, maxRadius):
    
    cpy = src.copy()
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    blr = cv2.GaussianBlur(gray, (0, 0), 1)

    # for fun
    h, w = src.shape[:2]
    mask = np.zeros((h, w), dtype= np.uint8)

    # do HoughCircles to find circles
    circles = cv2.HoughCircles(blr, cv2.HOUGH_GRADIENT, 1, minDist, 
                    param1= param1, param2= param2, 
                    minRadius= minRadius, maxRadius= maxRadius)
    # return type: np.ndarray, float32, (1, N, 3) [[(cx, cy, r)]...]

    # the number of coins
    numCoins = circles.shape[1]
    
    # distinguish the number of 100-coins and 10-coins
    # by color
    numHundreds, numTens = 0, 0

    for i in range(numCoins):
        circle_comp = circles[0][i]
        cx, cy, r = list(map(lambda x: int(x), circle_comp))
        
        if np.linalg.norm(src[cy, cx, :] - hundCoinColor, ord = 1) <= 1e-1:
            numHundreds += 1
        elif np.linalg.norm(src[cy, cx, :] - tenCoinColor, ord = 1) <= 1e-1:
            numTens += 1

        cv2.circle(cpy, (int(cx), int(cy)), int(r), (0,0,255), 2, cv2.LINE_AA)
        cv2.circle(mask, (int(cx), int(cy)), int(r), (255,255,255), -1, cv2.LINE_AA)

    return numCoins, numHundreds, numTens, cpy, mask

def onMouse(event:int, x:int, y:int, flags:int, param=None) -> None:
    global src, hundCoinColor, tenCoinColor

    if event == cv2.EVENT_LBUTTONUP:
        hundCoinColor = src[y,x,:]
        print('Color selected for 100 won')
        print(f'Color vale: {hundCoinColor}')
    elif event == cv2.EVENT_RBUTTONUP:
        tenCoinColor = src[y,x,:]
        print('Color selected for 10 won')
        print(f'Color vale: {tenCoinColor}')


def main(config):

    global src, hundCoinColor, tenCoinColor

    src = imread(config.src, cv2.IMREAD_COLOR) # BGR

    cv2.imshow('Image', src)
    cv2.setMouseCallback('Image', onMouse, src)

    # obtain the color of coins 
    while True:
        query = cv2.waitKey()
        if query == 27:
            break

    numCoins, numHundreds, numTens, cpy, mask = detectCoins(src, 
                                                hundCoinColor, tenCoinColor,
                                                config.minDist, config.param1, config.param2,
                                                config.minRadius, config.maxRadius
                                                )
    
    cv2.imshow('Coin detection', cpy)
    cv2.imshow('Mask', mask)
    print(f'The number of coins: {numCoins}')
    print(f'The number of hundred won coins: {numHundreds}')
    print(f'The number of ten won coins: {numTens}')
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    config = argparser()
    main(config)