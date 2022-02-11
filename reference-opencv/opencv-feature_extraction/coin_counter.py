import cv2
import argparse
from cv2 import cvtColor
import numpy as np
import sys

# Coin counter
# to implement:
# (1) detect coins (Kor)
# (2) distinguish coins 
#     assumption: we only have 100, 10 won.

def argparser():
    p = argparse.ArgumentParser()
    
    # source image path
    p.add_argument('--src', type= str, required= True)
    
    # HoughCircles' params
    p.add_argument('--minDist', type= int, default= 20)
    p.add_argument('--param1', type= int, default = 50)
    p.add_argument('--param2', type= int, default = 25)
    p.add_argument('--maxRadius', type= int, default= 150)
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
    src_hsv = cvtColor(src, cv2.COLOR_BGR2HSV)

    # do HoughCircles to find circles
    circles = cv2.HoughCircles(cpy, cv2.HOUGH_GRADIENT, 1, minDist, 
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
        
        if np.linalg.norm(src_hsv[cx, cy, :] - hundCoinColor, ord = 1) <= 1e-2:
            numHundreds += 1
        elif np.linalg.norm(src_hsv[cx, cy, :] - tenCoinColor, ord = 1) <= 1e-2:
            numTens += 1

    return numCoins, numHundreds, numTens


def main(config):

    src = imread(config.src, cv2.IMREAD_COLOR) # BGR

    # obtain the color of coins 
    hundCoinColor = None # todo
    tenCoinColor = None # todo

    numCoins, numHundreds, numTens = detectCoins(src, 
                                                hundCoinColor, tenCoinColor,
                                                config.minDist, config.param1, config.param2,
                                                config.minRadius, config.maxRadius
                                                )
    
    print(f'The number of coins: {numCoins}')
    print(f'The number of hundred won coins: {numHundreds}')
    print(f'The number of ten won coins: {numTens}')

if __name__ == '__main__':
    config = argparser()
    main(config)