import cv2
import matplotlib.pyplot as plt
import sys


src1 = cv2.imread('./data/lenna256.bmp', cv2.IMREAD_GRAYSCALE)
src2 = cv2.imread('./data/square.bmp', cv2.IMREAD_GRAYSCALE)

if src1 is None or src2 is None:
    print('--(!)Image load failed')
    sys.exit()

dst1 = cv2.add(src1, src2, dtype= cv2.CV_8U)
dst2 = cv2.addWeighted(src1, 0.5, src2, 0.5, 0.0)
dst3 = cv2.subtract(src1, src2)
dst4 = cv2.absdiff(src1, src2)

quaries = ['src1', 'src2', 'add', 'addWeighted', 'subtract', 'absdiff']

dsts = [src1, src2, dst1, dst2, dst3, dst4]
dsts = list(map(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB) , dsts))

fig, axes = plt.subplots(3,2,figsize= (15,15))


for i in range(6):
    fig.axes[i].axis('off')
    fig.axes[i].set_title(f'image: {quaries[i]}')
    fig.axes[i].imshow(dsts[i])

plt.show()