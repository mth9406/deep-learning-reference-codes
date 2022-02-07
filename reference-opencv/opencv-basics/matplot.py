import matplotlib.pyplot as plt
import cv2

imgBGR = cv2.imread('cat.bmp')
imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
imgGray = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)

# plt.axis('off')
# plt.imshow(imgRGB)
# plt.show()

fig, axes = plt.subplots(1,3)
axes[0].axis('off')
axes[0].imshow(imgRGB)
axes[1].axis('off')
axes[1].imshow(imgGray)
axes[2].axis('off')
axes[2].imshow(imgBGR)
plt.show()