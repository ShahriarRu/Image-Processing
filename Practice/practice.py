import matplotlib.pyplot as plt
import cv2

rgb = plt.imread('images/bird.jpg')
# plt.imshow(rgb)



binary = cv2.threshold(rgb, 127, 255, cv2.THRESH_BINARY)
plt.imshow(binary)