import matplotlib.pyplot as plt
import cv2
import numpy as np

img_path = 'images/bird.jpg'
rgb = plt.imread(img_path)
# plt.imshow(rgb)

gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

_, binary = cv2.threshold(gray, 127, 255, type = cv2.THRESH_BINARY)
plt.imshow(binary, cmap = 'gray')
print(binary.shape)

def erode(img, kernel):
    width, height = img.shape
    erode_img = np.ones((width+1, height+1), dtype = 'uint8')
    for x in range(1,width):
        for y in range(1, height):
            if(img[x,y] == 0):
                erode_img[x-1:x+2, y-1:y+2] = np.zeros(kernel.shape)
    
    return erode_img

def dialate(img, kernel):
    _,k = kernel.shape
    width, height = img.shape
    dial_img = np.zeros((width, height), dtype = 'uint8')
    for x in range(1,width-1):
        for y in range(1, height-1):
            if(img[x,y] == 255):
                dial_img[x-1:x+k-1, y-1:y+k-1] = kernel
    
    return dial_img

kernel = np.ones((3,3), dtype=np.uint8)

erosion = erode(binary, kernel)
dialation = dialate(binary, kernel)
plt.subplot(1,3,1)
plt.imshow(binary, cmap = 'gray')
plt.subplot(1,3,2)
plt.imshow(erosion, cmap = 'gray')
plt.subplot(1,3,3)
plt.imshow(dialation, cmap = 'gray')
plt.show()