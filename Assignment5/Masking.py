#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import cv2
import numpy as np


# In[2]:


img_path = "C:/Users/Asus/OneDrive/Documents/MyWebProjects/Image Processing/Practice/images/bird2.jpg"
img_path


# In[3]:


img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
grayscale = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)


# In[4]:


width,height = grayscale.shape
print(width,height)


# In[5]:


black = np.zeros((width,height), dtype = 'uint8')


# In[6]:


cv2.circle(black, (black.shape[1]//2 + 80, black.shape[0]//2 - 130), 280, (1,1,1), thickness=-1)


# In[7]:


def bitMasking(img, mask_img):
    w,h = img.shape
    sliced_img = img.copy()
    for x in range(w):
        for y in range(h):
            sliced_img[x,y] =  mask_img[x,y] and img[x,y]
            
    
    return sliced_img


# In[8]:


masked_image = bitMasking(grayscale, black)


# In[9]:


plt.figure(figsize=(25,15))
plt.subplot(2,2,1)
plt.title('RGB')
plt.imshow(img_rgb)

plt.subplot(2,2,2)
plt.title('GRAYSCALE')
plt.imshow(grayscale, cmap='gray')

plt.subplot(2,2,3)
plt.title('Black')
plt.imshow(black, cmap='gray')

plt.subplot(2,2,4)
plt.title('Masked Image')
plt.imshow(masked_image, cmap='gray')

plt.savefig('mask.png')


# In[ ]:





# In[ ]:




