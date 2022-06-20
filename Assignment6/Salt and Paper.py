#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import matplotlib.pyplot as plt
import cv2
import numpy as np
import random


# # Loading Image

# In[2]:


img_path = './images/flower.jpg'
img_path


# In[3]:


bgr_img = cv2.imread(img_path, -1)
rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
gray_img = cv2.imread(img_path, 0)
rgb_img.shape
gray_img.shape


# # Declaring Functions

# In[4]:


def salt_paper_noise(img, thres):
    
    w,h = img.shape    
    output = img.copy()
    
    for i in range(w):
        for j in range(h):
            rand = random.random()
            if rand < thres:
                output[i,j] = 0
            #elif rand > 1-thres:
                #output[i,j] = 0
    return output


# # Noisy Image

# In[5]:


noisy_img = salt_paper_noise(gray_img, 0.01)


# # gaussian blur

# In[6]:


average_blur = cv2.blur(noisy_img,(5,5))


# In[7]:


gauss_blur = cv2.GaussianBlur(noisy_img,(5,5),0)


# In[8]:


median_blur = cv2.medianBlur(noisy_img,5)


# In[9]:


plt.figure(figsize=(30,25))
plt.subplot(3,2,1)
plt.title('RGB')
plt.imshow(rgb_img, cmap='gray')

plt.subplot(3,2,2)
plt.title('GRAY')
plt.imshow(gray_img, cmap='gray')

plt.subplot(3,2,3)
plt.title('NOISY')
plt.imshow(noisy_img, cmap='gray')

plt.subplot(3,2,4)
plt.title('Average Blur')
plt.imshow(average_blur, cmap='gray')

plt.subplot(3,2,5)
plt.title('Gauss Blur')
plt.imshow(gauss_blur, cmap='gray')

plt.subplot(3,2,6)
plt.title('Madian Blurr')
plt.imshow(median_blur, cmap='gray')

plt.savefig('nise_blur.png')


# In[ ]:




