#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import matplotlib.pyplot as plt
import cv2
import math
import numpy as np


# # Loading Image

# In[2]:


img_path = './Bird2.jpg'
img_path


# In[3]:


rgb_img = cv2.imread(img_path)
rgb_img.shape


# # 2d grayscale conversion

# In[4]:


grayscale = cv2.cvtColor(rgb_img,cv2.COLOR_RGB2GRAY)
grayscale.shape


# # Padding the Image

# In[5]:


def padding(img):
    width, height = img.shape
    new_img = np.zeros(shape=(width+2,height+2))
    width, height = new_img.shape
    new_img[1:width-1,1:height-1] = img
    return new_img


# In[6]:


pad_img = padding(grayscale)
pad_img.shape


# # Convulotion Function

# In[7]:


def convolution(img, kernel):
    
    img = padding(np.array(img))
    
    _, k = kernel.shape
    width, height = img.shape
    
    new_width, new_height = width-k+1, height-k+1
    conv_image = np.zeros(shape=(new_width,new_height))
    
    for i in range(new_width):
        for j in range(new_height):
            mat = img[i:i+k, j:j+k]
            conv_image[i, j] = np.sum(np.multiply(kernel, mat))
            
            if conv_image[i,j] < 0:
                conv_image[i,j] = 0
            elif(conv_image[i,j]  > 255):
                conv_image[i,j] = 255
    
    return conv_image
    


# # Plot Function
# 

# In[8]:


def plot_img(img_set, title_set):
    n = len(img_set)
    plt.figure(figsize = (30, 20))
    for i in range(n):
        img = img_set[i]
        ch = len(img.shape)

        plt.subplot(2, 2, i + 1)
        if (ch == 3):
            plt.imshow(img_set[i])
        else:
            plt.imshow(img_set[i],cmap='gray')
        plt.title(title_set[i])
    plt.savefig('./Kernel_Output.png')


# # Declaring Kernel
# 

# In[9]:


kernel = np.array([[2, 0, -2], [1, 0, -1], [2, 0, -2]])
kernel


# # Calling Kernel Functions

# In[10]:


custom_kernel = convolution(grayscale, kernel)
built_in_kernel = cv2.filter2D(grayscale,-1, kernel)


# # Show Image

# In[11]:


img_set = [rgb_img, grayscale,custom_kernel, built_in_kernel]
title_set = ['RGB', 'GRAYSCALE', 'CUSTOM', 'BUILT-IN']
plot_img(img_set, title_set)


# In[ ]:




