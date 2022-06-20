#!/usr/bin/env python
# coding: utf-8

# # importing libraries

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import math


# # loading image

# In[2]:


img_path = './Bird2.jpg'
rgb_img = plt.imread(img_path)
rgb_img.shape


# # Convert to GrayScale

# In[3]:


grayscale = cv.cvtColor(rgb_img,cv.COLOR_RGB2GRAY)
grayscale.shape


# # Preparing Kernals

# In[4]:


e = 2.7182 #Eular's_number
pi = 3.1416
kernel1 = np.ones((3, 3), dtype = np.float32) * math.log(e)/(2*pi)
kernel1


# In[5]:


kernel2 = np.array([[2, 0, -2], [1, 0, -1], [2, 0, -2]])
kernel2


# In[6]:


kernel3 = np.array([[0, 0, 1.618], [0, -1, 0], [1.618, 0, 0]])
kernel3


# In[7]:


kernel4 = np.array([[2, 0, -1], [0, 0.7, 0], [1, 0, -2]])
kernel4


# In[8]:


kernel5 = np.array([[-1.5, -1, 0], [-1, 0.3, 1], [0, 1, 1.5]])
kernel5


# In[9]:


kernel6 = np.array([[1, -1, -2], [0, 0.5, 0], [2, 1, -1]])
kernel6


# In[10]:


processed_img1 = cv.filter2D(grayscale, -1, kernel1)
processed_img2 = cv.filter2D(grayscale, -1, kernel2)
processed_img3 = cv.filter2D(grayscale, -1, kernel3)
processed_img4 = cv.filter2D(grayscale, -1, kernel4)
processed_img5 = cv.filter2D(grayscale, -1, kernel5)
processed_img6 = cv.filter2D(grayscale, -1, kernel6)


# # Plotting Function

# In[11]:


def plot_img(img_set, title_set):
    n = len(img_set)
    plt.figure(figsize = (20, 30))
    for i in range(n):
        img = img_set[i]
        ch = len(img.shape)

        plt.subplot(4, 2, i + 1)
        if (ch == 3):
            plt.imshow(img_set[i])
        else:
            plt.imshow(img_set[i],cmap='gray')
        plt.title(title_set[i])
    plt.savefig('./Output.png')


# # Plotting Kernels

# In[12]:


img_set = [rgb_img, grayscale, processed_img1, processed_img2, processed_img3,processed_img4,processed_img5, processed_img6]
title_set = ['RGB', 'Grayscale', 'Kernel1', 'Kernel2', 'Kernel3','Kernel4','Kernel5','Kernel6']
plot_img(img_set, title_set)

