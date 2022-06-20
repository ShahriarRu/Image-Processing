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


# In[4]:


plt.figure(figsize = (50, 15))
plt.subplot(1, 2, 1)
plt.hist(grayscale.ravel(),256,[0,256]);


values, count = np.unique(grayscale, return_counts=True)
plt.subplot(1, 2, 2)
plt.bar([x for x in range(0,256)], count, width=0.9)

plt.savefig('./Histogram_Output.png')


# In[ ]:




