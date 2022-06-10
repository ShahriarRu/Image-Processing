#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import matplotlib.pyplot as plt
import cv2
import numpy as np
import math


# # Loading Image

# In[2]:


img_path = './minions.jpg'
rgb = plt.imread(img_path)
print(rgb.shape)


# In[3]:


grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
print(grayscale.shape)


# In[4]:


plt.imshow(grayscale,cmap = 'gray')


# # Navigation

# ### Transformation 1

# In[5]:


def transformation1(T1, T2):

    m, n = grayscale.shape
    processed_img = grayscale.copy()
    for x in range(m):
        for y in range(n):
            if(processed_img[x, y] >= T1 and processed_img[x, y] <= T2):
                processed_img[x, y] = 100
            else: 
                processed_img[x, y] = 10

    return processed_img


# ### Transformation 2

# In[6]:


def transformation2(T1, T2):

    m, n = grayscale.shape
    processed_img = grayscale.copy()
    for x in range(m):
        for y in range(n):
            if(processed_img[x, y] >= T1 and processed_img[x, y] <= T2):
                processed_img[x, y] = 100

    return processed_img


# ### Transformation 3
# 

# In[7]:


def transformation3(T1, T2, c):

    m, n = grayscale.shape
    processed_img = grayscale.copy()
    for x in range(m):
        for y in range(n):
            processed_img[x,y] = c * math.log(1+processed_img[x,y])

    return processed_img


# ### Transformation 4

# In[8]:


def transformation4(T1, T2, c, p, epsilon):

    m, n = grayscale.shape
    processed_img = grayscale.copy()
    for x in range(m):
        for y in range(n):
            processed_img[x,y] = c * (processed_img[x,y] + epsilon) ** p

    return processed_img


# # Plotting Image

# In[9]:


def plot_img(img_set, title_set):
	n = len(img_set)
	plt.figure(figsize = (20, 20))
	for i in range(n):
		img = img_set[i]
		ch = len(img.shape)
	
		plt.subplot( 3, 2, i + 1)
		if (ch == 3):
			plt.imshow(img_set[i])
		else:
			plt.imshow(img_set[i], cmap = 'gray')
		plt.title(title_set[i])
	plt.show()		
	


# # Calling the Conditions

# In[10]:


s1 = transformation1(100,200)


# In[11]:


s2 = transformation2(100,200)


# In[12]:


s3 = transformation3(100,200, 50)


# In[13]:


s4 = transformation4(100,200, 50, 4, 0.0000000001)


# # showing the image

# In[14]:


img_set = [rgb, grayscale,s1, s2, s3, s4]
title_set = ['RGB', 'GRAYSCALE', '100 or 10', '100 or r', 's = c log(1 + r)', 's = c ( r + epsilon ) ^ p']
plot_img(img_set, title_set)


# In[ ]:





# In[ ]:




