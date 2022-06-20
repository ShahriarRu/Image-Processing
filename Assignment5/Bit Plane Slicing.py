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


# In[4]:


grayscale = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)


# In[5]:


cv2.imshow("Bird", img)
cv2.waitKey(0)


# In[6]:


plt.figure(figsize=(25,15))
plt.subplot(1,2,1)
plt.title('RGB')
plt.imshow(img_rgb)

plt.subplot(1,2,2)
plt.title('GRAYSCALE')
plt.imshow(grayscale, cmap='gray')

plt.show()


# In[7]:


def bitSlicing(img, slice_bit):
    w,h = img.shape
    sliced_img = img.copy()
    for x in range(w):
        for y in range(h):
            sliced_img[x,y] = img[x,y] & slice_bit
            
    
    return sliced_img


# In[8]:


eighth = bitSlicing(grayscale,128)


# In[9]:


seventh = bitSlicing(grayscale,64)


# In[10]:


sixth =  bitSlicing(grayscale,32)


# In[11]:


fifth = bitSlicing(grayscale,16)


# In[12]:


fourth = bitSlicing(grayscale,8)


# In[13]:


third = bitSlicing(grayscale,4)


# In[14]:


second = bitSlicing(grayscale,2)


# In[15]:


first = bitSlicing(grayscale,1)


# In[16]:


Final = first + second + third + fourth + fifth + sixth + seventh + eighth
Final.shape


# In[ ]:





# In[17]:


plt.figure(figsize=(30,50))

plt.subplot(6,2,1)
plt.title('RGB')
plt.imshow(img_rgb)

plt.subplot(6,2,2)
plt.title('GRAYSCALE')
plt.imshow(grayscale, cmap='gray')

plt.subplot(6,2,3)
plt.title('Eighth')
plt.imshow(eighth, cmap='gray')

plt.subplot(6,2,4)
plt.title('Seventh')
plt.imshow(seventh, cmap='gray')

plt.subplot(6,2,5)
plt.title('Sixth')
plt.imshow(sixth, cmap='gray')

plt.subplot(6,2,6)
plt.title('Fifth')
plt.imshow(fifth, cmap='gray')

plt.subplot(6,2,7)
plt.title('Fourth')
plt.imshow(fourth, cmap='gray')

plt.subplot(6,2,8)
plt.title('Third')
plt.imshow(third, cmap='gray')

plt.subplot(6,2,9)
plt.title('Second')
plt.imshow(second, cmap='gray')

plt.subplot(6,2,10)
plt.title('First')
plt.imshow(first, cmap='gray')

plt.subplot(6,2,11)
plt.title('Final')
plt.imshow(Final, cmap='gray')

plt.subplot(6,2,12)
plt.title('Original')
plt.imshow(grayscale, cmap='gray')

plt.savefig('plane_slicing.png')


# In[ ]:




