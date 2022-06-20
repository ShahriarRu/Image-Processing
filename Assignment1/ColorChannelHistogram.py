#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import cv2 as cv
import matplotlib.pyplot as plt


# # Load Image

# In[2]:


img_path = './Bird.jpg'
#print(img_path)

img = plt.imread(img_path)

print(img.shape)
print(img)


# In[3]:


plt.imshow(img) 


# # Binary and Grascale setup

# In[4]:


grayscale_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
print(grayscale_img.shape)

_,binary_img = cv.threshold(grayscale_img, 127, 255, cv.THRESH_BINARY)
print(binary_img.shape)


# # Image Channel Division

# In[5]:


fig, ax = plt.subplots(figsize=(30,20))



plt.subplot(2,3,1)
plt.title("RGB")
plt.imshow(img, cmap = 'gray')

plt.subplot(2,3,2)
plt.title("RED")
plt.imshow(img[:,:,0])

plt.subplot(2,3,3)
plt.title("GREEN")
plt.imshow(img[:,:,1])

plt.subplot(2,3,4)
plt.title("BLUE")
plt.imshow(img[:,:,2])

plt.subplot(2, 3, 5)
plt.title('GRAYSCALE')
plt.imshow(grayscale_img, cmap = 'gray')

plt.subplot(2, 3, 6)
plt.title('Binary')
plt.imshow(binary_img, cmap = 'gray')



#ax.tick_params(axis='x', color='red')  
#ax.tick_params(axis='y', color='white')

#plt.show()
plt.savefig('./rgb_output.png')


# # Histogram

# In[6]:


fig, ax = plt.subplots(figsize=(20,10))

plt.subplot(1,3,1)
plt.title('GRAYSCALE')
plt.imshow(img[:,:,1], cmap='gray')

# usning matplotlib
plt.subplot(1,3,2)
plt.title('Histogram in Matplotlib')
plt.hist(img.ravel(),256,[0,256],color='blue')
plt.ylim(ymin=0,ymax=80000)



# usning opencv
plt.subplot(1,3,3)
plt.title('Histogram in OpenCv')
histr = cv.calcHist([img],[0],None,[256],[0,256])
plt.ylim(ymin=0,ymax=80000) 
plt.plot(histr)

#plt.show()
plt.savefig('./histogram_output.png')


# In[ ]:





# In[ ]:





# In[ ]:




