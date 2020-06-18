#!/usr/bin/env python
# coding: utf-8

# ## Blending and Pasting Images

# In[1]:


import numpy as np
import cv2
import os

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


path = os.getcwd()


# In[3]:


img1 = cv2.imread(path +  '\Images\dog_backpack.jpg')
img2 = cv2.imread(path +  '\Images\watermark_no_copy.png')

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)


# In[4]:


img2 = cv2.resize(img2,dsize = (600,600))
rows,cols,channels = img1.shape

x_offset = cols - 600
y_offset = rows - 600

roi = img1[y_offset:rows,x_offset:cols]


# In[5]:


img2gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
mask_inv = cv2.bitwise_not(img2gray)
fg = cv2.bitwise_or(img2, img2, mask=mask_inv)


# In[6]:


final_roi = cv2.bitwise_or(roi,fg)


# In[9]:


large_img = img1
small_img = final_roi

large_img[y_offset:y_offset+small_img.shape[0], x_offset:x_offset+small_img.shape[1]] = small_img

large_img = cv2.cvtColor(large_img,cv2.COLOR_RGB2BGR)
cv2.imwrite("Blended image.jpg",large_img)


# In[ ]:




