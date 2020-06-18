#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


img = cv2.imread("data/rainbow.jpg")
img = cv2.imread("data/rainbow.jpg",0)


# ### Binary Threshold

# In[3]:


ret,bi_threshold = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
bi_threshold = cv2.cvtColor(bi_threshold, cv2.COLOR_RGB2BGR)
cv2.imwrite("Binary_Threshold.jpg",bi_threshold)


# ### Binary Inverse Threshold

# In[6]:


ret,bi_inv_threshold = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
bi_inv_threshold = cv2.cvtColor(bi_inv_threshold, cv2.COLOR_RGB2BGR)
cv2.imwrite("BinaryInv_Threshold.jpg",bi_inv_threshold)


# ### Threshold Truncation

# In[7]:


ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
thresh3 = cv2.cvtColor(thresh3, cv2.COLOR_RGB2BGR)
cv2.imwrite("TruncThreshold.jpg",thresh3)


# ### Adaptive Threshold

# In[9]:


image = cv2.imread("data/crossword.jpg",0)


# In[12]:


image_adapThresh1 = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,8) 
image_adapThresh1 = cv2.cvtColor(image_adapThresh1, cv2.COLOR_RGB2BGR)
cv2.imwrite("AdapThreshold1.jpg",image_adapThresh1)


# In[13]:


image_adapThresh2 = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,8)
image_adapThresh2 = cv2.cvtColor(image_adapThresh2, cv2.COLOR_RGB2BGR)
cv2.imwrite("AdapThreshold2.jpg",image_adapThresh2)

