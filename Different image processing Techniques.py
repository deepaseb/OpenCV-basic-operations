#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


img = cv2.imread('data/giraffes.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# In[4]:


kernel = np.ones(shape=(4,4),dtype=np.float32)/10

dst = cv2.filter2D(img,-1,kernel)
dst = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)
cv2.imwrite("Giraffe_Kernel.jpg",dst)


# In[7]:


img = cv2.imread('data/giraffes.jpg',0)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
cv2.imwrite("Giraffe_Sobel.jpg",sobelx)


# In[12]:


img =cv2.imread('data/giraffes.jpg')
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.title('Giaraffes Histograms')
plt.savefig('Giaraffes Histograms.jpg')


# In[ ]:




