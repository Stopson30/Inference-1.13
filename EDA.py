#!/usr/bin/env python
# coding: utf-8

# In[42]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from glob import glob
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
import skimage
from skimage import io
from skimage.io import imread, imshow

from random import sample
from matplotlib import image

from itertools import chain
from random import sample 
import scipy
##Import any other packages you may need here


# EDA is open-ended, and it is up to you to decide how to look at different ways to slice and dice your data. A good starting point is to look at the requirements for the FDA documentation in the final part of this project to guide (some) of the analyses you do. 
# 
# This EDA should also help to inform you of how pneumonia looks in the wild. E.g. what other types of diseases it's commonly found with, how often it is found, what ages it affects, etc. 
# 
# Note that this NIH dataset was not specifically acquired for pneumonia. So, while this is a representation of 'pneumonia in the wild,' the prevalence of pneumonia may be different if you were to take only chest x-rays that were acquired in an ER setting with suspicion of pneumonia. 

# Perform the following EDA:
# * The patient demographic data such as gender, age, patient position,etc. (as it is available)
# * The x-ray views taken (i.e. view position)
# * The number of cases including: 
#     * number of pneumonia cases,
#     * number of non-pneumonia cases
# * The distribution of other diseases that are comorbid with pneumonia
# * Number of disease per patient 
# * Pixel-level assessments of the imaging data for healthy & disease states of interest (e.g. histograms of intensity values) and compare distributions across diseases.
# 
# Note: use full NIH data to perform the first a few EDA items and use `sample_labels.csv` for the pixel-level assassements. 

# Also, **describe your findings and how will you set up the model training based on the findings.**

# In[43]:


## Below is some helper code to read data for you.
## Load NIH data
all_xray_df = pd.read_csv('/data/Data_Entry_2017.csv')
all_xray_df.sample(3)

## Load 'sample_labels.csv' data for pixel level assessments
sample_df = pd.read_csv('sample_labels.csv')
sample_df.sample(4)


# In[44]:


## EDA starting with having one column label per disease
all_labels = np.unique(list(chain(*sample_df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
all_labels = [x for x in all_labels if len(x)>0]
print('All Labels ({}): {}'.format(len(all_labels), all_labels))
for c_label in all_labels:
    if len(c_label)>1: # leave out empty labels
        all_xray_df[c_label] = all_xray_df['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)
all_xray_df.sample(3)

data = sample_df['Image Index']


# In[45]:


len(all_labels)


# In[46]:


all_xray_df[all_labels].sum()/len(all_xray_df)


# In[47]:


ax = all_xray_df[all_labels].sum().plot(kind='bar')
ax.set(ylabel = 'Number of Images with Label')


# In[48]:


plt.figure(figsize=(16,6))
all_xray_df[all_xray_df.Infiltration==1]['Finding Labels'].value_counts()[0:30].plot(kind='bar') #most common cooccurrences:


# In[49]:


plt.figure(figsize=(16,6))
all_xray_df[all_xray_df.Pneumonia==1]['Finding Labels'].value_counts()[0:30].plot(kind='bar') #most common cooccurrences:


# In[50]:


all_xray_df[all_xray_df['Patient Age']<100]['Patient Age'].hist(bins=10)
plt.xticks(list(range(0,101,10)))
plt.title('Distribution of Ages')


# In[51]:


all_xray_df[all_xray_df['Patient Age']<100]['Patient Age'][all_xray_df.Pneumonia ==1].hist(bins=10)


# In[52]:


plt.figure(figsize=(6,6))
all_xray_df['Patient Gender'].value_counts().plot(kind='bar')


# In[53]:


plt.figure(figsize=(6,6))
all_xray_df[all_xray_df.Pneumonia ==1]['Patient Gender'].value_counts().plot(kind='bar')


# In[54]:


plt.figure(figsize=(6,6))
all_xray_df[all_xray_df.Effusion ==1]['Patient Gender'].value_counts().plot(kind='bar')


# In[55]:


# some more metrics for metadata to be explored, but meanwhile, below is some imaging intensity values analysis
# problems with reading in pixel data
import pydicom
im = pydicom.dcmread('test1.dcm')
im
plt.imshow(im.pixel_array,cmap='gray')


# In[56]:


plt.figure(figsize=(5,5))
plt.hist(im.pixel_array.ravel(), bins = 256)


# In[57]:


mean_intensity = np.mean(im.pixel_array)
mean_intensity


# In[58]:


std_intensity = np.std(im.pixel_array)
std_intensity


# In[59]:


ax = plt.hist(((im.pixel_array - np.mean(im.pixel_array))/np.std(im.pixel_array)).ravel(), bins=300)
im_title = plt.title('No Finding')
plt.show()


# In[72]:


#comorb = pd.read_csv('comorbidity.csv')
#comorb[comorb.Pneumonia==1]['comorbidity'].value_counts()
#comorb_count= comorb[comorb.Pneumonia==1]['comorbidity'].value_counts().sort_index()
#x = comorb_count.index
#y =comorb_count.values
#comorb[comorb.Pneumonia==1]['comorbidity'].value_counts().sort_index().index

#plt.figure(figsize=(10, 6))
#plt.title('Disease comorbidity')
#sns.barplot(x=comorb_count.index, y=comorb_count.values)


# In[62]:


img =io.imread('/data/images_006/images/00011702_043.png')
imgplot = imshow(img)
img_title = plt.title('Pneumonia')
plt.show


# In[63]:


ax = plt.hist(((img - np.mean(img))/np.std(img)).ravel(), bins=300)
img_title = plt.title('Pneumonia')
plt.show()


# In[64]:


img = io.imread('/data/images_009/images/00019576_024.png')
imgplot = imshow(img)
img_title = plt.title('Infiltration')
plt.show


# In[65]:


ax = plt.hist(((img - np.mean(img))/np.std(img)).ravel(), bins=300)
img_title = plt.title('Infiltration')
plt.show()


# In[66]:


img = io.imread('/data/images_010/images/00020966_000.png')
imgplot = imshow(img)
img_title = plt.title('Atelectasis')
plt.show


# In[67]:


ax = plt.hist(((img - np.mean(img))/np.std(img)).ravel(), bins=300)
img_title = plt.title('Atelectasis')
plt.show()


# In[68]:


img = io.imread('/data/images_004/images/00006875_008.png')
imgplot = imshow(img)
img_title = plt.title('Cardiomegaly')
plt.show


# In[69]:


ax = plt.hist(((img - np.mean(img))/np.std(img)).ravel(), bins=300)
img_title = plt.title('Cardiomegaly')
plt.show()


# In[70]:


img = io.imread('/data/images_007/images/00015799_007.png')
imgplot = imshow(img)
img_title = plt.title('Edema')
plt.show


# In[71]:


ax = plt.hist(((img - np.mean(img))/np.std(img)).ravel(), bins=300)
img_title = plt.title('Edema')
plt.show()


# In[ ]:




