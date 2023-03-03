#!/usr/bin/env python
# coding: utf-8

# In[165]:


import matplotlib.pyplot as plt
import numpy as np
img = plt.imread("hOIuY.jpg")/float(2**8)
plt.imshow(img)
plt.show()


# In[196]:


img2 = plt.imread("wp4473284.jpg")/float(2**8)
plt.imshow(img2)


# In[197]:


def draw_cicle(shape,diamiter):
    '''
    Input:
    shape    : tuple (height, width)
    diameter : scalar
    
    Output:
    np.array of shape  that says True within a circle with diamiter =  around center 
    '''
    assert len(shape) == 2
    TF = np.zeros(shape,dtype=np.bool)
    center = np.array(TF.shape)/2.0

    for iy in range(shape[0]):
        for ix in range(shape[1]):
            TF[iy,ix] = (iy- center[0])**2 + (ix - center[1])**2 < diamiter **2
    return(TF)

TFcircleIN   = draw_cicle(shape=img.shape[:2],diamiter=50)
TFcircleOUT  = ~TFcircleIN

TFcircleIN2   = draw_cicle(shape=img2.shape[:2],diamiter=50)
TFcircleOUT2  = ~TFcircleIN2


# In[198]:


fft_img = np.zeros_like(img,dtype=complex)
for ichannel in range(fft_img.shape[2]):
    fft_img[:,:,ichannel] = np.fft.fftshift(np.fft.fft2(img[:,:,ichannel]))
    
fft_img2 = np.zeros_like(img2,dtype=complex)
for ichannel in range(fft_img2.shape[2]):
    fft_img2[:,:,ichannel] = np.fft.fftshift(np.fft.fft2(img2[:,:,ichannel]))    


# In[199]:


def filter_circle(TFcircleIN,fft_img_channel):
    temp = np.zeros(fft_img_channel.shape[:2],dtype=complex)
    temp[TFcircleIN] = fft_img_channel[TFcircleIN]
    return(temp)

fft_img_filtered_IN = []
fft_img_filtered_OUT = []
## for each channel, pass filter
for ichannel in range(fft_img.shape[2]):
    fft_img_channel  = fft_img[:,:,ichannel]
    ## circle IN
    temp = filter_circle(TFcircleIN,fft_img_channel)
    fft_img_filtered_IN.append(temp)
    ## circle OUT
#     temp = filter_circle(TFcircleOUT,fft_img_channel)
#     fft_img_filtered_OUT.append(temp) 

## for each channel, pass filter
for ichannel in range(fft_img2.shape[2]):
    fft_img_channel  = fft_img2[:,:,ichannel]
    ## circle IN
#     temp = filter_circle(TFcircleIN,fft_img_channel)
#     fft_img_filtered_IN.append(temp)
    ## circle OUT
    temp = filter_circle(TFcircleOUT2,fft_img_channel)
    fft_img_filtered_OUT.append(temp) 
    
fft_img_filtered_IN = np.array(fft_img_filtered_IN)
fft_img_filtered_IN = np.transpose(fft_img_filtered_IN,(1,2,0))
fft_img_filtered_OUT = np.array(fft_img_filtered_OUT)
fft_img_filtered_OUT = np.transpose(fft_img_filtered_OUT,(1,2,0))


# In[200]:


abs_fft_img              = np.abs(fft_img)
abs_fft_img_filtered_IN  = np.abs(fft_img_filtered_IN)
abs_fft_img_filtered_OUT = np.abs(fft_img_filtered_OUT)


# In[201]:


def inv_FFT_all_channel(fft_img):
    img_reco = []
    for ichannel in range(fft_img.shape[2]):
        img_reco.append(np.fft.ifft2(np.fft.ifftshift(fft_img[:,:,ichannel])))
    img_reco = np.array(img_reco)
    img_reco = np.transpose(img_reco,(1,2,0))
    return(img_reco)


# img_reco              = inv_FFT_all_channel(fft_img)
img_reco_filtered_IN  = inv_FFT_all_channel(fft_img_filtered_IN)
img_reco_filtered_OUT = inv_FFT_all_channel(fft_img_filtered_OUT)


# In[202]:


plt.imshow(np.abs(img_reco_filtered_IN))


# In[203]:


plt.imshow(np.abs(img_reco_filtered_OUT))
img_reco_filtered_OUT.resize(1000,1000)


# In[204]:


hybrid_image = img_reco_filtered_IN + img_reco_filtered_OUT
plt.imshow(np.abs(hybrid_image))


# In[ ]:




