import matplotlib.pyplot as plt
import numpy as np


img = plt.imread("carton.jpg")/float(2**8)
plt.imshow(img)
plt.show()


shape = img.shape[:2]

#function that allows you to choose the length of diamiter of the chosen circle of the freq domain
def draw_cicle(shape,diamiter):

    assert len(shape) == 2
    TF = np.zeros(shape,dtype=np.bool)
    center = np.array(TF.shape)/2.0

    for iy in range(shape[0]):
        for ix in range(shape[1]):
            TF[iy,ix] = (iy- center[0])**2 + (ix - center[1])**2 < diamiter **2
    return(TF)


#for low pass filter
TFcircleIN   = draw_cicle(shape=img.shape[:2],diamiter=50)
#for high pass filter
TFcircleOUT  = ~TFcircleIN



#perform FFT on every channel of the original image.
fft_img = np.zeros_like(img,dtype=complex)
for ichannel in range(fft_img.shape[2]):
    fft_img[:,:,ichannel] = np.fft.fftshift(np.fft.fft2(img[:,:,ichannel]))



#function that apply the filter on the freq domain
def filter_circle(TFcircleIN,fft_img_channel):
    temp = np.zeros(fft_img_channel.shape[:2],dtype=complex)
    temp[TFcircleIN] = fft_img_channel[TFcircleIN]
    return(temp)

#list of arrays will carry the freq domain of the image after performing the high pass filter
fft_img_filtered_OUT = []

## for each channel, pass filter
for ichannel in range(fft_img.shape[2]):
    fft_img_channel  = fft_img[:,:,ichannel]
    ## circle OUT, high pass filter
    temp = filter_circle(TFcircleOUT,fft_img_channel)
    fft_img_filtered_OUT.append(temp) 
    
fft_img_filtered_OUT = np.array(fft_img_filtered_OUT)
fft_img_filtered_OUT = np.transpose(fft_img_filtered_OUT,(1,2,0))


#function that allows you to reverse the image to time domain again
def inv_FFT_all_channel(fft_img):
    img_reco = []
    for ichannel in range(fft_img.shape[2]):
        img_reco.append(np.fft.ifft2(np.fft.ifftshift(fft_img[:,:,ichannel])))
    img_reco = np.array(img_reco)
    img_reco = np.transpose(img_reco,(1,2,0))
    return(img_reco)

img_reco_filtered_OUT = inv_FFT_all_channel(fft_img_filtered_OUT)

plt.imshow(np.abs(img_reco_filtered_OUT))
plt.show()