import cv2
import numpy as np

# function that allows you to choose the length of diamiter of the chosen circle of the freq domain
def draw_cicle(shape,diamiter):
    assert len(shape) == 2
    TF = np.zeros(shape,dtype=np.uint8)
    center = np.array(TF.shape)/2.0

    for iy in range(shape[0]):
        for ix in range(shape[1]):
            TF[iy,ix] = (iy- center[0])**2 + (ix - center[1])**2 < diamiter **2
    return(TF)

# function that allows you to reverse the image to time domain again
def inv_FFT_all_channel(fft_img):
    img_reco = []
    for ichannel in range(fft_img.shape[2]):
        img_reco.append(np.fft.ifft2(np.fft.ifftshift(fft_img[:,:,ichannel])))
    img_reco = np.array(img_reco)
    img_reco = np.transpose(img_reco,(1,2,0))
    return(img_reco)


def hybrid_image(image1,image2):
    # low pass filter for the first image
    TFcircleIN  = draw_cicle(shape=image1.shape[:2],diamiter=50)

    # high pass filter for the second image
    TFcircleIN2   = draw_cicle(shape=image2.shape[:2],diamiter=50)
    TFcircleOUT2  = ~TFcircleIN2


    # perform FFT on every channel of the first original image.
    fft_img = np.zeros_like(image1,dtype=complex)
    for ichannel in range(fft_img.shape[2]):
        fft_img[:,:,ichannel] = np.fft.fftshift(np.fft.fft2(image1[:,:,ichannel]))

    # perform FFT on every channel of the second original image.    
    fft_img2 = np.zeros_like(image2,dtype=complex)
    for ichannel in range(fft_img2.shape[2]):
        fft_img2[:,:,ichannel] = np.fft.fftshift(np.fft.fft2(image2[:,:,ichannel]))    


    # function that apply the filter on the freq domain
    def filter_circle(TFcircleIN,fft_img_channel):
        temp = np.zeros(fft_img_channel.shape[:2],dtype=complex)
        temp[TFcircleIN] = fft_img_channel[TFcircleIN]
        return(temp)

    # list of arrays will carry the freq domain of the first image after performing the low pass filter
    fft_img_filtered_IN = []

    # list of arrays will carry the freq domain of the second image after performing the high pass filter
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

    img_reco_filtered_IN  = inv_FFT_all_channel(fft_img_filtered_IN)
    img_reco_filtered_OUT = inv_FFT_all_channel(fft_img_filtered_OUT)

    hybrid_image = img_reco_filtered_IN + img_reco_filtered_OUT

    return hybrid_image
