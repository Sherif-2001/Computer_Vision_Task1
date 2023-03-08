import numpy as np
import cv2
import matplotlib.pyplot as plt

# --------------------------------- Histogram -------------------------------------
def histogram(image):
    rows, columns = image.shape
    frequency = np.zeros(256,dtype=int)
    for i in range(rows):
        for j in range(columns):
            frequency[image[i][j]] = frequency[image[i][j]] + 1
    
    cumulative_sum_arr = frequency
    for i in range(1,frequency.size):
        cumulative_sum_arr[i] = cumulative_sum_arr[i] + cumulative_sum_arr[i-1]
    return cumulative_sum_arr 

# --------------------------------- Histogram Equalization -------------------------------------
def equalization(image,cumulative):
    l = 256
    n = image.size
    rows,columns = image.shape
    new_image = np.zeros(image.shape,dtype=int)
    arr = np.zeros(256,dtype=int)
    for i in range(256):
        arr[i] = l / n * cumulative[i] - 1
    for i in range (rows):
        for j in range(columns):
           new_image[i][j] =  arr[image[i][j]]
    return new_image

# --------------------------------- Normalization -------------------------------------
def normalization(image):
    max_level = np.max(image)
    min_level = np.min(image)
    new_image = np.zeros(image.shape,dtype=int)
    rows,columns = image.shape
    for i in range(rows):
        for j in range(columns):
            new_image[i][j] = ((image[i][j] - min_level)/(max_level-min_level))*255
    return new_image

# --------------------------------- Global Thresholding -------------------------------------
def global_threshold(image, thres_value, val_high, val_low):
    img = image.copy()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i,j] > thres_value:
                img[i,j] = val_high
            else:
                img[i,j] = val_low
    return img


# --------------------------------- Local Thresholding -------------------------------------

def calculate_mean(rowStartIndex, rowEndIndex, colStartIndex, colEndIndex, image):
    sum = 0
    for i in range(rowStartIndex,rowEndIndex):
        for j in range(colStartIndex,colEndIndex):
            sum += image[i,j]
    return sum/((rowEndIndex-rowStartIndex)*(colEndIndex-colStartIndex))


def local_threshold(image, val_high, val_low, block_size):
    img = image.copy()
    i=0 
    j=0 
    lastMean = 127
    while i+block_size-1 < image.shape[0]:
        j=0
        while j+block_size-1 < image.shape[1]:
            mean=calculate_mean(i,i+block_size,j,j+block_size,image)
            lastMean = mean
            for k in range(i,i+block_size):
                for l in range(j,j+block_size):
                    if image[k,l] > mean:
                        img[k,l] = val_high
                    else:
                        img[k,l] = val_low
            j+=block_size           
        i+=block_size
        
    for i in range(i,image.shape[0]):
        for j in range(j,image.shape[1]):
            if image[i,j] > lastMean:
                img[i,j] = val_high
            else:
                img[i,j] = val_low
    return img

# --------------------------------- RGB Histograms -------------------------------------

#function that splits rgb image into 3 channels and orders them properly
def rgb_split(image):
    b_channel, g_channel, r_channel = cv2.split(image)
    return r_channel, g_channel, b_channel

#function for combining rgb channels into one image
def rgb_combine(r,g,b):
    image = cv2.merge((r,g,b))
    return image

#change from bgr to rgb
def show_img(image, cmap = "rgb"):
    if cmap == "rgb":
        r,g,b = rgb_split(image)
        image = rgb_combine(r,g,b)
        plt.imshow(image)
    else:
        plt.imshow(image,cmap="gray")

#this function takes the RGB image and type of require conversion and returns a grayscale image
def rgb_to_grayscale(image, type = "UHDTV"):
    r,g,b = rgb_split(image)
    if type == "basic":
       return 1/3*(r+g+b) 
    if type == "NTSC":
        return 0.2989*r+0.5870*g+0.1140*b
    if type == "UHDTV":
        return 0.2627*r+0.6780*g+0.0593*b
    
#generates histogram for the overall image RGB
def rgb_hist_cumulative(image):
    # calculate mean value from RGB channels and flatten to 1D array
    # vals = image.mean(axis=2).flatten()
    #get the rgb channels
    r_vals, g_vals, b_vals = rgb_split(image)
    r_hist = twoD_fast_hist(r_vals)
    g_hist = twoD_fast_hist(g_vals)
    b_hist = twoD_fast_hist(b_vals)
    fig, ax = plt.subplots(2,3)
    
    #plot r g b histograms
    ax[0,0].bar(np.arange(0,256),r_hist,color="red")
    ax[0,1].bar(np.arange(0,256),g_hist,color="green")
    ax[0,2].bar(np.arange(0,256),b_hist,color="blue")

    #plot r g b cumulatives
    ax[1,0].bar(np.arange(0,256),np.cumsum(r_hist),color="red")
    ax[1,1].bar(np.arange(0,256),np.cumsum(g_hist),color="green")
    ax[1,2].bar(np.arange(0,256),np.cumsum(b_hist),color="blue")

    
def twoD_fast_hist(matrix):
    # calculate histogram
    values = matrix.flatten()
    counts, bins = np.histogram(values, range(257))
    return counts