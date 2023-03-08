import random
import statistics as stat
import cv2
import numpy as np
import scipy.signal as sig


# --------------------------------- ADD Noise -------------------------------------

def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def unifrom_noise(image):
    x, y = image.shape
    mean = 0
    max = 0.1
    noise = np.zeros((x,y), dtype=np.float64)
    for i in range(x):
        for j in range(y):
            noise[i][j] = np.random.uniform(mean,max)
    noise_img = image + noise
    # noise_img +=255
    noise_img = noise_img*255
    noise_img = np.clip(noise,0,1)
    return noise_img

# --------------------------------- Noise Filters -------------------------------------

def average_filter(image, maskSize = [3,3]):
    # Make average filter mask
    mask = np.ones(maskSize, dtype = int)
    mask = mask / sum(sum(mask))

    # Convolve the image and the mask
    average = sig.convolve2d(image, mask, mode="same")
    average = average.astype(np.uint8)

def gaussian_filter(image, mask_size = 3,sigma = 1):
    # Make gaussian filter mask using gaussian function
    p1 = 1/(2*np.pi*sigma**2)
    p3 = (2*np.square(sigma))
    mask = np.fromfunction(lambda x, y: p1 * np.exp(-(np.square(x-(mask_size-1)/2) + np.square(y-(mask_size-1)/2)) / p3), (mask_size, mask_size))
    mask = mask/np.sum(mask)

    # Convolve the image and the mask
    gaussian = sig.convolve2d(image, mask, mode="same")
    gaussian = gaussian.astype(np.uint8)

def median_filter(image,filter_size = 3):
    # Make an image with the same size of the original
    m, n = image.shape
    filteredImage = np.zeros([m,n])

    # Index that is used for every filter size
    filter_index = filter_size // 2
    for i in range(1, m-1):
        for j in range(1, n-1):
            temp = []
            for k in range(-filter_index,filter_index):
                temp.append(image[i+k, j+k])
            filteredImage[i, j]= stat.median_low(sorted(temp))
    
    filteredImage = filteredImage.astype(np.uint8)

# --------------------------------- Edge Detection Filters -------------------------------------

def canny_edge_detection(img, weak_th = None, strong_th = None):
# defining the canny detector function 
# weak_th and strong_th are thresholds for double thresholding step
       
    # Noise reduction step
    img = cv2.GaussianBlur(img, (5, 5), 1.4)
       
    # Calculating the gradients
    gx = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, 3)
    gy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, 3)
      
    # Conversion of Cartesian coordinates to polar 
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees = True)
       
    # setting the minimum and maximum thresholds 
    # for double thresholding
    mag_max = np.max(mag)
    if not weak_th:weak_th = mag_max * 0.1
    if not strong_th:strong_th = mag_max * 0.5
      
    # getting the dimensions of the input image  
    height, width = img.shape
       
    # Looping through every pixel of the grayscale 
    # image
    for i_x in range(width):
        for i_y in range(height):
               
            grad_ang = ang[i_y, i_x]
            grad_ang = abs(grad_ang-180) if abs(grad_ang)>180 else abs(grad_ang)
               
            # selecting the neighbours of the target pixel
            # according to the gradient direction
            # In the x axis direction
            if grad_ang<= 22.5:
                neighb_1_x, neighb_1_y = i_x-1, i_y
                neighb_2_x, neighb_2_y = i_x + 1, i_y
              
            # top right (diagonal-1) direction
            elif grad_ang>22.5 and grad_ang<=(22.5 + 45):
                neighb_1_x, neighb_1_y = i_x-1, i_y-1
                neighb_2_x, neighb_2_y = i_x + 1, i_y + 1
              
            # In y-axis direction
            elif grad_ang>(22.5 + 45) and grad_ang<=(22.5 + 90):
                neighb_1_x, neighb_1_y = i_x, i_y-1
                neighb_2_x, neighb_2_y = i_x, i_y + 1
              
            # top left (diagonal-2) direction
            elif grad_ang>(22.5 + 90) and grad_ang<=(22.5 + 135):
                neighb_1_x, neighb_1_y = i_x-1, i_y + 1
                neighb_2_x, neighb_2_y = i_x + 1, i_y-1
              
            # Now it restarts the cycle
            elif grad_ang>(22.5 + 135) and grad_ang<=(22.5 + 180):
                neighb_1_x, neighb_1_y = i_x-1, i_y
                neighb_2_x, neighb_2_y = i_x + 1, i_y
               
            # Non-maximum suppression step
            if width>neighb_1_x>= 0 and height>neighb_1_y>= 0:
                if mag[i_y, i_x]<mag[neighb_1_y, neighb_1_x]:
                    mag[i_y, i_x]= 0
                    continue
   
            if width>neighb_2_x>= 0 and height>neighb_2_y>= 0:
                if mag[i_y, i_x]<mag[neighb_2_y, neighb_2_x]:
                    mag[i_y, i_x]= 0
   
    ids = np.zeros_like(img)
       
    # double thresholding step
    for i_x in range(width):
        for i_y in range(height):
              
            grad_mag = mag[i_y, i_x]
              
            if grad_mag<weak_th:
                mag[i_y, i_x]= 0
            elif strong_th>grad_mag>= weak_th:
                ids[i_y, i_x]= 1
            else:
                ids[i_y, i_x]= 2
       
def prewitt_edge_detection(image):
    maskX = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    maskY = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])

    prewittx = sig.convolve2d(image, maskX)
    prewitty = sig.convolve2d(image, maskY)
    prewitt = np.add(prewittx, prewitty)

def roberts_edge_detection(image):
    maskX = np.array([[0,1],[-1,0]])
    maskY = np.array([[1,0],[0,-1]])
    
    image /= 255
    robertsx = sig.convolve2d(image, maskX)
    robertsy = sig.convolve2d(image, maskY)
    roberts = np.add(robertsx ,robertsy)

def sobel_edge_detection(image):
    maskX = [[1, 0, -1],
             [2, 0, -2],
             [1, 0, -1]]
    
    maskY = [[1, 2, 1],
             [0, 0, 0],
             [-1, -2, -1]]
    
    sobelX = sig.convolve2d(image, maskX)
    sobelY = sig.convolve2d(image, maskY )
    sobel = np.add(sobelX,sobelY)
    
    sobel = sobel.astype(np.float64)
