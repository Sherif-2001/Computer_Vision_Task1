import random
import statistics as stat
import cv2
import numpy as np
import scipy.signal as sig


# --------------------------------- ADD Noise -------------------------------------

def salt_pepper_noise(image,prob = 1/100):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    noisy_image = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                noisy_image[i][j] = 0
            elif rdn > thres:
                noisy_image[i][j] = 255
            else:
                noisy_image[i][j] = image[i][j]
    return noisy_image

def gaussian_noise(image):
    row,col= image.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    noisy_image = image + gauss
    return noisy_image

def unifrom_noise(image):
    x, y = image.shape
    mean = 0
    max = 0.1
    noise = np.zeros((x,y), dtype=np.float64)
    for i in range(x):
        for j in range(y):
            noise[i][j] = np.random.uniform(mean,max)
    noisy_img = image + noise
    # noise_img +=255
    noisy_img = noisy_img*255
    noisy_img = np.clip(noise,0,1)
    return noisy_img

# --------------------------------- Noise Filters -------------------------------------

def average_filter(image, maskSize = [3,3]):
    # Make average filter mask
    mask = np.ones(maskSize, dtype = int)
    mask = mask / sum(sum(mask))

    # Convolve the image and the mask
    filtered_image = sig.convolve2d(image, mask, mode="same")
    return filtered_image

def gaussian_filter(image, mask_size = 3,sigma = 1):
    # Make gaussian filter mask using gaussian function
    p1 = 1/(2*np.pi*sigma**2)
    p3 = (2*np.square(sigma))
    mask = np.fromfunction(lambda x, y: p1 * np.exp(-(np.square(x-(mask_size-1)/2) + np.square(y-(mask_size-1)/2)) / p3), (mask_size, mask_size))
    mask = mask/np.sum(mask)

    # Convolve the image and the mask
    filtered_image = sig.convolve2d(image, mask, mode="same")
    return filtered_image

def median_filter(image,filter_size = 3):
    # Make an image with the same size of the original
    row, col = image.shape
    filtered_image = np.zeros([row+2,col+2])

    # Index that is used for every filter size
    filter_index = filter_size // 2
    for i in range(1, row-1):
        for j in range(1, col-1):
            temp = []
            for k in range(-filter_index,filter_index):
                temp.append(image[i+k, j+k])
            filtered_image[i, j]= stat.median_low(sorted(temp))
    
    return filtered_image

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
    magnitude, ang = cv2.cartToPolar(gx, gy, angleInDegrees = True)
       
    # setting the minimum and maximum thresholds 
    # for double thresholding
    mag_max = np.max(magnitude)
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
                if magnitude[i_y, i_x]<magnitude[neighb_1_y, neighb_1_x]:
                    magnitude[i_y, i_x]= 0
                    continue
   
            if width>neighb_2_x>= 0 and height>neighb_2_y>= 0:
                if magnitude[i_y, i_x]<magnitude[neighb_2_y, neighb_2_x]:
                    magnitude[i_y, i_x]= 0
   
    ids = np.zeros_like(img)
       
    # double thresholding
    for i_x in range(width):
        for i_y in range(height):
              
            grad_mag = magnitude[i_y, i_x]
              
            if grad_mag<weak_th:
                magnitude[i_y, i_x]= 0
            elif strong_th>grad_mag>= weak_th:
                ids[i_y, i_x]= 1
            else:
                ids[i_y, i_x]= 2

    return magnitude

def prewitt_edge_detection(image):
    maskX = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    maskY = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])

    prewittx = sig.convolve2d(image, maskX)
    prewitty = sig.convolve2d(image, maskY)
    prewitt = np.add(prewittx, prewitty)
    return prewitt

def roberts_edge_detection(image):
    maskX = np.array([[0,1],[-1,0]])
    maskY = np.array([[1,0],[0,-1]])
    
    image /= 255
    robertsx = sig.convolve2d(image, maskX)
    robertsy = sig.convolve2d(image, maskY)
    roberts = np.add(robertsx ,robertsy)
    return roberts

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
    return sobel
