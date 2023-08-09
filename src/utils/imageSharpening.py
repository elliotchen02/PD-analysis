# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 14:14:07 2023

@author: hansd
"""

import cv2 as cv2
import numpy as np

# The movement of the hand causes each individual frame of the video to be hard to landmark
# Thus we are attempting to find reliable ways to sharpen each individual frames within the video
# Hoping that the sharpen edges can improve the precision of the landmarking, improving the reliability of the model

def image_sharpening(img):
    """
    Using filters to sharpen edges on image frames. 
    Possible kernels for edge detection:
    ([1, 0, 1], 
     [0, 0, 0], 
     [-1, 0, 1])

    ([0, -1, 0], 
     [-1, 4, -1], 
     [0, -1, 0])

    ([-1, -1, -1], 
     [-1, 8, -1], 
     [-1, -1, -1])
    """
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_image = cv2.filter2D(img, -1, kernel)
    return sharpened_image
    
    
def wiener_filter(img):
    """
    Utilizing weiner filter to deblur image frames. 
    Possible noise values:
       50000000000, 10000000000
    ------------------------------------------
    """
    # read image as grayscale
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # take fourier transform of grey scale image
    dft = np.fft.fft2(grey_img)

    # get power spectral density of dft = square of magnitude
    pspec = (np.abs(dft))**2
    print(np.amin(pspec))
    print(np.amax(pspec))

    # estimate noise power spectral density
    # need to try different values to achieve compromise between noise reduction and softening/blurring
    noise = 5000000000
    
    # wiener filtering
    wiener = pspec/(pspec+noise)
    wiener = wiener*dft
    
    # dft to restore
    restored = np.fft.ifft2(wiener)
    
    # take real() component
    restored = np.real(restored)
    print(np.amin(restored))
    print(np.amax(restored))
    
    # clip and convert to uint8
    restored = restored.clip(0,255).astype(np.uint8)
    
    # save results
    cv2.imwrite('wienerfilered_',restored)
    
