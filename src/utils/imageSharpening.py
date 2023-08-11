# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 14:14:07 2023

@author: hansd
"""
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv2
#import joblib
#import os

import skimage
import scipy.signal

#The movement of the hand causes each individual frame of the video to be hard to landmark
#Thus we are attempting to find reliable ways to sharpen each individual frames within the video
#Hoping that the sharpen edges can improve the precision of the landmarking, improving the reliability of the model

def image_sharpening(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    #this is the kernel for image sharpening
    #the following is the possible kernels for edge detection
    #([1, 0, 1], [0, 0, 0], [-1, 0, 1])
    #([0, -1, 0], [-1, 4, -1], [0, -1, 0])
    #([-1, -1, -1], [-1, 8, -1], [-1, -1, -1])
    sharpened_image = cv2.filter2D(img, -1, kernel)
    return sharpened_image
    
    
#The blur of the image could also be a problem, we could use a deblurring method called the weinerfilter
def wienerFilter(img, noise):
    # read image as grayscale
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # take fourier transform of grey scale image
    dft = np.fft.fft2(grey_img)

    # get power spectral density of dft = square of magnitude
    pspec = (np.abs(dft))**2

    #estimate noise power spectral density
    #Need to try different values to achieve compromise between noise reduction and softening/blurring
    #other possible values 50000000000, 10000000000
    
    # do wiener filtering
    wiener = pspec/(pspec+noise)
    wiener = wiener*dft
    
    # do dft to restore
    restored = np.fft.ifft2(wiener)
    
    # take real() component
    restored = np.real(restored)
    
    # clip and convert to uint8
    restored = restored.clip(0,255).astype(np.uint8)
    
    # save results
    return restored


def imageDeblur(img):
    # Using unsupervised wiener method to deconvolve images for deblurring
    greyImg = skimage.color.rgb2gray(img)
    rng = np.random.default_rng()
    
    psf = np.ones((5, 5)) / 25
    greyImg = scipy.signal.convolve2d(greyImg, psf, 'same')
    greyImg += 0.1 * greyImg.std() * rng.standard_normal(greyImg.shape)

    deconvolved, _ = skimage.restoration.unsupervised_wiener(greyImg, psf)
    
    return deconvolved


def histogram_equalization(img):
    #converting image from rbg color space into HSV colorspace
    HSVimage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    #split - creating array with HSV information of image
    HSVarray = cv2.split(HSVimage)
    
    #HSVarray originally is a tuple, need to be converted to list for item reassignment
    HSVarray = list(HSVarray) 
    
    #histogram equalization of VALUE idex within HSV image
    HSVarray[2] = cv2.equalizeHist(HSVarray[2])
    
    #merging array into new deblurred image
    HSVimageOut = cv2.merge(HSVarray)
    
    #converting image back into rgb
    outputImg = cv2.cvtColor(HSVimageOut, cv2.COLOR_HSV2BGR)
    
    return(outputImg)



if __name__ == '__main__':
    #generate handlandmarker on original image
    img = cv2.imread("screenshot.jpg", cv2.IMREAD_COLOR)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    returnHandmarkerImgFromArray(img)

    #generate handlandmarker on sharpened image
    SharpenedImg = imageSharpening(img)
    cv2.imshow("image", SharpenedImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    returnHandmarkerImgFromArray(SharpenedImg)

    #generate handlandmarker on wiener image
    WienerImg = wienerFilter(img, 50000000000)
    cv2.imshow("image", WienerImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    returnHandmarkerImgFromArray(WienerImg)

    #generate handlandmarker on equalized image
    EqualImg = HistogramEqualization(img)
    cv2.imshow("image", EqualImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    returnHandmarkerImgFromArray(EqualImg)



        
