"""
Alexandre Shea et Clément Poulin 
Code pour analyse d'image par transformée de fourier
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
from skimage import color, exposure, transform
from skimage.exposure import equalize_hist


imageLinux = imread('LinuxImageTestFFT.png')[:,:,:3] # Lecture image png

imageLinuxGrise = rgb2gray(imageLinux) # Application filtre gris
plt.figure(num=None, figsize=(8, 6), dpi=80) 
plt.imshow(imageLinuxGrise, cmap='gray');


imageLinuxFFT = np.fft.fftshift(np.fft.fft2(imageLinuxGrise)) # Computation de fft sur img
plt.figure(num=None, figsize=(8,6), dpi=80)
plt.imshow(np.log(abs(imageLinuxFFT)), cmap='gray')