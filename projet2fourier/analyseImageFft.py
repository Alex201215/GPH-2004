import numpy as np
import scipy as sc
import cv2
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
from skimage import color, exposure, transform
from skimage.exposure import equalize_hist


class ImageCamera:

    # Fonction d'initialisation
    def __init__(self, fichier_avec_path=None, image_deja_fft=None):
        self.path_fichier = fichier_avec_path
        self.path_fftdeja = image_deja_fft

        if image_deja_fft != None:
            self.image_fft = rgb2gray(imread(f"{self.path_fftdeja}")[:, :, :3])
        else:
            self.image_fft = None

        self.image_ifft = None
        self.image_originale = rgb2gray(imread(f"{self.path_fichier}")[:, :, :3])  # Image originale

    # Calcul de la transformé de fourier
    def calcul_fft(self):
        self.image_fft = np.fft.fftshift(np.fft.fft2(self.image_originale))
        return self.image_fft

    # Calcul de la transformé inverse
    def calcul_ifft(self):
        # Si l'instance de classe n'a pas d'image fft
        if len(self.image_fft) <=1  and len(self.image_ifft) <= 1:
            
            raise ValueError("Il n'y a aucune fft associée à l'objet")

        elif len(self.image_fft) > 1:
            self.image_ifft = np.fft.ifft2(np.fft.ifftshift(self.image_fft))
            return self.image_ifft

        elif len(self.path_fftdeja) > 1:
            self.image_ifft = np.fft.ifft2(np.fft.ifftshift(self.image_deja_fft))
            return self.image_deja_if

    # Fonction qui gère la compression d'image
    def resize_image(self, taille_voulue):
        pass

    # Affiche la transformé de fourier de l'image objet
    def afficher_fft(self, num=None):
        plt.figure(num=None, figsize=(8, 6), dpi=80)
        plt.imshow(np.log(abs(self.image_fft)), cmap='gray')

    # Affiche la transformé de fourier inverse du array fft
    def afficher_ifft(self, num=None):
        plt.figure(num=None, figsize=(8, 6), dpi=80)
        plt.imshow((abs(self.image_ifft)), cmap='gray')
        # plt.imshow(np.log(abs(self.image_ifft)), cmap='gray')

    # Affiche l'image originale sans aucun traitement
    def afficher_image_originale(self):
        plt.figure(num=None, figsize=(8, 6), dpi=80)
        plt.imshow(self.image_originale, cmap='gray')


objet1 = ImageCamera("..\TPOP\projet2fourier\LinuxImageTestFFT.png")
objet1_fft = objet1.calcul_fft()
objet1.afficher_fft()
objet1.afficher_image_originale()
objet1_ifft = objet1.calcul_ifft()
objet1.afficher_ifft()
# plt.show()
