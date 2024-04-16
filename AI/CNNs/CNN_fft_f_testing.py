import torch as t
import torchvision as tv
import torchvision.datasets as datasets
import torchvision.io
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import random
import glob
import os

from torch import nn
from torchinfo import summary
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from timeit import default_timer as timer
from pathlib import Path
from PIL import Image

from CNN_fft_f import CNN_FFT
from functions import *


""" AGNOSTIC CODE """

device = "cuda" if t.cuda.is_available() else "cpu"


""" LOAD CUSTOM IMAGE """

data_transform = transforms.Compose([
    transforms.Resize(size=(64, 64), antialias=True),
    transforms.ConvertImageDtype(t.float32)
])

custom_image_path = r"C:\Clément MSI\Code\Python\PyTorch\Learn PyTorch\TPOP - Projet 2\ImageFFTGrid.png"
custom_image_float32_resized = data_transform(torchvision.io.read_image(custom_image_path))

plt.imshow(custom_image_float32_resized.permute(1, 2, 0), cmap="gray")
plt.show()

custom_image = (custom_image_float32_resized.to(device)).unsqueeze(0)
class_names = ['full grid', 'lines']


""" LOAD THE MODEL """

# create model save
MODEL_PATH = Path(r"C:\Clément MSI\Code\Python\PyTorch\Learn PyTorch\models - State_dicts")
MODEL_NAME = "CNN_fft.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# instantiate a new instance of our model class
loaded_CNN_fft_f = CNN_FFT(input_shape=3,
                           hidden_units=10,
                           output_shape=2).to(device)

# Load the saved state_dict() of LinearRegressionModel_1 (this will update the new instance with updated parameters)
loaded_CNN_fft_f.load_state_dict(t.load(f=MODEL_SAVE_PATH))

# put the loaded model on the GPU :
loaded_CNN_fft_f.to(device)

# print(loaded_CNN_fft_f.state_dict())


""" SUMMARIZE MODEL """

# summary(loaded_CNN_fft_f, input_size=[1, 3, 64, 64])


""" TEST 1 IMAGE """

loaded_CNN_fft_f.eval()
with t.inference_mode():
    custom_preds = loaded_CNN_fft_f(custom_image)

custom_probs = t.softmax(custom_preds, dim=1)
custom_labels = t.argmax(custom_probs, dim=1).cpu()
print(f"\nCustom probabilities: {custom_probs}\nPrediction: {class_names[custom_labels]}")
