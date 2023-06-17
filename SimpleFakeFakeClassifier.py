import os
import PIL.Image #for image preprocessing
import numpy as np #for multi-dimensional arrays processing
import sklearn.linear_model #contain multiple machine learning functions
#More info: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model
import matplotlib.pyplot as plt

# %matplotlib inline magic command for jupyter notebook.

import scikitplot as skplt #for advance visualisation of model performance

from tqdm import tqdm
from PIL import Image
import gdown

# Looking at Real Image vs Fake Image
print("Flickr Real Image")
# with Image.open('data/train/ffhq/00032.png') as img:
#     img.show()
#
print("StyleGan2 Fake Image")
# with Image.open('data/train/stylegan2/000005.png') as img:
#     img.show()

file_path = "./data"
for dirpath, dirnames, filenames in os.walk(file_path):
    N_c = len(filenames)
    print("No. of files in ", dirpath, "is", N_c)

#Indicates Location of folders containing images
# Training Set
NEG_TRAIN_DIR = "data/train/ffhq"
POS_TRAIN_DIR = "data/train/stylegan2"

# Validation Set
NEG_VAL_DIR = "data/validation/ffhq"
POS_VAL_DIR = "data/validation/stylegan2"