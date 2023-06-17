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


# Preprocess each image
WIDTH = 256
HEIGHT = 256


# Features removal to reduce dimension for processing.
def format_data(data_dir, filename):
    infile = data_dir + "/" + filename
    # Documentation of PIL.Image https://pillow.readthedocs.io/en/3.1.x/reference/Image.html#the-image-class
    img = PIL.Image.open(infile)

    # Convert into grayscale to speed up processing time
    img = img.convert('L')

    # Resize to speed up processing time
    img = img.resize((WIDTH, HEIGHT))
    np_img = np.asarray(img)

    # Normalise from pixel values from [0,255] to [0,1],
    # https://towardsdatascience.com/understand-data-normalization-in-machine-learning-8ff3062101f0
    np_img = np_img / 255

    # We are making it into a 1D array for simplicity
    pixel_count = np.shape(np_img)[0] * np.shape(np_img)[1]
    np_img = np.reshape(np_img, pixel_count)

    return np_img

# Function to perform data preprocessing of all images in training and validation sets.
#label classes
REAL_CLASS = "real"
FAKE_CLASS = "fake"

#assign label to each data
def load_files(pos_train_dir, neg_train_dir, pos_val_dir, neg_val_dir):
    train_data = []
    train_labels = []
    val_data = []
    val_labels = []
    rel_test_data = []

    print('Processing Positive Labels in Training Set')

    for filename in tqdm(os.listdir(pos_train_dir)):
        #preprocessed data
        np_img = format_data(pos_train_dir, filename)
        #load preprocessed
        train_data.append(np_img)
        #load annotations
        train_labels.append(FAKE_CLASS)

    print('Processing Negative Labels in Training Set')

    for filename in tqdm(os.listdir(neg_train_dir)):
        np_img = format_data(neg_train_dir, filename)
        train_data.append(np_img)
        train_labels.append(REAL_CLASS)

    print('Processing Positive Labels in Validation Set')

    for filename in tqdm(sorted(os.listdir(pos_val_dir))):
        np_img = format_data(pos_val_dir, filename)
        val_data.append(np_img)
        val_labels.append(FAKE_CLASS)

    print('Processing Negative Labels in Validation Set')

    for filename in tqdm(sorted(os.listdir(neg_val_dir))):
        np_img = format_data(neg_val_dir, filename)
        val_data.append(np_img)
        val_labels.append(REAL_CLASS)

    return train_data, train_labels, val_data, val_labels


train_data, train_labels, val_data, val_labels = load_files(POS_TRAIN_DIR, NEG_TRAIN_DIR, POS_VAL_DIR, NEG_VAL_DIR)

model = sklearn.linear_model.LogisticRegression(verbose=True, max_iter = 100)
model.fit(train_data, train_labels)

predictions = model.predict(val_data)
accuracy = sklearn.metrics.accuracy_score(val_labels, predictions) * 100
print("Model accuracy = ", accuracy, "%")

misclass_fake_images = []
misclass_real_images = []

# get index of wrongly classify images
misclass_index = np.where(predictions != val_labels)

for index in misclass_index[0]:
    if val_labels[index] == FAKE_CLASS:
        # False Negative (Fake predicted as Real)
        misclass_fake_images.append(index)
    else:
        # False Positive (Real predicted as Fake)
        misclass_real_images.append(index)

truepos_images = []
truepos_index = np.where(predictions == val_labels)

# True Positive (Correctly classified fake images)
for index in truepos_index[0]:
    if val_labels[index] == FAKE_CLASS:
        # True Negative
        truepos_images.append(index)

print("False Negative Images")
num_to_show = min(10, len(misclass_fake_images))
fig, plot = plt.subplots(1, num_to_show, figsize=(40, 4))
for count, index in enumerate(misclass_fake_images[:num_to_show]):
    reshaped = np.reshape(val_data[index], (WIDTH, HEIGHT))
    plot[count].imshow(reshaped, cmap='gray', vmin=0, vmax=1)

print("False Positive Images")
num_to_show = min(10, len(misclass_real_images))
fig, plot = plt.subplots(1, num_to_show, figsize=(40, 4))
for count, index in enumerate(misclass_real_images[:num_to_show]):
    reshaped = np.reshape(val_data[index], (WIDTH, HEIGHT))
    plot[count].imshow(reshaped, cmap='gray', vmin=0, vmax=1)

print("True Positive Images")
num_to_show = min(10, len(truepos_images))
fig, plot = plt.subplots(1, num_to_show, figsize=(40, 4))
for count, index in enumerate(truepos_images[:num_to_show]):
    reshaped = np.reshape(val_data[index], (WIDTH, HEIGHT))
    plot[count].imshow(reshaped, cmap='gray', vmin=0, vmax=1)