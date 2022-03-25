from multi_unet_model import multi_unet_model  # Uses softmax

import tensorflow as tf
from tensorflow.keras.utils import normalize, to_categorical
from tensorflow.keras.metrics import MeanIoU
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt

import os
import glob
import random


n_classes = 3


train_images = np.load(r"C:\Users\Lincoln\Desktop\images_encoded.npy")
train_masks_input = np.load(r"C:\Users\Lincoln\Desktop\masks_encoded.npy")

# Create a subset of data for quick testing
# Picking 10% for testing and remaining for training
X1, X_test, y1, y_test = train_test_split(train_images, train_masks_input, test_size=0.10, random_state=42)
print(X1.shape, y1.shape)

# Further split training data t a smaller subset for quick testing of models
X_train, X_do_not_use, y_train, y_do_not_use = train_test_split(X1, y1, test_size=0.2, random_state=42)

train_masks_cat = to_categorical(y_train, num_classes=n_classes)
y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))

test_masks_cat = to_categorical(y_test, num_classes=n_classes)
y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))

model = tf.keras.models.load_model('test.hdf5')

for index in range(32):
    test_img_number = index     # random.randint(0, len(X_test))
    test_img = X_test[test_img_number]
    ground_truth = y_test[test_img_number]
    test_img_norm = test_img[:, :, 0][:, :, None]
    test_img_input = np.expand_dims(test_img_norm, 0)
    prediction = (model.predict(test_img_input))
    predicted_img = np.argmax(prediction, axis=3)[0, :, :]

    plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test_img[:, :, 0], cmap='gray')
    plt.subplot(232)
    plt.title('Testing Label')
    plt.imshow(ground_truth[:, :, 0], cmap='jet')
    plt.subplot(233)
    plt.title('Prediction on test image')
    plt.imshow(predicted_img, cmap='jet')
    plt.show()
