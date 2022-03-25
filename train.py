from multi_unet_model import multi_unet_model  # Uses softmax

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt


n_classes = 3

# Read preprocessed images and labels
train_images = np.load(r"C:\Users\Lincoln\Desktop\images_encoded.npy")
train_masks_input = np.load(r"C:\Users\Lincoln\Desktop\masks_encoded.npy")

"""
# Code below is necessary for inference on new data
labelencoder = LabelEncoder()
n, h, w = train_masks.shape

train_masks_reshaped = train_masks.reshape(-1, 1)
train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)

np.unique(train_masks_encoded_original_shape)

train_images = np.expand_dims(train_images, axis=3)
train_images = normalize(train_images, axis=1)

train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)
"""

# 75% data for training
X_train, X_test, y_train, y_test = train_test_split(train_images, train_masks_input, test_size=0.25, random_state=42)

print(f"Class values in the dataset are {np.unique(y_train)}")  # 0 is the background

train_masks_cat = to_categorical(y_train, num_classes=n_classes)
y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))

test_masks_cat = to_categorical(y_test, num_classes=n_classes)
y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))

IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = X_train.shape[1:]

model = multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)

# Set callbacks to save best model (in terms of val_loss)
check_point = tf.keras.callbacks.ModelCheckpoint('best.h5', monitor='val_loss', save_best_only=True)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train_cat,
                    batch_size=8,
                    verbose='auto',
                    epochs=200,
                    validation_data=(X_test, y_test_cat),
                    shuffle=True,
                    callbacks=[check_point])

model.save('last.h5')

# Plot the training and validation accuracy and loss at each epoch
# Evaluation on unseen data is omitted here, since VRAM will be insufficient on most consumer PCs, please refer to evaluate.py
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['acc']
val_acc = history.history['val_acc']

plt.plot(epochs, acc, 'y', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
