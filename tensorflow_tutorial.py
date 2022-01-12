# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 10:47:54 2022

@author: User

Tutorial: https://www.tensorflow.org/tutorials/images/classification?hl=en
image classification of different flowers, 5 classes, ~3700 photos
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib

# fetch images
# default location: C:\Users\User\.keras\datasets
# note that I changed the file path -> meh, using the SSD again
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)
# data_dir = pathlib.Path("flower_photos") # use only this to load from E:
image_count = len(list(data_dir.glob('*/*.jpg')))
print(f"#images:{image_count}")

# create the dataset
batch_size = 32
img_height = 180
img_width = 180

# split into train and valid dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# inspect class names
class_names = train_ds.class_names
print(f"classes: {class_names}")

# visualize some data

if False:
   plt.figure(figsize=(10, 10))
   for images, labels in train_ds.take(1):
     for i in range(9):
       ax = plt.subplot(3, 3, i + 1)
       plt.imshow(images[i].numpy().astype("uint8"))
       plt.title(class_names[labels[i]])
       plt.axis("off")

# improve performance by using a cache and prefetching data
# fetch the number of elements to be prefetched dynamically at runtime
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# test: inspect a batch
image_batch, labels_batch = next(iter(train_ds))
first_image = image_batch[0]
print(image_batch.shape) # (32, 180, 180, 3)
# tensor: batch of 32 images of shape 180x180 with 3 color channels
print(np.min(first_image), np.max(first_image)) # (32,)
# tensor: one label per image

# data augmentation layer: add slightly altered copies of all images
data_augmentation = Sequential(
  [
    layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

num_classes = len(class_names)

# create the model
# note that the input_shape should be provided in the first layer
model = Sequential([
  data_augmentation,
  layers.Rescaling(1./255),
  # layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

# compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# view the layers of the model
model.summary()


# train the model
epochs = 16
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)


# visualize training results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# predict on unseen data
sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

img = tf.keras.utils.load_img(
    sunflower_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

