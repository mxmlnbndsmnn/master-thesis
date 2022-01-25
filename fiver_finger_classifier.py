# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 21:03:29 2022

@author: User
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import pathlib
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


data_dir = pathlib.Path(os.path.join("stft_images", "SubjectC-151204-9ch-cut"))
image_count = len(list(data_dir.glob('*/*.png')))

list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

batch_size = 40
img_height = 9
img_width = 117

class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
print(f"Classes: {class_names}")
num_classes = len(class_names)

# split the dataset
train_size = int(image_count * 0.8)
valid_size = int(image_count * 0.1)
# 80% training
train_ds = list_ds.take(train_size)
# 10% validation
valid_ds = list_ds.skip(train_size).take(valid_size)
# 10% test
test_ds = list_ds.skip(train_size+valid_size)

print(f"#Training images: {tf.data.experimental.cardinality(train_ds).numpy()}")
print(f"#Validation images: {tf.data.experimental.cardinality(valid_ds).numpy()}")
print(f"#Test images: {tf.data.experimental.cardinality(test_ds).numpy()}")


def get_label(file_path):
    # Convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    one_hot = parts[-2] == class_names
    # Integer encode the label
    return tf.argmax(one_hot)


def decode_img(img):
    # Convert the compressed string to a 3D uint8 tensor
    # channels: 3 = RGB, 1 = grayscale
    channels = 3
    img = tf.io.decode_png(img, channels=channels)
    # Resize the image to the desired size
    return tf.image.resize(img, [img_height, img_width])


def process_path(file_path):
    label = get_label(file_path)
    # Load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
valid_ds = valid_ds.map(process_path, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.map(process_path, num_parallel_calls=AUTOTUNE)


def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds


train_ds = configure_for_performance(train_ds)
valid_ds = configure_for_performance(valid_ds)
test_ds = configure_for_performance(test_ds)


# create the model
model = Sequential([
  # standardize values to be in range [0,1]
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  # layers.Rescaling(1./255),
  # data_augmentation:
  # layers.RandomZoom(-0.2),
  # layers.RandomTranslation(0., (-0.1, 0.1)),
  # layers.RandomCrop(img_height, int(img_width*0.8)),
  
  layers.Conv2D(16, 3, padding='same', activation='elu'),
  # layers.MaxPooling2D(),
  # layers.Dropout(0.1),
  layers.Conv2D(32, 3, padding='same', activation='elu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.1),
  layers.Flatten(),
  # layers.Dense(128, activation='elu'),
  layers.Dense(32, activation='elu'),
  layers.Dense(num_classes)
])


# instantiate an optimizer
optimizer = tf.keras.optimizers.Adagrad(
  learning_rate=0.001, initial_accumulator_value=0.1, epsilon=1e-07,
  name='Adagrad')

# compile the model
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# view the layers of the model
model.summary()

# train the model
num_epochs = 80
history = model.fit(train_ds, validation_data=valid_ds, epochs=num_epochs,
                    verbose=0)


# visualize training results
def visualize_training(history, num_epochs):
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']
  
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  
  epochs_range = range(num_epochs)
  
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


visualize_training(history, num_epochs)


# evaluation

def print_all_scores(score, class_names):
  for i,class_name in enumerate(class_names):
    print(f"{class_name}: {100 * score[i]:10.2f}%")



# evaluate the model using a test dataset
if True:
  result = model.evaluate(test_ds)
  result_dict = dict(zip(model.metrics_names, result))
  print(result_dict)


# predict on a test dataset (details)
if False:
  predictions = model.predict(test_ds)
  for prediction in predictions:
    score = tf.nn.softmax(prediction).numpy()
    print_all_scores(score, class_names)


