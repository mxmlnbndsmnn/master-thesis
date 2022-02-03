# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 20:40:34 2022

@author: User
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import pathlib
import seaborn as sn
# import sys
# import sklearn.metrics

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.math import confusion_matrix


data_dir = pathlib.Path("parts3")
image_count = len(list(data_dir.glob('*/*.jpg')))

list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

batch_size = 10
# H x W could be 150x300
img_height = 100
img_width = 120

do_convert_to_grayscale = False

# for f in list_ds.take(5):
#   print(f.numpy())

# obtain class names from folder names
# ['3021' '30359' '3666' '3709' '6233']
class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
num_classes = len(class_names)
print(f"{image_count} images in {num_classes} classes:")
print(f"Classes: {class_names}")

# split dataset
if True:
  train_size = int(image_count * 0.7)
  valid_size = int(image_count * 0.15)
  # 70% training
  train_ds = list_ds.take(train_size)
  # 15% validation
  valid_ds = list_ds.skip(train_size).take(valid_size)
  # 15% test
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
    if do_convert_to_grayscale:
      channels = 1
    img = tf.io.decode_jpeg(img, channels=channels)
    # Resize the image to the desired size
    return tf.image.resize(img, [img_height, img_width])


def process_path(file_path):
    label = get_label(file_path)
    # Load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
if True:
  AUTOTUNE = tf.data.AUTOTUNE
  train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
  valid_ds = valid_ds.map(process_path, num_parallel_calls=AUTOTUNE)
  test_ds = test_ds.map(process_path, num_parallel_calls=AUTOTUNE)

# for image, label in train_ds.take(1):
#     print("Image shape: ", image.numpy().shape)
#     print("Label: ", label.numpy())


# store some test data to obtain labels for the confusion matrix
# should be done before the configure_for_performance call(?)
# use test data only
cm_test_ds = test_ds.take(-1)
# use all data
cm_full_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE).take(-1)


def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=len(ds))
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds


if True:
  train_ds = configure_for_performance(train_ds)
  valid_ds = configure_for_performance(valid_ds)
  test_ds = configure_for_performance(test_ds)


# this only works after the performance configuration stuff
if True:
  image_batch, label_batch = next(iter(train_ds))
  plt.figure(figsize=(10, 10))
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[i].numpy().astype("uint8"))
    label = label_batch[i]
    plt.title(class_names[label])
    plt.axis("off")



# if do_convert_to_grayscale: shape 1 instead of 3

def create_model():
  num_channels = 3
  if do_convert_to_grayscale:
    num_channels = 1
  
  # data augmentation layer: add slightly altered copies of all images
  data_augmentation = Sequential(
    [
      layers.RandomFlip("horizontal", input_shape=(img_height, img_width, num_channels)),
      # layers.RandomFlip("horizontal"),
      layers.RandomRotation(0.1),
      layers.RandomZoom(0.1),
    ]
  )
  
  # create the model
  # note that the input_shape should be provided in the first layer
  model = Sequential([
    # standardize values to be in range [0,1]
    layers.Rescaling(1./255, input_shape=(img_height, img_width, num_channels)),
    # data_augmentation,
    # layers.Rescaling(1./255),
    layers.Conv2D(16, 3, padding='same', activation='elu'),
    # layers.MaxPooling2D(),
    # layers.Dropout(0.1),
    layers.Conv2D(32, 3, padding='same', activation='elu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.1),
    layers.Conv2D(64, 3, padding='same', activation='elu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.1),
    layers.Flatten(),
    layers.Dense(128, activation='elu'),
    layers.Dense(num_classes)
  ])
  
  # last results with this model:
  # about 90% accuracy, no big difference between valid and train acc
  # confusion matrix looks good
  # some errors between 30359 and 3666
  # note though that the learn rate is low, images are larger -> takes time
  
  return model


if True:
  model = create_model()
  model.summary()
  
  # 0.001 seems a bit too high? - val acc jumps up and down
  learn_rate = 0.0008
  print(f"Learn rate: {learn_rate}")
  
  # instantiate an optimizer
  optimizer = tf.keras.optimizers.Adagrad(
    learning_rate=learn_rate, initial_accumulator_value=0.1, epsilon=1e-07,
    name='Adagrad')
  
  # compile the model
  model.compile(optimizer=optimizer,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
  
  # view the layers of the model
  # model.summary()
  
  # train the model
  num_epochs = 30
  print(f"Training for {num_epochs} epochs.")
  history = model.fit(train_ds, validation_data=valid_ds, epochs=num_epochs,
                      verbose=0)


# save it to a folder structure
# note that it cannot be saved when using the data_augmentation layers!
# moving the layers directly into the model creation did not help
# model.save("parts_models/1")

# load a saved model
# model = tf.keras.models.load_model('parts_models/1')


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


# predict on unseen data

def print_highest_score(score, class_names):
  print(
      "This image most likely belongs to {} with a {:.2f} percent confidence."
      .format(class_names[np.argmax(score)], 100 * np.max(score))
  )


def print_all_scores(score, class_names):
  for i,class_name in enumerate(class_names):
    print(f"{class_name}: {100 * score[i]:10.2f}%")


# print all scores above a certain limit
def print_class_scores_above(score, class_names, min_score_percent=20):
  num_classes = len(class_names)
  score_dict = dict()
  for i in range(num_classes):
    score_dict[i] = score[i]
  
  sorted_scores = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
  # note that if no score is high enough, nothing gets displayed!
  for item in sorted_scores:
    this_score = round(item[1] * 100, 2)
    if this_score < min_score_percent:
      break
    print(f"Class: {class_names[item[0]]} - Score: {this_score}%")


# alternative: only display the top N elements
def print_top_n_scores(score, class_names, num_top_scores=3):
  score_dict = dict()
  for i in range(num_classes):
    score_dict[i] = score[i]
  sorted_scores = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
  for i in range(num_top_scores):
    print(f"{class_names[sorted_scores[i][0]]} {sorted_scores[i][1]:.2f}%")


# works:
# predictions = model.predict(test_ds)
# predictions contains the scores for all images in the test_ds

# predict class of a single image
def predict_single_image(model, image_path):
  image = tf.keras.preprocessing.image.load_img(image_path)
  image = tf.image.resize(image, [img_height, img_width])
  # if using only one channel (grayscale)
  if do_convert_to_grayscale:
    image = tf.image.rgb_to_grayscale(image)
  input_arr = tf.keras.preprocessing.image.img_to_array(image)
  input_arr = np.array([input_arr])  # Convert single image to a batch.
  predictions = model.predict(input_arr)
  score = tf.nn.softmax(predictions[0]).numpy()
  return score


# os.path.join("parts", "30359", "20220112_190626_027.jpg")
# image_path = os.path.join("parts", "3021", "20220112_190746_017.jpg")
# score = predict_single_image(model, image_path)

test_image_paths = list()
test_image_paths.append(os.path.join("parts", "30359", "20220112_190626_027.jpg"))
test_image_paths.append(os.path.join("parts", "3021", "20220112_190746_017.jpg"))
test_image_paths.append(os.path.join("parts", "6249", "00235.jpg"))
test_image_paths.append(os.path.join("unknown_image.jpg")) # 3666

for image_path in test_image_paths:
  score = predict_single_image(model, image_path)
  
  print(f"Try to predict class for image {image_path}")
  
  # print result
  print_highest_score(score, class_names)
  
  print("All scores:")
  print_all_scores(score, class_names)
  
  # print("Scores above 20%:")
  # print_class_scores_above(score, class_names, 20)
  
  # print("Top 3 scores:")
  # print_top_n_scores(score, class_names, 3)


# evaluate the model using a test dataset
if True:
  print(f"Evaluate model on {len(test_ds)} test samples:")
  result = model.evaluate(test_ds)
  result_dict = dict(zip(model.metrics_names, result))
  print(result_dict)


# works too: predict on multiple images
# TODO: make into list + method
if False:
  image1 = tf.keras.preprocessing.image.load_img(os.path.join("parts", "3021", "20220112_190746_017.jpg"))
  image2 = tf.keras.preprocessing.image.load_img(os.path.join("parts", "30359", "20220112_190626_027.jpg"))
  image1 = tf.image.resize(image1, [img_height, img_width])
  image2 = tf.image.resize(image2, [img_height, img_width])
  input1 = tf.keras.preprocessing.image.img_to_array(image1)
  input2 = tf.keras.preprocessing.image.img_to_array(image2)
  input_arr = np.array([input1, input2])  # Convert to a batch
  predictions = model.predict(input_arr)
  for prediction in predictions:
    score = tf.nn.softmax(prediction).numpy()
    print_class_scores_above(score, class_names, 20)


# visualize some images
# since this ds was not batched, take(1) actually returns one image+label
if False:
  plt.figure(figsize=(10, 10))
  for images, labels in cm_full_ds.skip(9).take(9):
    for i in range(9):
      ax = plt.subplot(3, 3, i + 1)
      plt.imshow(images[i].numpy().astype("uint8"))
      plt.title(class_names[labels[i]])
      plt.axis("off")


# iterate over the (previously copied) test dataset and predict labels
# to create a confusion matrix
def plot_confusion_matrix(dataset, title=None):
  print(f"Create confusion matrix for {len(dataset)} samples (batches).")
  cm_samples = []
  cm_labels = []
  predicted_labels = []
  for sample, label in dataset:
    # print(sample)
    # print(label)
    cm_samples.append(sample)
    cm_labels.append(label.numpy())
  
  predictions = model.predict(np.array(cm_samples))
  for prediction in predictions:
    score = tf.nn.softmax(prediction).numpy()
    # print("Scores:")
    # print_all_scores(score, class_names)
    predicted_labels.append(np.argmax(score))
  
  cm = confusion_matrix(cm_labels, predicted_labels).numpy()
  # print(cm)

  ticklabels = [c for c in class_names]
  sn.heatmap(cm, annot=True, xticklabels=ticklabels, yticklabels=ticklabels,
             cmap='summer', cbar=False, fmt="d")
  plt.xlabel("Predicted labels")
  plt.ylabel("True labels")
  if title is not None:
    plt.title(title)

  plt.show()
  
  # calculate precision and recall from confusion matrix
  tp = cm.diagonal()
  tp_and_fn = cm.sum(1)
  tp_and_fp = cm.sum(0)
  # note: might run into div by zero!
  precision = tp / tp_and_fp # how many predictions for this class are correct
  recall = tp / tp_and_fn # how many of the class trials have been found
  print("Precision:")
  print(precision)
  print("Recall:")
  print(recall)


plot_confusion_matrix(cm_full_ds, title="full set")
plot_confusion_matrix(cm_test_ds, title="test set")

