# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 21:03:29 2022

@author: User
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.math import confusion_matrix
import seaborn as sn
# from sklearn.metrics import precision_score, recall_score

# subject_folder = "SubjectC-151204-9ch-cut"
subject_folder = "SubjectF-160210-9ch-HFREQ_2"
print(f"Subject folder: {subject_folder}")

data_dir = pathlib.Path(os.path.join("stft_images", subject_folder))
image_count = len(list(data_dir.glob('*/*.png')))

list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

batch_size = 20
# with trials of different length, the images have different sizes too!
img_height = 27
img_width = 41

class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
print(f"Classes: {class_names}")
num_classes = len(class_names)

###############################################################################

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


# store some test data to obtain labels for the confusion matrix
# use test data only
# cm_test_ds = test_ds.take(-1)
# OR use all data
cm_full_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE).take(-1)


# how does this interfere with k-fold cross-validation splits?
def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=len(ds))
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

###############################################################################

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


###############################################################################

# evaluation

def print_all_scores(score, class_names):
  for i,class_name in enumerate(class_names):
    print(f"{class_name}: {100 * score[i]:10.2f}%")


def print_all_scores(score, class_names):
  for i,class_name in enumerate(class_names):
    print(f"{class_name}: {100 * score[i]:10.2f}%")


# iterate over the (previously copied) test dataset and predict labels
# to create a confusion matrix
def plot_confusion_matrix(dataset, title=None):
  print(f"Create confusion matrix for {len(dataset)} samples.")
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
  f_score = []
  print("Precision:")
  print(precision)
  print("Recall:")
  print(recall)
  for pr, re in zip(precision, recall):
    f1 = 0
    if pr > 0 or re > 0:
      f1 = 2 * (pr*re) / (pr+re)
    f_score.append(f1)
  print("F1 score:")
  print(f_score)


###############################################################################

# split the dataset for k-fold cross-validation
k = 5
valid_size = int(image_count / k)
print(f"Create {k} folds of size {valid_size}")

# calculate the average metrics
all_acc_train = []
all_acc_valid = []

for i in range(k):
  print("-"*80)
  print(f"Fold {i+1}:")
  train_ds = list_ds.take(i*valid_size).concatenate(list_ds.skip((i+1)*valid_size))
  valid_ds = list_ds.skip(i*valid_size).take(valid_size)
  
  print(f"#Training images: {tf.data.experimental.cardinality(train_ds).numpy()}")
  print(f"#Validation images: {tf.data.experimental.cardinality(valid_ds).numpy()}")
  # print(f"#Test images: {tf.data.experimental.cardinality(test_ds).numpy()}")
  
  
  train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
  valid_ds = valid_ds.map(process_path, num_parallel_calls=AUTOTUNE)
  # test_ds = test_ds.map(process_path, num_parallel_calls=AUTOTUNE)
  
  
  train_ds = configure_for_performance(train_ds)
  valid_ds = configure_for_performance(valid_ds)
  # test_ds = configure_for_performance(test_ds)
  
  # fetch a single image + label - not working?
  # only one return value that contains a tensor (img) and a 2nd tensor (label)
  # image_batch, label_batch = next(iter(valid_ds))
  # print(f"1st label: {label_batch.numpy()}")
  
  
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
    layers.Dropout(0.1),
    layers.Conv2D(32, 3, padding='same', activation='elu'),
    # layers.MaxPooling2D(),
    layers.Dropout(0.1),
    layers.Conv2D(64, 3, padding='same', activation='elu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.1),
    layers.Conv2D(128, 3, padding='same', activation='elu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.1),
    layers.Flatten(),
    # layers.Dense(128, activation='elu'),
    layers.Dense(32, activation='elu'),
    layers.Dense(num_classes)
  ])
  
  
  # instantiate an optimizer
  learn_rate = 0.001
  print(f"Learn rate: {learn_rate}")
  optimizer = tf.keras.optimizers.Adagrad(
    learning_rate=learn_rate, initial_accumulator_value=0.1, epsilon=1e-07,
    name='Adagrad')
  
  # compile the model
  model.compile(optimizer=optimizer,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
  
  # view the layers of the model
  model.summary()
  
  # train the model
  num_epochs = 60
  print(f"Training for {num_epochs} epochs.")
  history = model.fit(train_ds, validation_data=valid_ds, epochs=num_epochs,
                      verbose=0)
  
  
  visualize_training(history, num_epochs)
  
  # get the most recent accuracy for training and validation
  # acc_train = history.history['accuracy'][-1]
  # acc_valid = history.history['val_accuracy'][-1]
  
  # alt: get the highest accuracy for training and validation
  acc_train = np.array(history.history['accuracy']).max()
  acc_valid = np.array(history.history['val_accuracy']).max()
  
  all_acc_train.append(acc_train)
  all_acc_valid.append(acc_valid)
  
  cm_title = f"Fold {i+1}"
  plot_confusion_matrix(cm_full_ds, title=cm_title)


# get the mean accuracy and standard deviation from all folds
all_acc_train = np.array(all_acc_train)
all_acc_valid = np.array(all_acc_valid)
acc_mean_train = all_acc_train.mean()
acc_std_train = all_acc_train.std()
acc_mean_valid = all_acc_valid.mean()
acc_std_valid = all_acc_valid.std()
print(f"Mean accuracy (train) is {acc_mean_train} and STD is {acc_std_train}")
print(f"Mean accuracy (valid) is {acc_mean_valid} and STD is {acc_std_valid}")
# reminder: arithmetic mean = average
# but there are other types of mean...

# evaluate the model using a test dataset
# not needed when using CV
# if False:
#   print(f"Evaluate model on {len(test_ds)} test samples:")
#   result = model.evaluate(test_ds)
#   result_dict = dict(zip(model.metrics_names, result))
#   print(result_dict)


# predict on a test dataset (details)
# not needed when using CV
# if False:
#   predictions = model.predict(test_ds)
#   for prediction in predictions:
#     score = tf.nn.softmax(prediction).numpy()
#     print_all_scores(score, class_names)



# plot_confusion_matrix(cm_full_ds)
# plot_confusion_matrix(cm_test_ds)

