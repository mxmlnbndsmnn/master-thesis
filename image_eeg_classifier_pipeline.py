# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 11:48:53 2022

@author: User

Full pipeline:
- load eeg data for one subject recording (or more?)
- obtain a list of trial data and a list of labels
- calculate STFT images for selected channels per trial
- X = list of num_trials x num_channels x num_times
- y = list of labels
- use k-fold cross validation to split the lists k times
- feed them into a CNN
"""

from os import path as os_path
import numpy as np
import matplotlib.pyplot as plt
from eeg_data_loader import eeg_data_loader
from create_stft_image import create_stft_for_channel
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sys import exit


# EEG data source
eeg_data_folder = "A large MI EEG dataset for EEG BCI"
subject_data_file = "5F-SubjectA-160405-5St-SGLHand.mat"
# subject_data_file = "5F-SubjectB-151110-5St-SGLHand.mat"
subject_data_path = os_path.join(eeg_data_folder, subject_data_file)

eeg_data_loader_instance = eeg_data_loader()
eeg_data = eeg_data_loader_instance.load_eeg_from_mat(subject_data_path)
sample_frequency = eeg_data_loader_instance.sample_frequency
num_samples = eeg_data_loader_instance.num_samples

# channel names
ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
            'A1', 'A2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz', 'X3']
# note that X3 is only for synchronization and A1, A2 are ground leads

# channels closest to the primary motor cortex
# F3, Fz, F4, C3, Cz, C4, P3, Pz, P4
ch_picks = [2, 3, 4, 5, 6, 7, 18, 19, 20]
# print([ch_names[i] for i in ch_picks])

# nperseg for stft generation, trade-off between time and frequency resolution
stft_window = 80

# obtain trial data and labels for this subject
trials, labels = eeg_data_loader_instance.get_trials_x_and_y()
# X = np.array(trials)
y = np.array(labels) - 1  # labels should range from 0-4 (?)

num_classes = 5

# generate the "images" per channel for all trials
list_of_trial_data = []
for trial, label in zip(trials, labels):
  trial_data = []
  for ch_index in ch_picks:
    ch = trial[ch_index]
    # nperseg should divide the channel shape (240) without remainder
    # smaller nperseg = better time resolution, higher = better frequency resolution
    stft, f, t = create_stft_for_channel(ch, sample_frequency=sample_frequency, nperseg=stft_window)
    trial_data.append(stft)
    
    # note that if the dimensions of (t,f) are equal to those of stft, cannot use flat shading
    # plt.pcolormesh(t, f, stft, cmap="Greys", shading='nearest')
    # plt.title('STFT Magnitude')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()
  list_of_trial_data.append(trial_data)

X = np.array(list_of_trial_data)
# print("X:", type(X), X.shape)
# should be (num_trials, num_channels, num_f, num_t)

# some layers (conv, max pool) assume the input to have the channels as the
# last dimension (channels_last), e.g. (batch, dim1, dim2, channel)
X = tf.constant(X)
# print(X[0].shape) e.g. (9, 17, 7)
X = tf.transpose(X, perm=[0,2,3,1])
# print(X[0].shape) e.g. (17, 7, 9)
input_shape = X[0].shape

dataset = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))

num_trials = len(dataset)
print(f"Trials: {num_trials}")

# in order to normalize the input, we have to find the value range
# the lower bound is 0, and the upper bound can be found by checking each
# elements maximum value (of the stft data structure)
max_stft_value = 0
for element in dataset.as_numpy_iterator():
  local_max = element[0].max()
  if local_max > max_stft_value:
    max_stft_value = local_max
max_stft_value = int(max_stft_value) + 1
scale_factor = 1/max_stft_value
print(f"max_stft_value: {max_stft_value}")
print(f"scale_factor: {scale_factor}")


# to be called after the dataset has been preprocessed and split for k-fold cv
def configure_for_performance(ds):
  ds = ds.cache()
  # ds = ds.shuffle(buffer_size=len(ds))
  
  # I am not sure about using batches, since a lot of operations that fetch
  # elements from a dataset will yield full batches, instead of single trials
  # however, the model cannot be created when not using batches
  # input shape is (9,17,7) but expected to be (None,9,17,7)
  # passing in (None,9,17,7) as input_shape seems to not work either
  ds = ds.batch(32)
  ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
  return ds


# exit()

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

# split the dataset for k-fold cross-validation
k = 5
valid_size = int(num_trials / k)
print(f"Create {k} folds of size {valid_size}")

# calculate the average metrics
all_acc_train = []
all_acc_valid = []

for i in range(k):
  print("-"*80)
  print(f"Fold {i+1}:")
  train_ds = dataset.take(i*valid_size).concatenate(dataset.skip((i+1)*valid_size))
  valid_ds = dataset.skip(i*valid_size).take(valid_size)
  train_ds = configure_for_performance(train_ds)
  valid_ds = configure_for_performance(valid_ds)
  
  # create the model
  model = Sequential()
  # standardize values to be in range [0,1]
  # this actually drags the accuracy down by a significant amount :o
  # layers.Rescaling(scale_factor, input_shape=input_shape),
  
  model.add(layers.Conv2D(32, 5, padding='same', activation='elu', input_shape=input_shape))
  # print(model.output_shape)
  # model.add(layers.BatchNormalization())
  # model.add(layers.MaxPooling2D())
  # model.add(layers.Dropout(0.4))
  model.add(layers.Conv2D(64, 5, padding='same', activation='elu'))
  model.add(layers.BatchNormalization())
  model.add(layers.MaxPooling2D())
  model.add(layers.Dropout(0.5))
  model.add(layers.Flatten())
  # do we need more than one dense layer?
  model.add(layers.Dense(64, activation='elu'))
  model.add(layers.Dense(num_classes))
  
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
  num_epochs = 100
  print(f"Training for {num_epochs} epochs.")
  history = model.fit(train_ds, validation_data=valid_ds, epochs=num_epochs,
                      verbose=0)
  
  # visualize training
  visualize_training(history, num_epochs)
  
  # calculate average metrics
  # get the highest accuracy for training and validation
  acc_train = np.array(history.history['accuracy']).max()
  acc_valid = np.array(history.history['val_accuracy']).max()
  
  all_acc_train.append(acc_train)
  all_acc_valid.append(acc_valid)
  
  # break


# get the mean accuracy and standard deviation from all folds
all_acc_train = np.array(all_acc_train)
all_acc_valid = np.array(all_acc_valid)
acc_mean_train = all_acc_train.mean()
acc_std_train = all_acc_train.std()
acc_mean_valid = all_acc_valid.mean()
acc_std_valid = all_acc_valid.std()
print(f"Mean accuracy (train) is {acc_mean_train:.3f} and STD is {acc_std_train:.3f}")
print(f"Mean accuracy (valid) is {acc_mean_valid:.3f} and STD is {acc_std_valid:.3f}")

