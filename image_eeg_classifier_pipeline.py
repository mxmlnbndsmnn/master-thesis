# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 11:48:53 2022

@author: User

Full pipeline:
- load eeg data for one subject recording (or more?)
- obtain a list of trial data and a list of labels
- calculate STFT/CWT images for selected channels per trial
- X = list of num_trials x num_channels x num_times
- y = list of labels
- use k-fold cross validation to split the lists k times
- feed them into a CNN
"""

from os import path as os_path
import numpy as np
import matplotlib.pyplot as plt
from eeg_data_loader import eeg_data_loader
from create_eeg_image import create_stft_for_channel, create_ctw_for_channel
from confusion_matrix import get_confusion_matrix, plot_confusion_matrix, calculate_cm_scores
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
# from sklearn.model_selection import StratifiedKFold
from sys import exit


# EEG data source
eeg_data_folder = "A large MI EEG dataset for EEG BCI"
# subject_data_file = "5F-SubjectA-160405-5St-SGLHand.mat"
subject_data_file = "5F-SubjectC-151204-5St-SGLHand.mat"
# subject_data_file = "5F-SubjectB-151110-5St-SGLHand.mat"
# subject_data_file = "5F-SubjectF-160209-5St-SGLHand.mat"
subject_data_path = os_path.join(eeg_data_folder, subject_data_file)

eeg_data_loader_instance = eeg_data_loader()
eeg_data = eeg_data_loader_instance.load_eeg_from_mat(subject_data_path)
sample_frequency = eeg_data_loader_instance.sample_frequency
num_samples = eeg_data_loader_instance.num_samples
events = eeg_data_loader_instance.find_all_events()

# channel names
ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
            'A1', 'A2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz', 'X3']
# note that X3 is only for synchronization and A1, A2 are ground leads

# channels closest to the primary motor cortex
# F3, Fz, F4, C3, Cz, C4, P3, Pz, P4
# ch_picks = [2, 3, 4, 5, 6, 7, 18, 19, 20]
# pick all channels except reference and Fp1, Fp2
ch_picks = [2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# print([ch_names[i] for i in ch_picks])

# obtain trial data and labels for this subject
def get_trials_x_and_y(eeg_data, events, sfreq, duration=1., prefix_time=0.2,
                       suffix_time=0.2, downsample_step=1, ch_picks=None):
  # reshape eeg data -> n_channels x n_times
  transposed_eeg_data = eeg_data.transpose()
  X = list()
  y = list()
  
  # trial duration is actually variable, but cannot be determined precisely anyway
  trial_frames = int(duration * sfreq)
  
  # (optional) start a bit earlier + extend
  prefix_frames = int(sfreq * prefix_time)
  affix_frames = int(sfreq * suffix_time)
  
  for event in events:
    start_i = event['start'] - prefix_frames
    
    # no trial data before time 0 (should be given implicit, but who knows)
    if start_i < 0:
      print("get_trials_x_and_y: skip trial (start_i < 0)")
      continue
    # assert start_i >= 0
    
    # stop_i = event['stop']
    stop_i = start_i + trial_frames + affix_frames
    
    # for ch_index in ch_picks:
      # transposed_eeg_data[ch_index][start_i:stop_i:downsample_step]
    # this is equal to:
    # for all picked channels, select all frames (with downsample step) from start to stop
    trial = np.array([transposed_eeg_data[ch_index][start_i:stop_i:downsample_step] for ch_index in ch_picks])
    
    # filter outliers with too large amplitude
    if trial.max() > 200 or trial.min() < -200:
      print("Skip trial with amplitude > 200.")
      continue
    
    X.append(trial)
    
    # event type (1-5)
    y.append(event['event'])
  
  return X, y

downsample_step = 1
trials, labels = get_trials_x_and_y(eeg_data, events, sample_frequency,
                                    downsample_step=downsample_step,
                                    ch_picks=ch_picks)

# X = np.array(trials)
# y = np.array(labels) - 1  # labels should range from 0-4 (?)

num_classes = 5

# generate the "images" per channel for all trials
list_of_trial_data = []
list_of_labels = []

# CTW images
for trial, label in zip(trials, labels):
  trial_data = []
  for ch in trial:
    cwt = create_ctw_for_channel(ch, widths_max=25)
    trial_data.append(cwt)
    
    # note for pretty images use another (no) cmap
    # plt.imshow(cwt, cmap="Greys", vmax=abs(cwt).max(), vmin=-abs(cwt).max())
    # plt.title('CWT')
    # plt.show()
  list_of_trial_data.append(trial_data)
  list_of_labels.append(label)


# STFT images
if False:
  # nperseg for stft generation, trade-off between time and frequency resolution
  stft_window = 60
  print(f"STFT window size: {stft_window}")

  for trial, label in zip(trials, labels):
    trial_data = []
    for ch_index in ch_picks:
      ch = trial[ch_index]
      # nperseg should divide the channel shape (240) without remainder
      # smaller nperseg = better time resolution, higher = better frequency resolution
      stft, f, t = create_stft_for_channel(ch, sample_frequency=sample_frequency, nperseg=stft_window)
      trial_data.append(stft)
      
      # note that if the dimensions of (t,f) are equal to those of stft, cannot use flat shading
      # note for pretty images use another (no) cmap and maybe gouraud shading
      # plt.pcolormesh(t, f, stft, cmap="Greys", shading='nearest')
      # plt.title('STFT Magnitude')
      # plt.ylabel('Frequency [Hz]')
      # plt.xlabel('Time [sec]')
      # plt.show()
    list_of_trial_data.append(trial_data)
    list_of_labels.append(label)

X = np.array(list_of_trial_data)
# print("X:", type(X), X.shape)
# should be (num_trials, num_channels, num_f, num_t)

y = np.array(list_of_labels) - 1  # 0-4 instead of 1-5

# some layers (conv, max pool) assume the input to have the channels as the
# last dimension (channels_last), e.g. (batch, dim1, dim2, channel)
X = tf.constant(X)
# print(X[0].shape) # e.g. (9, 17, 7)
X = tf.transpose(X, perm=[0,2,3,1])
print(X[0].shape) # e.g. (17, 7, 9)
input_shape = X[0].shape


# stratified k-fold cv
# k = 5
# skf = StratifiedKFold(n_splits=k)
# for train, test in skf.split(X, y):
#   print(len(train))
#   print(train.shape)
#   print(len(test))
#   print(test.shape)
#   print(len(train)+len(test))
# exit()

# TODO
# do some numpy magic to only pick the selected indices for each train/valid set
# create a dataset for each fold, no need to use take and skip

dataset = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))

num_trials = len(dataset)
print(f"Trials: {num_trials}")

if False:
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
  ds = ds.batch(16)
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
# precision_per_class = [0] * num_classes
# recall_per_class = [0] * num_classes

# sum over all confusion matrices
cumulative_cm = None

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
  
  model.add(layers.Conv2D(30, 5, padding='same', activation='elu', input_shape=input_shape))
  print(model.output_shape)
  model.add(layers.BatchNormalization())
  model.add(layers.MaxPooling2D(pool_size=(1,3)))
  model.add(layers.Dropout(0.3))
  model.add(layers.Conv2D(60, 7, padding='same', activation='elu'))
  model.add(layers.BatchNormalization())
  model.add(layers.MaxPooling2D(pool_size=(1,3)))
  model.add(layers.Dropout(0.3))
  model.add(layers.Conv2D(90, 7, padding='same', activation='elu'))
  model.add(layers.BatchNormalization())
  model.add(layers.MaxPooling2D(pool_size=(1,3)))
  model.add(layers.Dropout(0.3))
  model.add(layers.Flatten())
  # model.add(layers.Dense(64, activation='elu'))
  model.add(layers.Dense(num_classes, activation='softmax'))
  
  exit()
  
  # instantiate an optimizer
  learn_rate = 0.001
  print(f"Learn rate: {learn_rate}")
  optimizer = tf.keras.optimizers.Adagrad(
    learning_rate=learn_rate, initial_accumulator_value=0.1, epsilon=1e-07,
    name='Adagrad')
  
  # compile the model
  model.compile(optimizer=optimizer,
                # from_logits=True (if not using a softmax activation as last layer)
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=["accuracy"])
  
  # view the layers of the model
  # model.summary()
  
  # early stopping
  # monitors the validation accuracy and stops training after [patience] epochs
  # that show no improvements
  es_callback = tf.keras.callbacks.EarlyStopping(monitor="accuracy",
                                                 patience=4,
                                                 restore_best_weights=True)
  
  # train the model
  num_epochs = 10
  print(f"Training for up to {num_epochs} epochs.")
  history = model.fit(train_ds, validation_data=valid_ds, epochs=num_epochs,
                      verbose=0, callbacks=[es_callback])
  
  # visualize training
  # using early stopping, the actual number of epochs might be lower than num_epochs!
  true_num_epochs = len(history.history["loss"])
  if true_num_epochs < num_epochs:
    print(f"Early stop training after epoch {true_num_epochs}")
  visualize_training(history, true_num_epochs)
  
  # calculate average metrics
  # get the highest accuracy for validation + training acc for the same epoch
  best_valid_acc = np.array(history.history['val_accuracy']).max()
  best_valid_epoch = np.array(history.history['val_accuracy']).argmax()
  # acc_train = np.array(history.history['accuracy']).max()
  acc_train = np.array(history.history['accuracy'])[best_valid_epoch]
  acc_valid = best_valid_acc
  print(f"Highest validation accuracy ({best_valid_acc:.3f}) at epoch {best_valid_epoch+1}")
  
  print(history.history)
  
  all_acc_train.append(acc_train)
  all_acc_valid.append(acc_valid)
  
  true_labels = []
  for sample_batch, label_batch in valid_ds:
    for label in label_batch:
      true_labels.append(label.numpy())
  
  predicted_labels = []
  predictions = model.predict(valid_ds)
  for prediction in predictions:
    # score = tf.nn.softmax(prediction).numpy()
    # since the last layer already uses a softmax activation
    # there is no need to calculate the softmax again!
    predicted_labels.append(np.argmax(prediction))
    # here, one could also query the "confidence" with wich the network decides
    # which class it chooses by looking at the softmax value for each class
    # (predictions is an array of size num_classes)
  cm = get_confusion_matrix(true_labels, predicted_labels)
  plot_confusion_matrix(cm, "Konfusionsmatrix, Fold "+str(i+1))
  
  if cumulative_cm is None:
    cumulative_cm = cm
  else:
    cumulative_cm = cumulative_cm + cm
  
  precision, recall, f_score = calculate_cm_scores(cm)
  print("Precision:", precision, "Mean:", np.array(precision).mean())
  print("Recall:", recall, "Mean:", np.array(recall).mean())
  print("F1 Score:", f_score, "Mean:", np.array(f_score).mean())
  
  break

# only need to print this once
model.summary()

# get the mean accuracy and standard deviation from all folds
all_acc_train = np.array(all_acc_train)
all_acc_valid = np.array(all_acc_valid)
acc_mean_train = all_acc_train.mean()
acc_std_train = all_acc_train.std()
acc_mean_valid = all_acc_valid.mean()
acc_std_valid = all_acc_valid.std()
print(f"Mean accuracy (train) is {acc_mean_train:.3f} and STD is {acc_std_train:.3f}")
print(f"Mean accuracy (valid) is {acc_mean_valid:.3f} and STD is {acc_std_valid:.3f}")

# precision and recall per class
precision, recall, f_score = calculate_cm_scores(cumulative_cm)
print("Mean precision per class:")
for i, p in enumerate(precision):
  print(f"{i}: {p:.2f}")
print("Mean recall per class:")
for i, r in enumerate(recall):
  print(f"{i}: {r:.2f}")

print("Cumulative confusion matrix:")
print(cumulative_cm)

