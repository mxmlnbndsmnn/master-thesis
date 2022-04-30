# -*- coding: utf-8 -*-
"""
Use one session for training and a different one for testing.
"""

import sys
from os import path as os_path
import time
import numpy as np
import scipy.signal as signal
from eeg_data_loader import eeg_data_loader
from create_eeg_image import create_ctw_for_channel
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.math import confusion_matrix

print(sys.version)
print("Available GPUs:")
gpu_list = tf.config.list_physical_devices('GPU')
print(gpu_list)
if len(gpu_list) < 1:
  sys.exit()

run_index = 0

if len(sys.argv) == 1:
  print("Warning: Missing parameter [run_index]")
  sys.exit()

run_index = int(sys.argv[1])


eeg_data_folder = "eeg-data"

# use the first file for train+validation and the second for testing
# sample datasets from subjects C and E for inter-subject session-to-session training
files_c = ["5F-SubjectC-151204-5St-SGLHand.mat",
           "5F-SubjectC-160429-5St-SGLHand-HFREQ.mat"]
files_e = ["5F-SubjectE-160321-5St-SGLHand-HFREQ.mat",
           "5F-SubjectE-160415-5St-SGLHand-HFREQ.mat",
           "5F-SubjectE-160429-5St-SGLHand-HFREQ.mat"]

# create 2x6 combinations
subject_data_files = []
for f_c in files_c:
  for f_e in files_e:
    subject_data_files.append([f_c, f_e])
for f_e in files_e:
  for f_c in files_c:
    subject_data_files.append([f_e, f_c])

# data used for training
subject_data_file_1 = subject_data_files[run_index][0]
subject_data_path_1 = os_path.join(eeg_data_folder, subject_data_file_1)
print(f"Load subject data from path: {subject_data_path_1} (training)")

# data used for testing
subject_data_file_2 = subject_data_files[run_index][1]
subject_data_path_2 = os_path.join(eeg_data_folder, subject_data_file_2)
print(f"Load subject data from path: {subject_data_path_2} (testing)")

start_time_load_data = time.perf_counter()

# training data
eeg_data_loader_instance_1 = eeg_data_loader()
eeg_data_1 = eeg_data_loader_instance_1.load_eeg_from_mat(subject_data_path_1)
sample_frequency_1 = eeg_data_loader_instance_1.sample_frequency
num_samples_1 = eeg_data_loader_instance_1.num_samples

if sample_frequency_1 == 200:
  trials_1, labels_1 = eeg_data_loader_instance_1.get_trials_x_and_y()
elif sample_frequency_1 == 1000:
  # downsample using every 5th data point to go from 1000Hz to 200Hz
  trials_1, labels_1 = eeg_data_loader_instance_1.get_trials_x_and_y_downsample(5)
  # use the downsampled frequency...
  sample_frequency_1 = 200
  print("Downsample from 1000Hz to 200Hz.")
else:
  raise RuntimeError("Unexpected sample frequency:", sample_frequency_1)

y_1 = np.array(labels_1) - 1  # labels should range from 0-4 (?)

# testing data
eeg_data_loader_instance_2 = eeg_data_loader()
eeg_data_2 = eeg_data_loader_instance_2.load_eeg_from_mat(subject_data_path_2)
sample_frequency_2 = eeg_data_loader_instance_2.sample_frequency
num_samples_2 = eeg_data_loader_instance_2.num_samples

if sample_frequency_2 == 200:
  trials_2, labels_2 = eeg_data_loader_instance_2.get_trials_x_and_y()
elif sample_frequency_2 == 1000:
  # downsample using every 5th data point to go from 1000Hz to 200Hz
  trials_2, labels_2 = eeg_data_loader_instance_2.get_trials_x_and_y_downsample(5)
  # use the downsampled frequency...
  sample_frequency_2 = 200
  print("Downsample from 1000Hz to 200Hz.")
else:
  raise RuntimeError("Unexpected sample frequency:", sample_frequency_2)

y_2 = np.array(labels_2) - 1  # labels should range from 0-4 (?)


end_time_load_data = time.perf_counter()
print(f"Time to load EEG-data: {end_time_load_data-start_time_load_data:.2f}s")

###############################################################################

num_classes = 5

ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
            'A1', 'A2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz', 'X3']
print("Use EEG channels:")
# pick all channels except reference and Fp1, Fp2
ch_picks = [2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20]
print([ch_names[i] for i in ch_picks])

###############################################################################

# apply butterworth bandpass filter

# second-order sections
def butter_bandpass_sos(lowcut, highcut, sample_freq, order=3):
  nyq = sample_freq * 0.5
  low = lowcut / nyq
  high = highcut / nyq
  sos = signal.butter(order, [low, high], analog=False, btype="bandpass", output="sos")
  return sos


# default axis is -1, but here we want to filter data for each channel
def butter_bandpass_filter(data, lowcut, highcut, sample_freq, order=3, axis=1):
  sos = butter_bandpass_sos(lowcut, highcut, sample_freq, order=order)
  y = signal.sosfilt(sos, data, axis=axis)
  return y

# print("Bandpass filter EEG data (4-40Hz)")
# start_time_bandpass = time.perf_counter()
# eeg_data_1 = butter_bandpass_filter(eeg_data_1, 4.0, 40.0, sample_frequency_1, order=6, axis=1)
# eeg_data_2 = butter_bandpass_filter(eeg_data_2, 4.0, 40.0, sample_frequency_2, order=6, axis=1)

# end_time_bandpass = time.perf_counter()
# print(f"Time to apply bandpass filter: {end_time_bandpass-start_time_bandpass:.2f}s")

###############################################################################

# generate the "images" per channel for all trials
list_of_trial_data_1 = []
list_of_trial_data_2 = []

# CTW images
start_time_cwt = time.perf_counter()

# train data
for trial, label in zip(trials_1, labels_1):
  trial_data = []
  for ch_index in ch_picks:
    ch = trial[ch_index]
    cwt = create_ctw_for_channel(ch, widths_max=30)
    trial_data.append(cwt)
  list_of_trial_data_1.append(trial_data)

X_1 = np.array(list_of_trial_data_1)
print("X1:", type(X_1), X_1.shape)
# should be (num_trials, num_channels, num_f, num_t)

# test data
for trial, label in zip(trials_2, labels_2):
  trial_data = []
  for ch_index in ch_picks:
    ch = trial[ch_index]
    cwt = create_ctw_for_channel(ch, widths_max=30)
    trial_data.append(cwt)
  list_of_trial_data_2.append(trial_data)

X_2 = np.array(list_of_trial_data_2)
print("X2:", type(X_2), X_2.shape)

end_time_cwt = time.perf_counter()
print(f"Time to generate CWTs: {end_time_cwt-start_time_cwt:.2f}s")

###############################################################################

# some layers (conv, max pool) assume the input to have the channels as the
# last dimension (channels_last), e.g. (batch, dim1, dim2, channel)
X_1 = tf.constant(X_1)
X_1 = tf.transpose(X_1, perm=[0,2,3,1])
input_shape = X_1[0].shape

dataset_1 = tf.data.Dataset.from_tensor_slices((tf.constant(X_1), tf.constant(y_1)))

num_trials_1 = len(dataset_1)
print(f"Total number of trials: {num_trials_1} (training)")

# do the same for the test data
X_2 = tf.constant(X_2)
X_2 = tf.transpose(X_2, perm=[0,2,3,1])
input_shape = X_2[0].shape

dataset_2 = tf.data.Dataset.from_tensor_slices((tf.constant(X_2), tf.constant(y_2)))

num_trials_2 = len(dataset_2)
print(f"Total number of trials: {num_trials_2} (testing)")

###############################################################################

# to be called after the dataset has been preprocessed and split
def configure_for_performance(ds):
  ds = ds.cache()
  # ds = ds.shuffle(buffer_size=len(ds))
  ds = ds.batch(16)
  ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
  return ds


# calculate precision, recall and f1 score from confusion matrix
def calculate_cm_scores(cm):
  tp = cm.diagonal()
  tp_and_fn = cm.sum(1)
  tp_and_fp = cm.sum(0)
  # note: might run into div by zero!
  precision = tp / tp_and_fp # how many predictions for this class are correct
  recall = tp / tp_and_fn # how many of the class trials have been found
  f_score = []
  for pr, re in zip(precision, recall):
    f1 = 0
    if pr > 0 or re > 0:
      f1 = 2 * (pr*re) / (pr+re)
    f_score.append(f1)
  return precision, recall, f_score

###############################################################################

learn_rate = 0.001
print(f"Learn rate: {learn_rate}")
num_epochs = 80
print(f"Training for up to {num_epochs} epochs.")

start_time_train = time.perf_counter()

num_training_trials = int(0.8 * num_trials_1)
train_ds = dataset_1.take(num_training_trials)
valid_ds = dataset_1.skip(num_training_trials)
train_ds = configure_for_performance(train_ds)
valid_ds = configure_for_performance(valid_ds)

test_ds = dataset_2.take(-1)
test_ds = configure_for_performance(test_ds)

model = Sequential()

model.add(layers.Conv2D(30, 5, padding="same", activation="elu", input_shape=input_shape))
# print(model.output_shape)
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(3,1)))
model.add(layers.Dropout(0.3))
model.add(layers.Conv2D(60, 7, padding="same", activation="elu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(3,1)))
model.add(layers.Dropout(0.3))
model.add(layers.Conv2D(90, 7, padding="same", activation="elu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(3,1)))
model.add(layers.Dropout(0.3))
model.add(layers.Flatten())
# model.add(layers.Dense(32, activation="elu"))
model.add(layers.Dense(num_classes, activation="softmax"))

# instantiate an optimizer
optimizer = tf.keras.optimizers.Adagrad(
  learning_rate=learn_rate, initial_accumulator_value=0.1, epsilon=1e-07,
  name='Adagrad')

# compile the model
model.compile(optimizer=optimizer,
              # from_logits=True (if not using a softmax activation as last layer)
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"])

# early stopping
# monitors the validation accuracy and stops training after [patience] epochs
# that show no improvements
es_callback = tf.keras.callbacks.EarlyStopping(monitor="accuracy",
                                               patience=4,
                                               restore_best_weights=True)

# train the model
history = model.fit(train_ds, validation_data=valid_ds, epochs=num_epochs,
                    verbose=0, callbacks=[es_callback])

# using early stopping, the actual number of epochs might be lower than num_epochs!
true_num_epochs = len(history.history["loss"])
if true_num_epochs < num_epochs:
  print(f"Early stop training after epoch {true_num_epochs}")

# calculate average metrics
# get the highest accuracy for validation + training acc for the same epoch
best_valid_epoch = np.array(history.history['val_accuracy']).argmax()
best_valid_acc = np.array(history.history['val_accuracy']).max()
acc_valid = best_valid_acc
# get the train acc for the same epoch as the valid acc
acc_train = np.array(history.history['accuracy'])[best_valid_epoch]
print(f"Highest validation accuracy ({best_valid_acc:.3f}) at epoch "\
      f"{best_valid_epoch+1} (training accuracy is {acc_train:.3f})")
# print(history.history)

end_time_train = time.perf_counter()
print(f"Time to train models: {end_time_train-start_time_train:.2f}s")

model.summary()

###############################################################################

# evaluate the model on the validation data
true_labels = []
for sample_batch, label_batch in valid_ds:
  for label in label_batch:
    true_labels.append(label.numpy())

predicted_labels = []
predictions = model.predict(valid_ds)
for prediction in predictions:
  predicted_labels.append(np.argmax(prediction))
cm = confusion_matrix(true_labels, predicted_labels).numpy()
print("Confusion matrix (validation data)")
print(cm)

# precision and recall per class
precision, recall, f_score = calculate_cm_scores(cm)
print("Mean precision per class:")
for i, p in enumerate(precision):
  print(f"{i}: {p:.2f}")
print("Mean recall per class:")
for i, r in enumerate(recall):
  print(f"{i}: {r:.2f}")
print("Mean F1 score per class:")
for i, r in enumerate(f_score):
  print(f"{i}: {r:.2f}")

###############################################################################

print("-"*80)

# evaluate the model by testing it with data from another session
true_test_labels = []
for sample_batch, label_batch in test_ds:
  for label in label_batch:
    true_test_labels.append(label.numpy())

predicted_test_labels = []
predictions_test = model.predict(test_ds)
for prediction in predictions_test:
  predicted_test_labels.append(np.argmax(prediction))
cm_test = confusion_matrix(true_test_labels, predicted_test_labels).numpy()
print("Confusion matrix (test data)")
print(cm_test)

true_positives = cm_test.diagonal().sum()
acc_test = true_positives/num_trials_2
print(f"Test accuracy: {acc_test:.3f}")


# precision and recall per class
precision, recall, f_score = calculate_cm_scores(cm_test)
print("Mean precision per class:")
for i, p in enumerate(precision):
  print(f"{i}: {p:.2f}")
print("Mean recall per class:")
for i, r in enumerate(recall):
  print(f"{i}: {r:.2f}")
print("Mean F1 score per class:")
for i, r in enumerate(f_score):
  print(f"{i}: {r:.2f}")
