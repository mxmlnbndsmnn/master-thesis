# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 11:04:42 2022

@author: User

Processing pipeline to load eeg data and train one model per class to perform
one-versus-rest classification
Dataset is split into train, valid and test (no CV)
Labels for train + valid data are mapped for each (rest) class
Each trial in the test dataset is then fed to all 5 models
If exactly one model classifies the trial as "his class", this class is chosen
Otherwise - if the models do not agree - the class with the higher
"confidence" is selected (higher value of the prediction score)
"""

import sys
from os import path as os_path
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from eeg_data_loader import eeg_data_loader
from create_eeg_image import create_ctw_for_channel
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.math import confusion_matrix

print(sys.version)
# print("Available GPUs:")
# gpu_list = tf.config.list_physical_devices('GPU')
# print(gpu_list)
# if len(gpu_list) < 1:
#   sys.exit()


# EEG data source
eeg_data_folder = "A large MI EEG dataset for EEG BCI"
subject_data_file = "5F-SubjectC-151204-5St-SGLHand.mat"
subject_data_path = os_path.join(eeg_data_folder, subject_data_file)

eeg_data_loader_instance = eeg_data_loader()
eeg_data = eeg_data_loader_instance.load_eeg_from_mat(subject_data_path)
sample_frequency = eeg_data_loader_instance.sample_frequency
num_samples = eeg_data_loader_instance.num_samples

# channel names
ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
            'A1', 'A2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz', 'X3']

# pick all channels except reference and Fp1, Fp2
ch_picks = [2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20]
print("Use EEG channels:")
print([ch_names[i] for i in ch_picks])

# obtain trial data and labels for this subject
if sample_frequency == 200:
  trials, labels = eeg_data_loader_instance.get_trials_x_and_y()
elif sample_frequency == 1000:
  # downsample using every 5th data point to go from 1000Hz to 200Hz
  trials, labels = eeg_data_loader_instance.get_trials_x_and_y_downsample(5)
  # use the downsampled frequency...
  sample_frequency = 200
  print("Downsample from 1000Hz to 200Hz.")
else:
  raise RuntimeError("Unexpected sample frequency:", sample_frequency)

trials = np.array(trials)
labels = np.array(labels) - 1  # labels should range from 0-4

###############################################################################

# generate CTW images once
list_of_trial_data = []
for trial, label in zip(trials, labels):
  trial_data = []
  for ch_index in ch_picks:
    ch = trial[ch_index]
    cwt = create_ctw_for_channel(ch, widths_max=30)
    trial_data.append(cwt)
  list_of_trial_data.append(trial_data)

trials = np.array(list_of_trial_data)
print("trials:", type(trials), trials.shape)
# should be (num_trials, num_channels, num_f, num_t)
num_trials = trials.shape[0]

# create an X and y for all 5 models
# this allows to undersample trials of each "rest" class
Xs = []
ys = []
# randomly select trials with a 25% chance
rng = default_rng()
debug_label_counter = []
for target_class in range(5):
  print(f"Compare class {target_class} vs the rest")
  print("Trials in target class =", (labels==target_class).sum())
  print("Trials in other classes =", (labels!=target_class).sum())
  
  Xs.append([])
  ys.append([])
  debug_label_counter.append([0, 0, 0, 0, 0])
  
  for trial, label in zip(trials, labels):
    fake_label = 0
    chance = 1.0
    if label == target_class:
      # pick all trials for the target class
      # chance = 1.0
      fake_label = 0
    else:
      # for the rest classes, only pick aprox every 4th
      chance = 0.25
      fake_label = 1
    
    if chance == 1.0 or chance > rng.random():
      Xs[target_class].append(trial)
      ys[target_class].append(fake_label)
      debug_label_counter[target_class][label] += 1

# they should be roughly even in numbers (except for each target class)
print("Debug: #labels per model per class:")
print(debug_label_counter)

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

def configure_for_performance(ds):
  ds = ds.cache()
  # ds = ds.shuffle(buffer_size=len(ds))
  ds = ds.batch(16)
  ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
  return ds

###############################################################################

# calculate precision, recall and f1 score from confusion matrix
def calculate_cm_scores(cm):
  tp = cm.diagonal()
  tp_and_fn = cm.sum(1)
  tp_and_fp = cm.sum(0)
  # note: might run into div by zero!
  precision = tp / tp_and_fp # how many predictions for this class are correct
  recall = tp / tp_and_fn # how many of the class trials have been found
  f_score = []
  # print("Precision:")
  # print(precision)
  # print("Recall:")
  # print(recall)
  for pr, re in zip(precision, recall):
    f1 = 0
    if pr > 0 or re > 0:
      f1 = 2 * (pr*re) / (pr+re)
    f_score.append(f1)
  # print("F1 score:")
  # print(f_score)
  return precision, recall, f_score

###############################################################################

# create one model per class
models = []
learn_rate = 0.001
print(f"Learn rate: {learn_rate}")
num_epochs = 8
print(f"Training for up to {num_epochs} epochs.")

train_size = int(0.6 * num_trials)
valid_size = int(0.2 * num_trials)
test_size = int(0.2 * num_trials)

for target_class in range(5):
  
  # some layers (conv, max pool) assume the input to have the channels as the
  # last dimension (channels_last), e.g. (batch, dim1, dim2, channel)
  X = tf.constant(Xs[target_class])
  X = tf.transpose(X, perm=[0,2,3,1])
  input_shape = X[0].shape

  y = ys[target_class]
  dataset = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
  
  train_ds = dataset.take(train_size)
  valid_ds = dataset.skip(train_size).take(valid_size)
  test_ds = dataset.skip(train_size+valid_size).take(test_size)
  train_ds = configure_for_performance(train_ds)
  valid_ds = configure_for_performance(valid_ds)
  test_ds = configure_for_performance(test_ds)
  
  print(f"Train model {target_class+1}/5 ...")
  
  # create the model
  model = Sequential()
  
  model.add(layers.Conv2D(32, 5, padding='same', activation='elu', input_shape=input_shape))
  # print(model.output_shape)
  model.add(layers.Conv2D(64, 5, padding='same', activation='elu'))
  # model.add(layers.BatchNormalization())
  model.add(layers.MaxPooling2D())
  model.add(layers.Dropout(0.5))
  model.add(layers.Flatten())
  model.add(layers.Dense(2, activation='softmax'))
  
  # instantiate an optimizer
  optimizer = tf.keras.optimizers.Adagrad(
    learning_rate=learn_rate, initial_accumulator_value=0.1, epsilon=1e-07,
    name='Adagrad')
  
  # compile the model
  model.compile(optimizer=optimizer,
                # sparse categorical crossentropy for integer labels
                # non-sparse for one-hot encodings (not used here)
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=["accuracy"])
  
  # early stopping
  es_callback = tf.keras.callbacks.EarlyStopping(monitor="accuracy",
                                                 patience=4,
                                                 restore_best_weights=True)
  
  # train the model
  history = model.fit(train_ds, validation_data=valid_ds, epochs=num_epochs,
                      verbose=0, callbacks=[es_callback])
  print(history.history.keys())
  
  true_num_epochs = len(history.history["loss"])
  if true_num_epochs < num_epochs:
    print(f"Early stop training after epoch {true_num_epochs}")
  visualize_training(history, true_num_epochs)
  
  # check the accuracy this model reached during training
  best_valid_acc = np.array(history.history['val_accuracy']).max()
  best_valid_epoch = np.array(history.history['val_accuracy']).argmax()
  acc_train = np.array(history.history['accuracy'])[best_valid_epoch]
  acc_valid = best_valid_acc
  print(f"Highest validation accuracy ({best_valid_acc:.3f}) at epoch {best_valid_epoch+1}")
  
  # also get precision and recall for this model (validation dataset)
  true_labels = []
  for sample_batch, label_batch in valid_ds:
    for label in label_batch:
      true_labels.append(label.numpy())
  
  predicted_labels = []
  predictions = model.predict(valid_ds)
  for prediction in predictions:
    predicted_labels.append(np.argmax(prediction))
  cm = confusion_matrix(true_labels, predicted_labels).numpy()
  print(f"Confusion matrix {target_class+1}:")
  print(cm)
  precision, recall, f_score = calculate_cm_scores(cm)
  print("Mean precision per class:")
  for i, p in enumerate(precision):
    print(f"{i}: {p:.2f}")
  print("Mean recall per class:")
  for i, r in enumerate(recall):
    print(f"{i}: {r:.2f}")
  
  # store the model for this target class to be used for predictions later
  models.append(model)

###############################################################################

# get the true labels (class 0-4) for all entries in the test set
true_test_labels = labels[-test_size:]

# iterate over the test dataset with each model to do 5 one-vs-rest queries
binary_predictions = []
for target_class, model in zip(range(5), models):
  predictions = model.predict(test_ds)
  binary_predictions.append(predictions)

print("-"*80)
final_predictions = []  # class 0-4; combined from all binary classifiers
for index, prediction in zip(range(test_size), predictions):
  print(f"Trial nr. {index+1}")
  confidences = []  # for this one trial; per target class (= per model)
  num_claims = 0
  for target_class, model in zip(range(5), models):
    prediction = binary_predictions[target_class][index]
    # how certain is this model, that this trial belongs to it's target class?
    confidence = prediction[0]
    confidences.append(confidence)
    # class 0 = target class, class 1 = "rest class"
    if prediction.argmax() == 0:
      # confidence = prediction.max()  # in this case, this is equal to prediction[0]
      num_claims += 1
      # print(f"Model {target_class} claims trial {index} belongs to this class.")
      # print(f"Confidence: {confidence }")
  
  # confidences for all classes have been collected, choose the largest
  confidences = np.array(confidences)
  print(confidences.round(3))
  selected_class = confidences.argmax()
  final_predictions.append(selected_class)
  if num_claims == 1:
    # exactly one model claims this trial belongs to it's class
    print("1 model identified the class.")
  else:
    # no model or more than one model think this trial belongs to it's class
    # select the model (class) with the highest confidence
    print(f"{num_claims} models identified the class. Pick one.")
  
  correct_class = true_test_labels[index]
  if selected_class == correct_class:
    print(f"Selected correct class: {selected_class}")
  else:
    print(f"Select class: {selected_class}, true label is: {correct_class}")
  print("-"*80)
  
  # break

# now one can compare the final_predictions with the true_test_labels
final_predictions = np.array(final_predictions)
# print(final_predictions==true_test_labels)

cm = confusion_matrix(true_test_labels, final_predictions).numpy()
print("Confusion matrix:")
print(cm)

tp = cm.diagonal().sum()
acc_test = tp / test_size
print(f"Test accuracy: {acc_test:.3f}")

precision, recall, f_score = calculate_cm_scores(cm)
print("Mean precision per class:")
for i, p in enumerate(precision):
  print(f"{i}: {p:.2f}")
print("Mean recall per class:")
for i, r in enumerate(recall):
  print(f"{i}: {r:.2f}")
print("Mean F1 score per class:")
for i, f1 in enumerate(f_score):
  print(f"{i}: {f1:.2f}")

# for trial in test_ds:
#   for target_class, model in zip(range(5), models):
#     prediction = model(trial, training=False)
#     print(target_class)
#     print(prediction)
#   break

