# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 11:50:45 2022

@author: User

raw eeg data + mne + braindecode approach
"""

import os
from sys import exit
import mne
import numpy as np
import time
from braindecode.datasets import create_from_X_y, create_from_mne_raw # used to be in .datautil
from braindecode.models import ShallowFBCSPNet, Deep4Net
from braindecode import EEGClassifier
# from braindecode.preprocessing import exponential_moving_standardize, preprocess, Preprocessor

import torch
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split
from sklearn.model_selection import train_test_split
# from tensorflow.math import confusion_matrix

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from eeg_data_loader import eeg_data_loader
from confusion_matrix import get_confusion_matrix, plot_confusion_matrix, calculate_cm_scores


# EEG data source
eeg_data_folder = "A large MI EEG dataset for EEG BCI"

# all files
subject_data_files = ['5F-SubjectA-160405-5St-SGLHand.mat',  # 0
                      '5F-SubjectA-160408-5St-SGLHand-HFREQ.mat',
                      '5F-SubjectB-151110-5St-SGLHand.mat',
                      '5F-SubjectB-160309-5St-SGLHand-HFREQ.mat',
                      '5F-SubjectB-160311-5St-SGLHand-HFREQ.mat',
                      '5F-SubjectB-160316-5St-SGLHand.mat',
                      '5F-SubjectC-151204-5St-SGLHand.mat',
                      '5F-SubjectC-160429-5St-SGLHand-HFREQ.mat',
                      '5F-SubjectE-160321-5St-SGLHand-HFREQ.mat',
                      '5F-SubjectE-160415-5St-SGLHand-HFREQ.mat',
                      '5F-SubjectE-160429-5St-SGLHand-HFREQ.mat',  # 10
                      '5F-SubjectF-151027-5St-SGLHand.mat',
                      '5F-SubjectF-160209-5St-SGLHand.mat',
                      '5F-SubjectF-160210-5St-SGLHand-HFREQ.mat',
                      '5F-SubjectG-160413-5St-SGLHand-HFREQ.mat',
                      '5F-SubjectG-160428-5St-SGLHand-HFREQ.mat',
                      '5F-SubjectH-160804-5St-SGLHand-HFREQ.mat',
                      '5F-SubjectI-160719-5St-SGLHand-HFREQ.mat',
                      '5F-SubjectI-160723-5St-SGLHand-HFREQ.mat']  # 18

# choose the subject file from the file list
file_index = 10
subject_data_file = subject_data_files[file_index]

print(f"Loading eeg data from {subject_data_file}")
start_time_load_data = time.perf_counter()

subject_data_path = os.path.join(eeg_data_folder, subject_data_file)

# load eeg data from file
eeg_data_loader_instance = eeg_data_loader()
eeg_data = eeg_data_loader_instance.load_eeg_from_mat(subject_data_path)

# channel names
ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
            'A1', 'A2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz', 'X3']
# note that X3 is only for synchronization and A1, A2 are ground leads

# channels closest to the primary motor cortex
# F3, Fz, F4, C3, Cz, C4, P3, Pz, P4
# ch_picks = [2, 3, 4, 5, 6, 7, 18, 19, 20]
# pick almost all channels
# ch_picks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# ch_picks = [1, 2, 3, 4, 5, 6, 7, 9, 13, 15, 17, 18, 19, 20]
ch_picks = [0, 2, 3, 4, 5, 6, 7, 8, 12, 14, 16, 18, 19, 20]
print("Channels:")
print([ch_names[i] for i in ch_picks])

num_channels = len(ch_picks)

num_classes = 5

# each item in this list is a dict containing start + stop index and event type
events = eeg_data_loader_instance.find_all_events()

sample_frequency = eeg_data_loader_instance.sample_frequency
num_samples = eeg_data_loader_instance.num_samples

###############################################################################

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
    # if trial.max() > 200 or trial.min() < -200:
    #   print("Skip trial with amplitude > 200.")
    #   continue
    
    X.append(trial)
    
    # event type (1-5)
    y.append(event['event'])
  
  return X, y


downsample_step = 1
if sample_frequency == 1000:
  downsample_step = 5
elif sample_frequency == 200:
  downsample_step = 1
else:
  raise RuntimeError("Unexpected target frequency:", sample_frequency)

trials, labels = get_trials_x_and_y(eeg_data, events, sample_frequency,
                                    downsample_step=downsample_step, ch_picks=ch_picks)

print("trial shape:", type(trials[0]), trials[0].shape)  # trials is a simple list
# print("y:", type(y), y.shape)

end_time_load_data = time.perf_counter()
print(f"Time to load EEG-data: {end_time_load_data-start_time_load_data:.2f}s")

###############################################################################

# shuffle both trials and labels in unison
def parallel_shuffle(a1, a2):
  assert len(a1) == len(a2)
  permutation = np.random.permutation(len(a1))
  return [a1[i] for i in permutation], [a2[i] for i in permutation]


trials, labels = parallel_shuffle(trials, labels)

###############################################################################

X = np.array(trials)
y = np.array(labels)

# answer from git issue for braindecode:
# class labels should start at 0, thus range from 0-4, not 1-5
y -= 1

# safety check valid values for all labels
assert np.min(y) >= 0
assert np.max(y) <= 4


# change labels to classify one-versus-rest
# class to classify = 0
# all other classes = 1
# def map_label(label):
#   if label == 4:
#     return 0
#   return 1

# print("Map class labels for one-against-rest ...")
# print("One: class 4 (pinky) -> is class 0")
# print("Rest: class 0,1,2,3 -> is class 1")
# y = [map_label(v) for v in y_raw]
# y = np.array(y_raw)


# count the number of trials per class
# for i in [0,1,2,3,4]:
  # print("Trials per class:", i, "=", (y_raw==i).sum())


# pick trials and labels to do a 1 vs 1 classification
# only pick trials for two classes, discard the rest
"""
num_classes = 2
X = list()
y = list()
# compare 0-1, 0-2, 0-3, 0-4, 1-2, 1-3, 1-4, 2-3, 2-4, 3-4
first_class = 3   # change here
second_class = 4  # change here
print(f"Compare class {first_class} vs {second_class}")
print("Trials per class:", first_class, "=", (y_raw==first_class).sum())
print("Trials per class:", second_class, "=", (y_raw==second_class).sum())
# map first_class to class 0 and second_class to class 1
for trial, label in zip(X_raw, y_raw):
  if label == first_class:
    X.append(trial)
    y.append(0)
  elif label == second_class:
    X.append(trial)
    y.append(1)

X = np.array(X)
y = np.array(y)

del X_raw, y_raw
"""

# exit()

###############################################################################

# plot raw eeg data
if len(events) > 0 and False:
  first_event = events[0]
  # inspect a time fram from event onset (change in marker) for 1 second
  # (1 multiplied with the sample frequency)
  start = first_event['start']
  # stop = start_i + sample_frequency
  # add one because the range (end) is exclusive
  stop = first_event['stop'] + 1
  
  picks = ch_picks
  n_picks = len(picks)
  
  # stack multiple subplots in one dimension
  fig = plt.figure()
  gridspec = fig.add_gridspec(n_picks, hspace=0)
  axs = gridspec.subplots(sharex=True, sharey=True)
  
  for pick_index, ch_index in zip(range(n_picks), picks):
    # x = range(num_samples)
    x = range(start, stop)
    # y = [eeg_data[i][ch_index] for i in range(num_samples)]
    y = [eeg_data[i][ch_index] for i in range(start, stop)]
    # plt.plot(x,y)
    axs[pick_index].plot(x, y)
    # parameter ylim=(-8.0,8.0)
    axs[pick_index].set(ylabel=ch_names[ch_index])
  
  # hide x labels between subplots
  for ax in axs:
    ax.label_outer()

###############################################################################

# create the core mne data structure from scratch
# https://mne.tools/dev/auto_tutorials/simulation/10_array_objs.html#tut-creating-data-structures
if False:
  # by creating an info object ...
  # ch_types = ['eeg'] * len(ch_names)
  ch_types = 'eeg'
  info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sample_frequency)
  info.set_montage('standard_1020', on_missing='ignore')
  # print(info)
  
  # ... create mne data from an array of shape (n_channels, n_times)
  # the data read from .mat files must be transposed to match this shape
  transposed_eeg_data = eeg_data.transpose()
  # the eeg data is expected to be in V but is actually in microvolts here!
  # see np.min(transposed_eeg_data) and np.max(transposed_eeg_data)
  raw_data = mne.io.RawArray(transposed_eeg_data, info)
  raw_data.load_data()
  raw_data.filter(4., 40.)
  raw_data.pick(['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4'])
  
  start_time = events[0]['start']/sample_frequency
  
  onsets = list()
  descriptions = list()
  for ev in events:
    onsets.append(ev['start']/sample_frequency)
    descriptions.append(str(ev['event']))
  anno = mne.Annotations(onset=onsets, duration=1., description=descriptions)
  
  raw_data.set_annotations(anno)
  # data is in microvolts, not volts!
  
  scalings = dict(eeg=8)
  raw_data.plot(show_scrollbars=False, show_scalebars=False,
                duration=20, start=start_time-1, scalings = scalings)
  
  # note that n_components cannot be greater than number of channels
  ica = mne.preprocessing.ICA(n_components=6, random_state=42, max_iter=800)
  ica.fit(raw_data)
  ica.plot_sources(raw_data, show_scrollbars=False)
  
  ica.plot_components()
  # ica.plot_overlay(raw_data, exclude=[0,1])
  
  exit()
  
  raw_copy = raw_data.copy()
  # raw_copy.filter(4., 38.)
  raw_copy.pick(['F3', 'F4', 'C3', 'C4'])
  # raw_copy.pick(['C4'])
  
  # note: can access raw data like raw_copy[0] (channel-wise)
  # but the structure is weird
  
  # data is in microvolts, not volts!
  scalings = dict(eeg=8)
  raw_copy.plot(show_scrollbars=False, show_scalebars=False,
                duration=10, start=start_time-1, scalings = scalings)


# exit()

###############################################################################

# another test: use mne.create_from_mne_raw
# raw_data must be a list, seems to work
# IndexError: invalid index to scalar variable.
if False:
  windows_dataset = create_from_mne_raw([raw_data], 0, 0, 200, 200, False)
  model = ShallowFBCSPNet(
    num_channels,
    5, # number of classes
    input_window_samples=240,
    final_conv_length='auto',
    # final_conv_length=10,
    # pool_time_length=2,
    # pool_time_stride=2,
  )
  
  X_train, X_test, y_train, y_test = train_test_split(windows_dataset, y,
                                                      test_size=0.2, random_state=42)
  # if y is a numpy list of shape (len,) this raises another error
  # ValueError: X and y have inconsistent lengths.
  y_train = list(y_train)
  
  batch_size = 32
  n_epochs = 10
  clf = EEGClassifier(
    model,
    criterion=torch.nn.NLLLoss,
    optimizer=torch.optim.AdamW,
    # train_split=predefined_split(X_test),  # using valid_set for validation
    optimizer__lr=0.0625 * 0.01,
    optimizer__weight_decay=0,
    batch_size=batch_size,
    # classes=[1,2,3,4,5],
    callbacks=[
        "accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
    ],
    device='cpu',
  )
  clf.fit(X_train, y=y_train, epochs=n_epochs)

###############################################################################

mne.set_log_level(verbose='warning', return_old_level=True)
# old level was 20, now is 30

# print(X.shape)
# print(y.shape)

# split the dataset for k-fold cross-validation
k = 5
num_trials = len(X)
valid_size = int(num_trials / k)
print(f"Create {k} folds of (validation) size {valid_size}")
# note that the train set might be a bit larger than (k-1) * valid_size

# calculate the average metrics
all_acc_train = []
all_acc_valid = []

# sum over all confusion matrices
cumulative_cm = None

start_time_train = time.perf_counter()

for i in range(k):
  print("-"*80)
  print(f"Fold {i+1}/{k}:")
  
  print(f"Split: from {i*valid_size} to {(i+1)*valid_size}")
  train_X = np.delete(X, np.s_[i*valid_size:(i+1)*valid_size], axis=0)
  train_y = np.delete(y, np.s_[i*valid_size:(i+1)*valid_size], axis=0)
  valid_X = X[i*valid_size:(i+1)*valid_size]
  valid_y = y[i*valid_size:(i+1)*valid_size]
  
  # distribution of different train class labels within this fold
  label_count_train = [(train_y == i).sum() for i in range(num_classes)]
  print("Number of labels per class: (train set)")
  print(label_count_train)
  label_count_train = np.array(label_count_train)
  print(f"std: {label_count_train.std():.2f}")
  
  # distribution of different validation class labels within this fold
  label_count_valid = [(valid_y == i).sum() for i in range(num_classes)]
  print("Number of labels per class: (validation set)")
  print(label_count_valid)
  label_count_valid = np.array(label_count_valid)
  # print(f"mean: {label_count_valid.mean()}")
  print(f"std: {label_count_valid.std():.2f}")
  
  # continue
  
  # create individual train and valid sets
  train_set = create_from_X_y(train_X, train_y, drop_last_window=False, sfreq=sample_frequency)
  valid_set = create_from_X_y(valid_X, valid_y, drop_last_window=False, sfreq=sample_frequency)
  
  # print("Train dataset description:")
  # print(train_set.description)
  # print("Valid dataset description:")
  # print(valid_set.description)
  
  cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
  device = 'cuda' if cuda else 'cpu'
  if cuda:
    torch.backends.cudnn.benchmark = True
  
  # shape is (22,240) when using all channels
  input_window_samples = train_set[0][0].shape[1]
  
  model_type = "deep"
  print(f"Creating model ({model_type}) ...")
  
  model = None
  if model_type == "shallow":
    model = ShallowFBCSPNet(
        num_channels,
        num_classes,
        input_window_samples=sample_frequency * 2,
        final_conv_length='auto',
    )
  elif model_type == "deep":
    model = Deep4Net(
        num_channels,
        num_classes,
        input_window_samples=input_window_samples, #  sample_frequency * 2,
        final_conv_length=6,
        pool_time_length=2,
        pool_time_stride=2,
        
        # for 1000 Hz data
        # final_conv_length='auto',
        # (no pool_time_length and pool_time_stride)
    )
  
  # Send model to GPU (if possible)
  if cuda:
    model.cuda()
  
  print("Creating classifier...")
  learn_rate = 0.001  # 0.000625
  print(f"Learn rate: {learn_rate}")
  batch_size = 16
  n_epochs = 20
  clf = EEGClassifier(
      model,
      criterion=torch.nn.NLLLoss,
      optimizer=torch.optim.AdamW,
      train_split=predefined_split(valid_set),  # using valid_set for validation
      optimizer__lr=learn_rate,
      optimizer__weight_decay=0,
      batch_size=batch_size,
      callbacks=[
          "accuracy",
          ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
      ],
      device=device,
  )
  print(f"Training for {n_epochs} epochs...")
  clf.fit(train_set, y=None, epochs=n_epochs)
  
###############################################################################

# if clf is not None:
  # plot the results
  # Extract loss and accuracy values for plotting from history object
  results_columns = ['train_loss', 'valid_loss', 'train_accuracy', 'valid_accuracy']
  df = pd.DataFrame(clf.history[:, results_columns], columns=results_columns,
                    index=clf.history[:, 'epoch'])

  # get percent of misclass for better visual comparison to loss
  df = df.assign(train_misclass=100 - 100 * df.train_accuracy,
                  valid_misclass=100 - 100 * df.valid_accuracy)
  # df = df.assign(train_accuracy=100 * df.train_accuracy,
                 # valid_accuracy=100 * df.valid_accuracy)

  plt.style.use('seaborn')
  fig, ax1 = plt.subplots(figsize=(8, 3))
  df.loc[:, ['train_loss', 'valid_loss']].plot(
      ax=ax1, style=['-', ':'], marker='o', color='tab:blue', legend=False, fontsize=14)

  ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=14)
  ax1.set_ylabel("Loss", color='tab:blue', fontsize=14)

  ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

  df.loc[:, ['train_misclass', 'valid_misclass']].plot(
  # df.loc[:, ['train_accuracy', 'valid_accuracy']].plot(
      ax=ax2, style=['-', ':'], marker='o', color='tab:red', legend=False)
  ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=14)
  ax2.set_ylabel("Misclassification Rate [%]", color='tab:red', fontsize=14)
  # ax2.set_ylabel("Accuracy [%]", color='tab:red', fontsize=14)
  ax2.set_ylim(ax2.get_ylim()[0], 85)  # make some room for legend
  ax1.set_xlabel("Epoch", fontsize=14)

  # where some data has already been plotted to ax
  handles = []
  handles.append(Line2D([0], [0], color='black', linewidth=1, linestyle='-', label='Train'))
  handles.append(Line2D([0], [0], color='black', linewidth=1, linestyle=':', label='Valid'))
  plt.legend(handles, [h.get_label() for h in handles], fontsize=14)
  plt.tight_layout()
  plt.show()
  
  # get the highest training and validation accuracies and store them
  # to compute an average at the end
  acc_train = df['train_accuracy'].max()
  acc_valid = df['valid_accuracy'].max()
  
  all_acc_train.append(acc_train)
  all_acc_valid.append(acc_valid)
  
  
  # predict on some data...
  # can use valid_set or valid_X
  predicted_labels = clf.predict(valid_set)
  
  # calculate the confusion matrix and some metrics
  cm = get_confusion_matrix(valid_y, predicted_labels)
  plot_confusion_matrix(cm, "Konfusionsmatrix, Fold "+str(i+1))
  if cumulative_cm is None:
    cumulative_cm = cm
  else:
    cumulative_cm = cumulative_cm + cm
  
  precision, recall, f_score = calculate_cm_scores(cm)
  print("Precision:", precision, "Mean:", np.array(precision).mean())
  print("Recall:", recall, "Mean:", np.array(recall).mean())
  print("F1 Score:", f_score, "Mean:", np.array(f_score).mean())
  
  # get (probabilities) for each class
  # actually this returns the output of the forward method
  # with all(?) values being negative
  # scores = clf.predict_proba(valid_X)
  
  # to be used for cropped mode:
  # predictions, labels = clf.predict_trials(valid_set, return_targets=True)
  
  # for testing: stop after the Nth fold
  # if i >= 1:
  # break

end_time_train = time.perf_counter()
print(f"Time to train models: {end_time_train-start_time_train:.2f}s")

# get the mean accuracy and standard deviation from all folds
all_acc_train = np.array(all_acc_train)
all_acc_valid = np.array(all_acc_valid)
acc_mean_train = all_acc_train.mean()
acc_std_train = all_acc_train.std()
acc_mean_valid = all_acc_valid.mean()
acc_std_valid = all_acc_valid.std()
print(f"Mean accuracy (train) is {acc_mean_train:.3f} and STD is {acc_std_train:.3f}")
print(f"Mean accuracy (valid) is {acc_mean_valid:.3f} and STD is {acc_std_valid:.3f}")

print("Cumulative confusion matrix:")
print(cumulative_cm)
plot_confusion_matrix(cumulative_cm, "Konfusionsmatrix, aufsummiert")

# precision and recall per class
precision, recall, f_score = calculate_cm_scores(cumulative_cm)
print("Mean precision per class:")
for i, p in enumerate(precision):
  print(f"{i}: {p:.2f}")
print("Mean recall per class:")
for i, r in enumerate(recall):
  print(f"{i}: {r:.2f}")
print("Mean F1 score per class:")
for i, f1 in enumerate(f_score):
  print(f"{i}: {f1:.2f}")

