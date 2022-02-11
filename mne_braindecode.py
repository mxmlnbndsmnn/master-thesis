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
from braindecode.datasets import create_from_X_y, create_from_mne_raw # used to be in .datautil
from braindecode.models import ShallowFBCSPNet, Deep4Net
from braindecode import EEGClassifier

import torch
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split
from sklearn.model_selection import train_test_split

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from eeg_data_loader import eeg_data_loader


# EEG data source
eeg_data_folder = "A large MI EEG dataset for EEG BCI"

# subject_data_file = "5F-SubjectB-151110-5St-SGLHand.mat"
subject_data_file = "5F-SubjectB-160316-5St-SGLHand.mat"

# subject_data_file = "5F-SubjectC-151204-5St-SGLHand.mat"

# subject_data_file = "5F-SubjectF-151027-5St-SGLHand.mat"
# subject_data_file = "5F-SubjectF-160210-5St-SGLHand-HFREQ.mat"

print(f"Loading eeg data from {subject_data_file}")

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
ch_picks = [2, 3, 4, 5, 6, 7, 18, 19, 20]
print("Channels:")
print([ch_names[i] for i in ch_picks])

num_channels = len(ch_picks)

# each item in this list is a dict containing start + stop index and event type
events = eeg_data_loader_instance.find_all_events()

sample_frequency = eeg_data_loader_instance.sample_frequency
num_samples = eeg_data_loader_instance.num_samples


trials, labels = eeg_data_loader_instance.get_trials_x_and_y()
X = np.array(trials)
y = np.array(labels)

# only pick some channels
X = np.take(X, ch_picks, axis=1)
# print(X.shape)
# shape (num_trials, num_channels, num_time_points)

# answer from git issue for braindecode:
# class labels should start at 0, thus range from 0-4, not 1-5
y -= 1

# safety check valid values for all labels
assert np.min(y) >= 0
assert np.max(y) <= 4

# exit()


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


# create the core mne data structure from scratch
# https://mne.tools/dev/auto_tutorials/simulation/10_array_objs.html#tut-creating-data-structures
if False:
  # by creating an info object ...
  # ch_types = ['eeg'] * len(ch_names)
  ch_types = 'eeg'
  info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sample_frequency)
  # info.set_montage('standard_1020')
  # print(info)
  
  # ... create mne data from an array of shape (n_channels, n_times)
  # the data read from .mat files must be transposed to match this shape
  transposed_eeg_data = eeg_data.transpose()
  # the eeg data is expected to be in V but is actually in microvolts here!
  # see np.min(transposed_eeg_data) and np.max(transposed_eeg_data)
  raw_data = mne.io.RawArray(transposed_eeg_data, info)
  raw_data.load_data()
  raw_data.filter(4., 38.)
  
  start_time = events[0]['start']/sample_frequency
  
  onsets = list()
  descriptions = list()
  for ev in events:
    onsets.append(ev['start']/sample_frequency)
    descriptions.append(str(ev['event']))
  anno = mne.Annotations(onset=onsets, duration=1., description=descriptions)
  
  raw_data.set_annotations(anno)
  
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


if True:
  mne.set_log_level(verbose='warning', return_old_level=True)
  # old level was 20, now is 30
  
  # split into train and valid set
  num_trials = len(X)
  split_i = int(num_trials * 0.8)
  
  train_X = X[:split_i]
  valid_X = X[split_i:]
  train_y = y[:split_i]
  valid_y = y[split_i:]
  
  # TODO standardize per channel?
  
  # create individual train and valid sets
  train_set = create_from_X_y(train_X, train_y, drop_last_window=False, sfreq=sample_frequency)
  valid_set = create_from_X_y(valid_X, valid_y, drop_last_window=False, sfreq=sample_frequency)
  print("Train dataset description:")
  print(train_set.description)
  print("Valid dataset description:")
  print(valid_set.description)
  
  cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
  device = 'cuda' if cuda else 'cpu'
  if cuda:
      torch.backends.cudnn.benchmark = True
  
  input_window_samples = train_set[0][0].shape[1] # shape is (22,240)
  
  model_type = "deep"
  print(f"Creating model ({model_type}) ...")
  
  model = None
  if model_type == "shallow":
    model = ShallowFBCSPNet(
        num_channels,
        5, # number of classes
        input_window_samples=sample_frequency * 2,
        final_conv_length='auto',
    )
  elif model_type == "deep":
    model = Deep4Net(
        num_channels,
        5, # number of classes
        input_window_samples=sample_frequency * 2,
        final_conv_length=6,
        pool_time_length=2,
        pool_time_stride=2,
    )
  
  # Send model to GPU (if possible)
  if cuda:
    model.cuda()
  
  print("Creating classifier...")
  learn_rate = 0.0625 * 0.01
  print(f"Learn rate: {learn_rate}")
  batch_size = 32
  n_epochs = 16
  clf = EEGClassifier(
      model,
      criterion=torch.nn.NLLLoss,
      optimizer=torch.optim.AdamW,
      train_split=predefined_split(valid_set),  # using valid_set for validation
      optimizer__lr=learn_rate,
      optimizer__weight_decay=0,
      batch_size=batch_size,
      callbacks=[
          "accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
      ],
      device=device,
  )
  print(f"Training for {n_epochs} epochs...")
  clf.fit(train_set, y=None, epochs=n_epochs)
  

if clf is not None:
  # plot the results
  # Extract loss and accuracy values for plotting from history object
  results_columns = ['train_loss', 'valid_loss', 'train_accuracy', 'valid_accuracy']
  df = pd.DataFrame(clf.history[:, results_columns], columns=results_columns,
                    index=clf.history[:, 'epoch'])

  # get percent of misclass for better visual comparison to loss
  df = df.assign(train_misclass=100 - 100 * df.train_accuracy,
                 valid_misclass=100 - 100 * df.valid_accuracy)

  plt.style.use('seaborn')
  fig, ax1 = plt.subplots(figsize=(8, 3))
  df.loc[:, ['train_loss', 'valid_loss']].plot(
      ax=ax1, style=['-', ':'], marker='o', color='tab:blue', legend=False, fontsize=14)

  ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=14)
  ax1.set_ylabel("Loss", color='tab:blue', fontsize=14)

  ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

  df.loc[:, ['train_misclass', 'valid_misclass']].plot(
      ax=ax2, style=['-', ':'], marker='o', color='tab:red', legend=False)
  ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=14)
  ax2.set_ylabel("Misclassification Rate [%]", color='tab:red', fontsize=14)
  ax2.set_ylim(ax2.get_ylim()[0], 85)  # make some room for legend
  ax1.set_xlabel("Epoch", fontsize=14)

  # where some data has already been plotted to ax
  handles = []
  handles.append(Line2D([0], [0], color='black', linewidth=1, linestyle='-', label='Train'))
  handles.append(Line2D([0], [0], color='black', linewidth=1, linestyle=':', label='Valid'))
  plt.legend(handles, [h.get_label() for h in handles], fontsize=14)
  plt.tight_layout()


