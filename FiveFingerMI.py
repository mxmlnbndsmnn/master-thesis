# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 11:34:11 2021

@author: User
"""

import os
import pathlib
import mne
from scipy.io import loadmat
import scipy.signal as signal
import numpy as np
from numpy import savetxt, loadtxt
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from braindecode.datasets import create_from_X_y, create_from_mne_raw # used to be in .datautil
from braindecode.models import ShallowFBCSPNet, Deep4Net
from braindecode import EEGClassifier
import torch
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split
from sklearn.model_selection import train_test_split
import pandas as pd

from eeg_data_loader import eeg_data_loader
from create_stft_image_for_trial import create_stft_image_for_trial, plot_trial_stft


eeg_data_folder = "A large MI EEG dataset for EEG BCI"

# subject_data_file = "5F-SubjectB-151110-5St-SGLHand.mat"
# subject_data_file = "5F-SubjectC-151204-5St-SGLHand.mat"
# subject_data_file = "5F-SubjectF-151027-5St-SGLHand.mat"
subject_data_file = "5F-SubjectF-160210-5St-SGLHand-HFREQ.mat"

# place where the STFT images should be stored
# parent folder
stft_folder = "stft_images"

# per subject folder
subject_image_folder = "SubjectF-160210-9ch-HFREQ_2"

# EEG data source
subject_data_path = os.path.join(eeg_data_folder, subject_data_file)

eeg_data_loader_instance = eeg_data_loader()
eeg_data = eeg_data_loader_instance.load_eeg_from_mat(subject_data_path)

# channel names
ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
            'A1', 'A2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz', 'X3']
# note that X3 is only for synchronization and A1, A2 are ground leads

# channels closest to the primary motor cortex
# F3, Fz, F4, C3, Cz, C4, P3, Pz, P4
ch_picks = [2, 3, 4, 5, 6, 7, 18, 19, 20]
# print([ch_names[i] for i in ch_picks])


# each item in this list is a dict containing start + stop index and event type
events = eeg_data_loader_instance.find_all_events()

sample_frequency = eeg_data_loader_instance.sample_frequency
num_samples = eeg_data_loader_instance.num_samples


# plot raw eeg data
if len(events) > 0 and False:
    first_event = events[0]
    # inspect a time fram from event onset (change in marker) for 1 second
    # (1 multiplied with the sample frequency)
    start = first_event['start']
    # stop = start_i + sample_frequency
    # add one because the range (end) is exclusive
    stop = first_event['stop'] + 1
    
    # display some raw data
    picks = [1, 2, 3, 4, 5]
    
    # stack multiple subplots in one dimension
    fig = plt.figure()
    gridspec = fig.add_gridspec(len(picks), hspace=0)
    axs = gridspec.subplots(sharex=True, sharey=True)
    
    for ch_index in picks:
        # x = range(num_samples)
        x = range(start, stop)
        # y = [eeg_data[i][ch_index] for i in range(num_samples)]
        y = [eeg_data[i][ch_index] for i in range(start, stop)]
        # plt.plot(x,y)
        axs[ch_index-1].plot(x, y)
        # parameter ylim=(-8.0,8.0)
        axs[ch_index-1].set(ylabel=ch_names[ch_index])
    
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
    
    # this results in errors related to missing data in info?
    # e.g. ValueError: No Source for sfreq found
    # or KeyError: 'time'
    # raw_data = mne.io.read_raw_fieldtrip(file_path, info, data_name="o")
    # print(raw_data.info)
    
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


trials, labels = eeg_data_loader_instance.get_trials_x_and_y()


# X must be: list of pre-cut trials as n_trials x n_channels x n_times
X = np.array(trials)
# y must be: targets corresponding to the trials
y = np.array(labels)
# note: can get min and max values in an array by np.amin and np.amax


# working method to create stft images for one subject
# use multiple channels and concatenate their stft result into one image per trial
if True:
  # save with prefix to allow to throw multiple groups of images together later
  file_prefix = "9ch"
  # n_trials = 10
  trial_index = 1
  stft_ch_picks = ch_picks
  
  for trial, event in zip(trials, events):
    
    event_type_string = str(event['event'])
    path = os.path.join(stft_folder, subject_image_folder, event_type_string)
    # print(trial)
    # print(trial.shape)
    
    # keep the original trial + create more trials by cutting off the extra
    # frames at the beginning and/or end of each trial
    forerun_frames = int(sample_frequency * 0.2)
    affix_frames = int(sample_frequency * 0.2)
    frs = [None, forerun_frames, None, forerun_frames]
    afs = [None, None, -affix_frames, -affix_frames]
    augmentation_index = 1
    for fr, af in zip(frs, afs):
      tr = trial[:, fr:af]
      # print(tr.shape)
      
      f_name = f"{file_prefix}_{trial_index:05}_{augmentation_index:02}.png"
      
      # ensure that a folder per event type exists
      # also create parent (subject) folder if needed
      # ignore if the folder already exists
      pathlib.Path(path).mkdir(parents=True, exist_ok=True)
      create_stft_image_for_trial(tr, sample_frequency, axis=0, path=path,
                                  picks=stft_ch_picks, file_name=f_name)
      augmentation_index += 1
    
    trial_index += 1
    
    # if trial_index > n_trials:
      # break


# plot a single trial stft for one channel
ch=trial[ch_picks[0]]
plot_trial_stft(ch, nperseg=40, sample_frequency=sample_frequency,
                file_name="foobar40.png")
plot_trial_stft(ch, nperseg=100,
                sample_frequency=sample_frequency,
                file_name="foobar100.png")


# another test: use mne.create_from_mne_raw
# ValueError: All picks must be < n_channels (22), got 22
if False:
  windows_dataset = create_from_mne_raw(raw_data, 0, 0, 500, 500, False)
  model = ShallowFBCSPNet(
      len(ch_names), # number of channels
      5, # number of classes
      input_window_samples=len(windows_dataset),
      final_conv_length='auto',
      # final_conv_length=10,
      # pool_time_length=2,
      # pool_time_stride=2,
  )
  
  X_train, X_test, y_train, y_test = train_test_split(windows_dataset, y,
                                                      test_size=0.2, random_state=42)
  
  batch_size = 64
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


if False:
    # mne.set_log_level(verbose='warning',return_old_level=True)
    # old level was 20, now is 30
    
    # split into train and valid set
    num_time_points = len(X) #windows_dataset
    split_i = int(num_time_points * 0.8)
    
    train_X = X[:split_i]
    valid_X = X[split_i:]
    train_y = y[:split_i]
    valid_y = y[split_i:]
    
    # TODO standardize per channel?
    
    # create individual train and valid sets
    train_set = create_from_X_y(train_X, train_y, drop_last_window=False, sfreq=sample_frequency)
    valid_set = create_from_X_y(valid_X, valid_y, drop_last_window=False, sfreq=sample_frequency)
    print("train dataset description:")
    print(train_set.description)
    print("valid dataset description:")
    print(valid_set.description)
    
    cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
    device = 'cuda' if cuda else 'cpu'
    if cuda:
        torch.backends.cudnn.benchmark = True
    
    print("creating model...")
    input_window_samples = len(train_set) # ?
    model = ShallowFBCSPNet(
    # model = Deep4Net(
        len(ch_names), # number of channels
        5, # number of classes
        input_window_samples=input_window_samples,
        final_conv_length='auto',
        # final_conv_length=10,
        # pool_time_length=2,
        # pool_time_stride=2,
    )
    
    # Send model to GPU (if possible)
    if cuda:
        model.cuda()
    
    print("creating classifier...")
    batch_size = 64
    n_epochs = 10
    clf = EEGClassifier(
        model,
        criterion=torch.nn.NLLLoss,
        optimizer=torch.optim.AdamW,
        train_split=predefined_split(valid_set),  # using valid_set for validation
        optimizer__lr=0.0625 * 0.01,
        optimizer__weight_decay=0,
        batch_size=batch_size,
        # classes=[1,2,3,4,5],
        callbacks=[
            "accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
        ],
        device=device,
    )
    clf.fit(train_set, y=None, epochs=n_epochs)
    
    # using ShallowFBCSPNet model
    # -> Error in 439 _conv_forward
    # RuntimeError: Calculated padded input size per channel: (7 x 1). Kernel
    # size: (21 x 1). Kernel size can't be greater than actual input size
    
    # changed model to Deep4Net
    # -> Error in 439 _conf_forward
    # RuntimeError: Calculated padded input size per channel: (3 x 1). Kernel
    # size: (10 x 1). Kernel size can't be greater than actual input size
    # or with auto final_conv_length Error in 718 _max_pool2d
    # RuntimeError: Given input size: (200x1x1). Calculated output size:
    # (200x0x1). Output size is too small
    
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


# if __name__ == '__main__':
    # main()


'''
? git issue ?
Error: Target is out of bounds when trying to clf.fit
Hello there! I followed the steps described in the [trialwise decoding example](https://braindecode.org/auto_examples/plot_bcic_iv_2a_moabb_trial.html) and tried to adapt the process to work with a different dataset. Namely the 5 finger MI from [this dataset](https://figshare.com/collections/A_large_electroencephalographic_motor_imagery_dataset_for_electroencephalographic_brain_computer_interfaces/3917698?q=5F).
'''