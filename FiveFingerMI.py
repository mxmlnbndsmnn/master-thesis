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


# def main():
eeg_data_folder = "A large MI EEG dataset for EEG BCI"

# subject_data_file = "5F-SubjectB-151110-5St-SGLHand.mat"
subject_data_file = "5F-SubjectC-151204-5St-SGLHand.mat"

file_path = os.path.join(eeg_data_folder, subject_data_file)

# the matlab structure is named 'o'
mat_data = loadmat(file_path)['o']
# print(mat_data)

# number of samples
num_samples = mat_data['nS'][0][0][0][0]

# sample frequency is also the same for the entire dataset
sample_frequency = mat_data['sampFreq'][0][0][0][0]

# key fields are: id, nS (num of EEG signal sampes), sampFreq, marker and data
# mat_data[0][0]["data"].shape as well as mat_data['data'][0][0].shape
# outputs (724600, 22)
# -> num samples is 724600 and 22 measured voltage time-series from
# 19 electrodes + 2 ground leads (A1, A2) + 1 synch channel (X3)
# marker codes: 1...5 for thumb ... pinky

# store the marker list in a file
marker_ugly = mat_data['marker'].item()
# to access each entry in the marker file one must use marker[i][0]
# so let's convert it to a flat list
# marker = [marker_ugly[i][0] for i in range(num_samples)]
# alternative method using transpose
marker = marker_ugly.transpose()[0]
# savetxt(os.path.join('csvs','marker.csv'), marker, fmt='%d', delimiter=',')

# store the eeg data in a file
# shape is (724600, 22)
eeg_data = mat_data['data'].item()
# savetxt(os.path.join('csvs','eeg_data.csv'), eeg_data, delimiter=',')

# channel names
ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
            'A1', 'A2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz', 'X3']
# note that X3 is only for synchronization and A1, A2 are ground leads

# channels closest to the primary motor cortex
# F3, Fz, F4, C3, Cz, C4, P3, Pz, P4
ch_picks = [2, 3, 4, 5, 6, 7, 18, 19, 20]
# print([ch_names[i] for i in ch_picks])


# find all events of a marker switching from 0 to >0
def find_all_events(marker) -> list:

    hits = list()
    index = 0
    while True:
        start_i = find_next_event_start(marker, index)
        if start_i < 0:
            break

        stop_i = find_next_event_stop(marker, start_i+1)
        if stop_i < 0:
            break
        hit = dict()
        hit['start'] = start_i
        hit['stop'] = stop_i
        hit['event'] = marker[start_i]
        hits.append(hit)
        # print("Found event:")
        # print(hit)

        # continue from last event end
        index = stop_i+1

    return hits


def find_next_event_stop(marker, start=0):
    # careful: the start parameter in enumerate does NOT work like the start value in range
    # it only acts as an offset for the index, but always starts with the first item!
    # for i, m in enumerate(marker, start):
    for i in range(start, len(marker)):
        if marker[i] == 0:
            return i-1
    return -1


def find_next_event_start(marker, start=0):
    for i in range(start, len(marker)):
        if marker[i] > 0:
            return i
    return -1


# each item in this list is a dict containing start + stop index and event type
events = find_all_events(marker)


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
    scalings = dict(eeg=4)
    raw_copy.plot(show_scrollbars=False, show_scalebars=False,
                  duration=6, start=start_time-1, scalings = scalings)


# cut trials from the full eeg data
if True:
    # reshape eeg data -> n_channels x n_times
    transposed_eeg_data = eeg_data.transpose()
    trials = list()
    y = list()
    # input_window_samples = 200
    for event in events:
        start_i = event['start']
        # stop_i = event['stop']
        stop_i = start_i + 200
        trial = np.array([[ch[i] for i in range(start_i, stop_i)] for ch in transposed_eeg_data])
        
        trials.append(trial)
        
        # event type (1-5)
        y.append(event['event'])
    
    # X must be: list of pre-cut trials as n_trials x n_channels x n_times
    X = np.array(trials)
    # y must be: targets corresponding to the trials
    y = np.array(y)

# can get min and max values in an array by np.amin and np.amax

# TODO: band pass filter - how to?
# use multiple channels and concatenate their stft result into one image per trial


# for a single trial; calculate the stft and create an image
# save it to a folder - one folder per event type (1-5)
def create_stft_image_for_trial(trial, freq, path, picks=None, nperseg=50,
                                axis=0, file_name="stft.png"):
  channels = None
  if picks is not None:
    channels = [trial[ch_index] for ch_index in picks]
  else:
    channels = trial
  
  image_data = None
  for ch in channels:
    f,t,Zxx = signal.stft(ch, fs=freq, nperseg=nperseg)
    absolutes = np.abs(Zxx)
    if image_data is not None:
      image_data = np.append(image_data, absolutes, axis=axis)
    else:
      image_data = absolutes
  
  # print(image_data.shape)
  img_path = os.path.join(path, file_name)
  plt.imsave(img_path, image_data, vmin=0, vmax=2)


# working method to create stft images for one subject
if True:
  stft_folder = "stft_images"
  subject_folder = "SubjectC-151204-9ch" # only pick some channels
  # save with prefix to allow to throw multiple groups of images together later
  file_prefix = "9ch"
  n_trials = 10
  trial_index = 1
  # for now, pick all channels except the last two
  # stft_ch_picks = list(range(20))
  stft_ch_picks = ch_picks
  for trial, event in zip(trials, events):
    f_name = f"{file_prefix}_{trial_index:05}.png"
    event_type_string = str(event['event'])
    path = os.path.join(stft_folder, subject_folder, event_type_string)
    # ensure that a folder per event type exists
    # also create parent (subject) folder if needed
    # ignore if the folder already exists
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    create_stft_image_for_trial(trial, sample_frequency, axis=1, path=path,
                                picks=stft_ch_picks, file_name=f_name)
    trial_index += 1
   
    # if trial_index > n_trials:
      # break


# does not work, output images are invalid
# values in raw_data are totally different from those in trials obtained from
# transposed_eeg_data (above)
if False:
  stft_folder = "stft_images"
  subject_folder = "SubjectC-151204-test2"
  trial_index = 1
  n_trials = 10
  # for now, pick all channels except the last two
  stft_ch_picks = list(range(20))
  for event in events:
    trial = [raw_data[ch][1] for ch in range(20)]
    f_name = f"{trial_index:05}.png"
    event_type_string = str(event['event'])
    path = os.path.join(stft_folder, subject_folder, event_type_string)
    # ensure that a folder per event type exists
    # also create parent (subject) folder if needed
    # ignore if the folder already exists
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    create_stft_image_for_trial(trial, sample_frequency, axis=1, path=path,
                                picks=stft_ch_picks, file_name=f_name)
    trial_index += 1
    if trial_index > n_trials:
      break


if False:
  trial = X[0]
  print(trial.shape) # (22, 200)
  # pick data from one channel -> shape (200,)
  ch=trial[5]
  
  # segment length is 256 by default, but input length is only 200 here
  nperseg = 50
  f,t,Zxx = signal.stft(ch, fs=sample_frequency, nperseg=nperseg)
  # for a sampling frequency of 200 this yields:
  # f - array of sample frequencies -> shape (33,)
  # t - array of segment times -> shape (8,)
  # Zxx - STFT of x -> shape (33, 8)
  # shading should be either nearest or gouraud
  # or flat when making the color map (Zxx) one smaller than t and f:
  cutZxx = Zxx[:-1,:-1]
  plt.pcolormesh(t, f, np.abs(cutZxx), vmin=0, vmax=2, shading='flat')
  plt.title(f'STFT Magnitude (segment length: {nperseg})')
  plt.ylabel('Frequency [Hz]')
  plt.xlabel('Time [sec]')
  # plt.show() # ? not needed; do not use for savefig to work!
  
  # why does it look like more than 1 second? because of the segment length?
  # yep, apparently since it fits when using dividers of 200 like 50
  
  # save the result to an image file
  # plt.savefig("stft.png") # with axis and labels
  # plt.imsave("test.png", np.abs(cutZxx), vmin=0, vmax=2) # data only
  
  # throw together multiple "images"
  absolutes = np.abs(Zxx)
  # axis must be 0 or 1 to keep the 2D shape
  image_data = np.append(absolutes, absolutes, axis=0)
  print(image_data.shape)
  plt.imsave("foobar.png", image_data, vmin=0, vmax=2)


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