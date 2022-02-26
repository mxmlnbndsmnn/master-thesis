# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 11:34:11 2021

@author: User

This script is (was) used to generate and store STFT images for all trials
for one subject eeg data .mat file.
"""

import os
import pathlib

from scipy.io import loadmat
import scipy.signal as signal
import numpy as np
from numpy import savetxt, loadtxt
import matplotlib.pyplot as plt

from eeg_data_loader import eeg_data_loader
from create_stft_image import create_stft_image_for_trial, plot_trial_stft


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


# if __name__ == '__main__':
    # main()

