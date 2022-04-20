# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 11:34:11 2021

@author: User

This script is (was) used to generate and store STFT images for all trials
for one subject eeg data .mat file.
"""

import os
import pathlib
import time

from scipy.io import loadmat
import scipy.signal as signal
from scipy.stats import kurtosis
import numpy as np
from numpy import savetxt, loadtxt
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA, PCA

from eeg_data_loader import eeg_data_loader
from create_eeg_image import create_stft_image_for_trial, plot_trial_stft, create_ctw_for_channel

from sys import exit


eeg_data_folder = "A large MI EEG dataset for EEG BCI"

subject_data_file = "5F-SubjectA-160405-5St-SGLHand.mat"
# subject_data_file = "5F-SubjectB-151110-5St-SGLHand.mat"
# subject_data_file = "5F-SubjectC-151204-5St-SGLHand.mat"
# subject_data_file = "5F-SubjectF-151027-5St-SGLHand.mat"
# subject_data_file = "5F-SubjectF-160210-5St-SGLHand-HFREQ.mat"

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
# ch_picks = [2, 3, 4, 5, 6, 7, 18, 19, 20]

# pick all channels except reference and Fp1, Fp2
ch_picks = [2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# print([ch_names[i] for i in ch_picks])

print(eeg_data.shape)
eeg_data = np.array([eeg_data.T[i] for i in ch_picks]).T
print(eeg_data.shape)

# each item in this list is a dict containing start + stop index and event type
events = eeg_data_loader_instance.find_all_events()

sample_frequency = eeg_data_loader_instance.sample_frequency
num_samples = eeg_data_loader_instance.num_samples

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

eeg_data = butter_bandpass_filter(eeg_data, 4.0, 40.0, sample_frequency, order=6, axis=1)

###############################################################################

varis = []
kurts = []
for event in events:
  # event = events[734]
  start_frame = event["start"]
  end_frame = event["stop"]
  
  # compute ICA
  X = eeg_data[start_frame:end_frame,:]
  # X = eeg_data[start_frame:end_frame,:-1]  # remove the last channel (X3)
  # not needed when picking channels
  ica = FastICA(n_components=10, random_state=42)
  S_ = ica.fit_transform(X)  # Get the estimated sources
  A_ = ica.mixing_  # Get estimated mixing matrix
  
  # plot raw data individually
  # plt.figure(figsize=(10, 10))
  # for i in range(len(ch_picks)):
  #   plt.subplot(5,5,i+1)
  #   plt.title(ch_names[i])
  #   plt.plot(X.T[i])
  
  # compute PCA
  # pca = PCA(n_components=5)
  # H = pca.fit_transform(X)  # estimate PCA sources
  
  sources = S_.T
  
  for s in sources:
    # print(f"Variance: {s.var():.4f}")
    # print(f"Kurtosis: {kurtosis(s):.2f}")
    varis.append(s.var())
    kurts.append(kurtosis(s))

varis = np.array(varis)
kurts = np.array(kurts)

exit()

plt.figure(figsize=(10,2))
for i in range(5):
  plt.subplot(1,5,i+1)
  plt.plot(sources[i])


# sources[3][:] = 0  # remove components manually
sources[4][:] = 0

restored = ica.inverse_transform(sources.T)

plt.figure(figsize=(9, 6))

models = [X, S_, restored]
names = ['Observations (mixed signal)',
         # 'True Sources',
         'ICA estimated sources',
         # 'PCA estimated sources',
         "ICA-restored signal"]
colors = ['red', 'steelblue', 'orange', "green", "blue"] * 5

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(len(models), 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)

plt.tight_layout()

exit()

###############################################################################

"""
# plot raw eeg data
if len(events) > 0:
    first_event = events[0]
    # inspect a time fram from event onset (change in marker) for 1 second
    # (1 multiplied with the sample frequency)
    start = first_event['start']
    # stop = start_i + sample_frequency
    # add one because the range (end) is exclusive
    stop = first_event['stop'] + 1
    
    # display some raw data
    picks = ch_picks  # [1, 2, 3, 4, 5]
    
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
"""


"""
start_1 = time.perf_counter()
eeg_data1 = butter_bandpass_filter(eeg_data, 4.0, 40.0, sample_frequency, order=3, axis=1)
stop_1 = time.perf_counter()
print(f"Time to 3rd order filter data: {stop_1-start_1:.2f}s")

start_2 = time.perf_counter()
eeg_data2 = butter_bandpass_filter(eeg_data, 4.0, 40.0, sample_frequency, order=6, axis=1)
stop_2 = time.perf_counter()
print(f"Time to 6th order filter data: {stop_2-start_2:.2f}s")

exit()
"""



# plot raw eeg data
if len(events) > 0:
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
  fig = plt.figure(figsize=(20,15))
  # plt.title("Raw signal")
  plt.title("EEG-Signal, kein Filter")
  # plt.xticks([])
  plt.yticks([])  # remove y ticks from "parent" plot
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
    # labelpad to add some space between the ylabel and the axis
    axs[pick_index].set_ylabel(ch_names[ch_index], rotation="horizontal", labelpad=10)
    axs[pick_index].set_xticks([])  # remove frame index on x axis
    axs[pick_index].set_yticks([])  # remove scale on y axis
  
  # hide x labels between subplots
  for ax in axs:
    ax.label_outer()
  
  # again but with filtered data, 3rd order butterworth
  filtered_eeg_data = butter_bandpass_filter(eeg_data, 4.0, 40.0, sample_frequency, order=3, axis=1)
  
  # stack multiple subplots in one dimension
  fig = plt.figure(figsize=(20,15))
  # plt.title("Filtered signal, 3rd order Butterworth filter")
  plt.title("Butterworth Filter 3. Ordnung")
  plt.yticks([])  # remove y ticks from "parent" plot
  gridspec = fig.add_gridspec(n_picks, hspace=0)
  axs = gridspec.subplots(sharex=True, sharey=True)
  
  for pick_index, ch_index in zip(range(n_picks), picks):
    x = range(start, stop)
    y = [filtered_eeg_data[i][ch_index] for i in range(start, stop)]
    # plt.plot(x,y)
    axs[pick_index].plot(x, y)
    # parameter ylim=(-8.0,8.0)
    axs[pick_index].set_ylabel(ch_names[ch_index], rotation="horizontal", labelpad=10)
    axs[pick_index].set_xticks([])  # remove frame index on x axis
    axs[pick_index].set_yticks([])  # remove scale on y axis
  
  # hide x labels between subplots
  for ax in axs:
    ax.label_outer()


  # again but with filtered data, 6th order butterworth
  filtered_eeg_data = butter_bandpass_filter(eeg_data, 4.0, 40.0, sample_frequency, order=6, axis=1)
  
  # stack multiple subplots in one dimension
  fig = plt.figure(figsize=(20,15))
  # plt.title("Filtered signal, 6th order Butterworth filter")
  plt.title("Butterworth Filter 6. Ordnung")
  plt.yticks([])  # remove y ticks from "parent" plot
  gridspec = fig.add_gridspec(n_picks, hspace=0)
  axs = gridspec.subplots(sharex=True, sharey=True)
  
  for pick_index, ch_index in zip(range(n_picks), picks):
    x = range(start, stop)
    y = [filtered_eeg_data[i][ch_index] for i in range(start, stop)]
    # plt.plot(x,y)
    axs[pick_index].plot(x, y)
    # parameter ylim=(-8.0,8.0)
    # axs[pick_index].set(ylabel=ch_names[ch_index])
    axs[pick_index].set_ylabel(ch_names[ch_index], rotation="horizontal", labelpad=10)
    axs[pick_index].set_xticks([])  # remove frame index on x axis
    axs[pick_index].set_yticks([])  # remove scale on y axis
    
  
  # hide x labels between subplots
  for ax in axs:
    ax.label_outer()



exit()


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

