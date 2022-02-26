# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 14:39:09 2022

@author: User
"""

import os
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt


# for a single trial; calculate the stft and create an image
# save it to a folder - one folder per event type (1-5)
def create_stft_image_for_trial(trial, freq, path, picks=None, nperseg=40,
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
    
    # remove frequencies above 40Hz
    keep_index = np.searchsorted(f, 40)
    # f = f[:keep_index+1]
    absolutes = absolutes[:keep_index+1, :]
    
    if image_data is not None:
      image_data = np.append(image_data, absolutes, axis=axis)
    else:
      image_data = absolutes
  
  # print(image_data.shape)
  img_path = os.path.join(path, file_name)
  # plt.imsave(img_path, image_data, vmin=0, vmax=2)
  plt.imsave(img_path, image_data)



# plot a single trial STFT image for one channel
def plot_trial_stft(ch, nperseg=40, sample_frequency=200, file_name=None):
  
  # segment length is 256 by default, but input length is only 200 or 1000 here
  f,t,Zxx = signal.stft(ch, fs=sample_frequency, nperseg=nperseg)
  # for a sampling frequency of 200 this yields:
  # f - array of sample frequencies -> shape (33,)
  # t - array of segment times -> shape (8,)
  # Zxx - STFT of x -> shape (33, 8)
  
  # find the index to cut off the data for higher frequencies
  keep_index = np.searchsorted(f, 40)
  # remove frequencies abouve 40 Hz
  f = f[:keep_index+1]
  Zxx = Zxx[:keep_index+1, :]
  
  # shading should be either nearest or gouraud
  # or flat when making the color map (Zxx) one smaller than t and f:
  cutZxx = Zxx[:-1,:-1]
  plt.pcolormesh(t, f, np.abs(cutZxx), vmin=0, vmax=2, shading='flat')
  plt.title(f'STFT Magnitude (segment length: {nperseg})')
  plt.ylabel('Frequency [Hz]')
  plt.xlabel('Time [sec]')
  if file_name is None:
    plt.show() # ? not needed; do not use for savefig to work!
    return
  
  # why does it look like more than 1 second? because of the segment length?
  # yep, apparently since it fits when using dividers of 200 like 50
  
  # save the result to an image file
  absolutes = np.abs(Zxx)
  image_data = absolutes
  # print(image_data.shape)
  # plt.savefig(file_name) # with axis and labels
  plt.imsave(file_name, image_data) # data only
  # plt.imsave(file_name, image_data, vmin=0, vmax=2) # data only


# Calculate the STFT values for data from one channel.
# Channel data is expected to be of shape (n_channels, n_times)
def create_stft_for_channel(ch, sample_frequency=200, nperseg=40, high_cut_freq=40, file_name=None):
  
  f,t,Zxx = signal.stft(ch, fs=sample_frequency, nperseg=nperseg)
  
  # find the index to cut off the data for higher frequencies
  if high_cut_freq is not None:
    keep_index = np.searchsorted(f, high_cut_freq)
    # remove frequencies above high_cut_freq Hz
    f = f[:keep_index+1]
    Zxx = Zxx[:keep_index+1, :]
  
  stft = np.abs(Zxx)
  if file_name is not None:
    plt.imsave(file_name, stft)
  
  return stft, f, t
  

