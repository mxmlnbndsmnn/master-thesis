# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 12:23:44 2022

@author: User
"""

import os
from scipy.io import loadmat
import numpy as np
from numpy import savetxt


class eeg_data_loader:
  
  eeg_data = []
  sample_frequency = 0
  num_samples = 0
  markers = []
  events = None
  
  def load_eeg_from_mat(self, subject_data_path):
    # the matlab structure is named 'o'
    mat_data = loadmat(subject_data_path)['o']
    marker_ugly = mat_data['marker'].item()
    self.eeg_data = mat_data['data'].item()
    self.marker = marker_ugly.transpose()[0]
    self.num_samples = mat_data['nS'][0][0][0][0]
    self.sample_frequency = mat_data['sampFreq'][0][0][0][0]
    return self.eeg_data
  
  
  def save_eeg_data_to_csv(self, file_path):
    savetxt(file_path, self.eeg_data, delimiter=',')
  
  
  # find all events of a marker switching from 0 to >0
  def find_all_events(self) -> list:
    hits = list()
    index = 0
    while True:
      start_i = self.find_next_event_start(start=index)
      if start_i < 0:
        break

      stop_i = self.find_next_event_stop(start=start_i+1)
      if stop_i < 0:
        break
      hit = dict()
      hit['start'] = start_i
      hit['stop'] = stop_i
      hit['event'] = self.marker[start_i]
      hits.append(hit)
      # print("Found event:")
      # print(hit)

      # continue from last event end
      index = stop_i+1
    
    self.events = hits
    return hits
  
  
  def find_next_event_stop(self, start=0):
    # careful: the start parameter in enumerate does NOT work like the start value in range
    # it only acts as an offset for the index, but always starts with the first item!
    # for i, m in enumerate(marker, start):
    for i in range(start, len(self.marker)):
      if self.marker[i] == 0:
        return i-1
    return -1
  
  
  def find_next_event_start(self, start=0):
    for i in range(start, len(self.marker)):
      # only sections where the marker changes to 1-5 are valid trials!
      if self.marker[i] in [1, 2, 3, 4, 5]:
        return i
    return -1
  
  
  # cut trials from the full eeg data
  # return a list of trial data and a list of labels
  def get_trials_x_and_y(self, duration=1., prefix_time=0.2, suffix_time=0.2):
    # reshape eeg data -> n_channels x n_times
    transposed_eeg_data = self.eeg_data.transpose()
    X = list()
    y = list()
    
    # trial duration is actually variable, but cannot be determined precisely anyway
    trial_frames = int(duration * self.sample_frequency)
    
    # (optional) start a bit earlier + extend
    prefix_frames = int(self.sample_frequency * prefix_time)
    affix_frames = int(self.sample_frequency * suffix_time)
    
    if self.events is None:
      self.find_all_events()
    for event in self.events:
      start_i = event['start'] - prefix_frames
      
      # no trial data before time 0 (should be given implicit, but who knows)
      if start_i < 0:
        print("get_trials_x_and_y: skip trial (start_i < 0)")
        continue
      # assert start_i >= 0
      
      # stop_i = event['stop']
      stop_i = start_i + trial_frames + affix_frames
      trial = np.array([[ch[i] for i in range(start_i, stop_i)] for ch in transposed_eeg_data])
      
      X.append(trial)
      
      # event type (1-5)
      y.append(event['event'])
    
    return X, y

