# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 10:10:57 2021

@author: User
"""

# TODO
# detect/repair (ocular) artifacts using ICA
# see: https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html

# import numpy as np
import mne
from mne.preprocessing import ICA #, create_eog_epochs, corrmap

# raw_data = mne.io.read_raw_fieldtrip("BCI_Comp4_Dataset_2b/A01T.mat", None, "A01T" )
# epoch_data = mne.io.read_epochs_fieldtrip("BCI_Comp4_Dataset_2b/A01E.mat", None, "01" )

# read the raw data from the file
raw_data = mne.io.read_raw_gdf("BCICIV_2a_gdf/A01E.gdf")
# raw_data = mne.io.read_raw_gdf("BCI_Comp3_Dataset_3/k3b.gdf")
# print some info
print(raw_data)
print(raw_data.info)
# print(raw_data.info['ch_names'])

# crop to only use a few seconds
# also load the data into RAM
raw_data.crop(tmin=360).load_data()

# mne.find_events(raw_data) does not work here as there is no stim channel
# convert annotations to events, see
# https://mne.tools/stable/auto_tutorials/intro/20_events_from_raw.html#tut-events-vs-annotations
# https://mne.tools/stable/generated/mne.events_from_annotations.html#mne.events_from_annotations
events, event_dict = mne.events_from_annotations(raw_data)

# plot (default plot duration is 10s)
# duration=5
scalings = dict(eeg=60e-6) # default is eeg=20e-6
#raw_data.plot(n_channels=8, start=367, duration=20, show_scrollbars=False, scalings=scalings)
raw_data.plot(events=events, n_channels=8, duration=40, show_scrollbars=False, scalings=scalings)
# not sure how to display the events
# the only visible difference is a "2" at the top left of the plot when passing in the events
# for the comp3 dataset, this seems to work better

# check for ocular artifacts
# error in create_eog_epochs: could not find any EOG channels
# inspecting raw_data.info['chs'] shows that even the channels named EOG-something
# are of 'kind' 'FIFFV_EEG_CH' (not EOG?)
# eog_evoked = create_eog_epochs(raw_data).average()
# eog_evoked.apply_baseline(baseline=(None, -0.2))
# eog_evoked-plot_joint()

# create a copy of the original raw data
raw2 = raw_data.copy()
# filter: remove low-frequency drifts; 1Hz high-pass filter
raw2.load_data().filter(l_freq=1., h_freq=None)

# set up the ICA
# nr of components is an initial guess
# random state to produce the same results on every run (ICA fitting is not deterministic)
ica = ICA(n_components=15, max_iter='auto', random_state=42)
ica.fit(raw2)

# plot the captured components
# use the unfiltered raw data here
# raw_data.load_data()
ica.plot_sources(raw_data, show_scrollbars=False)

# observation: IC #0 is a straight line
# IC #14 almost looks like strong white (gray?) noise

