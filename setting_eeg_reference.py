# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 11:01:36 2021

@author: User

See MNE Resource:
https://mne.tools/stable/auto_tutorials/preprocessing/55_setting_eeg_reference.html
"""

import os
import mne

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file, verbose=False)
raw.crop(tmax=60).load_data()
raw.pick(['EEG 0{:02}'.format(n) for n in range(41, 60)])

raw.plot()

# add new reference channel (all zero)
raw_new_ref = mne.add_reference_channels(raw, ref_channels=['EEG 999'])
raw_new_ref.plot()

# set reference to `EEG 050`
raw_new_ref.set_eeg_reference(ref_channels=['EEG 050'])
raw_new_ref.plot()

# setting the reference will not affect any "bad" channels

# raw object is modified in-place, create a copy first
# use the average of all channels as reference
raw_avg_ref = raw.copy().set_eeg_reference(ref_channels='average')
raw_avg_ref.plot()


# creating the average reference as a projector
raw.set_eeg_reference('average', projection=True)
#print(raw.info['projs'])

for title, proj in zip(['Original', 'Average'], [False, True]):
    fig = raw.plot(proj=proj, n_channels=len(raw))
    # make room for title
    fig.subplots_adjust(top=0.9)
    fig.suptitle('{} reference'.format(title), size='xx-large', weight='bold')

raw_bip_ref = mne.set_bipolar_reference(raw, anode=['EEG 054'], 
                                        cathode=['EEG 055'])
raw_bip_ref.plot()
