# CNN-based Classification of Finger Motor Imagery EEG Data

## Description
A pipeline to generate time-frequency representations of EEG data recorded for 5 different movement intention tasks and a CNN to classify them.
Written in python (3) with Tensorflow and Keras for the machine learning parts.

![cwt_example](https://user-images.githubusercontent.com/46690491/178151463-f284de48-6227-4018-8604-21cac7698f7e.png)

## Pipeline

The main file is "tf_script.py" located in the "remote scripts" folder.
After loading the EEG data from .mat (MATLAB) files (see [1], [2]), individual trials are extracted and filtered.
In this step, data from specific channels (electrodes) can be included/excluded.
Next, CWT (continuous wavelet transform) images are created for each channel and a tensorflow dataset is created thereof.
Note that all images that belong to one trial are passed to the neural net in parallel.
k-fold cross-validation is employed to train k models which are then evaluated based on their achieved accuracy.
Precision and recall values as well as confusion matrices are tracked, too.

## CNN architecture
Two models are present which include 2 or 3 convolutional layers each.
The basic structure looks like this:
- conv2d layer (ELU activation)
- batch norm layer
- max pooling 2d layer
- dropout layer

(repeat 2x or 3x)
- flatten layer
- dense layer (output dimension = number of MI classes, softmax activation)

## Usage notes
- The script is currently expecting at least one GPU to be available - to change that, simply remove the check at the beginning of the script
- Multiple EEG data files can be processed one after the other, one additional (integer) argument when executing the script can be used to define the file index in the predefined list of input data files (by default in a folder called "eeg-data")
- As of now, the output (achieved performance, confusion matrix etc.) can simply directly be obtained from the Python console output

## Main references
[1] Utilized data plus description: https://figshare.com/collections/A_large_electroencephalographic_motor_imagery_dataset_for_electroencephalographic_brain_computer_interfaces/3917698

[2] Associated paper: https://www.nature.com/articles/sdata2018211
