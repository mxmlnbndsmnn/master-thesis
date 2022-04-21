import sys
from os import path as os_path
import time
import numpy as np
import scipy.signal as signal
from scipy.stats import kurtosis
from sklearn.decomposition import FastICA
# import matplotlib.pyplot as plt
from eeg_data_loader import eeg_data_loader
from create_eeg_image import create_ctw_for_channel
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.math import confusion_matrix

print(sys.version)
print("Available GPUs:")
gpu_list = tf.config.list_physical_devices('GPU')
print(gpu_list)
if len(gpu_list) < 1:
  sys.exit()


file_index = 0


# if using an array job: assume the first argument (after the file path) to
# represent the file index in the list of all eeg data files
if len(sys.argv) == 1:
  print("Warning: Missing parameter [file_index]")
  sys.exit()

file_index = int(sys.argv[1])


eeg_data_folder = "eeg-data"
# all files
subject_data_files = ['5F-SubjectA-160405-5St-SGLHand.mat',  # 0
                      '5F-SubjectA-160408-5St-SGLHand-HFREQ.mat',
                      '5F-SubjectB-151110-5St-SGLHand.mat',
                      '5F-SubjectB-160309-5St-SGLHand-HFREQ.mat',
                      '5F-SubjectB-160311-5St-SGLHand-HFREQ.mat',
                      '5F-SubjectB-160316-5St-SGLHand.mat',
                      '5F-SubjectC-151204-5St-SGLHand.mat',
                      '5F-SubjectC-160429-5St-SGLHand-HFREQ.mat',
                      '5F-SubjectE-160321-5St-SGLHand-HFREQ.mat',
                      '5F-SubjectE-160415-5St-SGLHand-HFREQ.mat',
                      '5F-SubjectE-160429-5St-SGLHand-HFREQ.mat',  # 10
                      '5F-SubjectF-151027-5St-SGLHand.mat',
                      '5F-SubjectF-160209-5St-SGLHand.mat',
                      '5F-SubjectF-160210-5St-SGLHand-HFREQ.mat',
                      '5F-SubjectG-160413-5St-SGLHand-HFREQ.mat',
                      '5F-SubjectG-160428-5St-SGLHand-HFREQ.mat',
                      '5F-SubjectH-160804-5St-SGLHand-HFREQ.mat',
                      '5F-SubjectI-160719-5St-SGLHand-HFREQ.mat',
                      '5F-SubjectI-160723-5St-SGLHand-HFREQ.mat']  # 18

"""
# 1000Hz files
subject_data_files = ['5F-SubjectA-160408-5St-SGLHand-HFREQ.mat',  # 0
                      '5F-SubjectB-160309-5St-SGLHand-HFREQ.mat',
                      '5F-SubjectB-160311-5St-SGLHand-HFREQ.mat',
                      '5F-SubjectC-160429-5St-SGLHand-HFREQ.mat',
                      '5F-SubjectE-160321-5St-SGLHand-HFREQ.mat',
                      '5F-SubjectE-160415-5St-SGLHand-HFREQ.mat',
                      '5F-SubjectE-160429-5St-SGLHand-HFREQ.mat',
                      '5F-SubjectF-160210-5St-SGLHand-HFREQ.mat',
                      '5F-SubjectG-160413-5St-SGLHand-HFREQ.mat',
                      '5F-SubjectG-160428-5St-SGLHand-HFREQ.mat',
                      '5F-SubjectH-160804-5St-SGLHand-HFREQ.mat',  # 10
                      '5F-SubjectI-160719-5St-SGLHand-HFREQ.mat',
                      '5F-SubjectI-160723-5St-SGLHand-HFREQ.mat']
"""

subject_data_file = subject_data_files[file_index]
subject_data_path = os_path.join(eeg_data_folder, subject_data_file)

print(f"Load subject data from path: {subject_data_path}")

start_time_load_data = time.perf_counter()

eeg_data_loader_instance = eeg_data_loader()
eeg_data = eeg_data_loader_instance.load_eeg_from_mat(subject_data_path)
sample_frequency = eeg_data_loader_instance.sample_frequency
num_samples = eeg_data_loader_instance.num_samples
events = eeg_data_loader_instance.find_all_events()

num_classes = 5

###############################################################################

def get_trials_x_and_y(eeg_data, events, sfreq, duration=1., prefix_time=0.2,
                       suffix_time=0.2, downsample_step=1, ch_picks=None):
  # reshape eeg data -> n_channels x n_times
  transposed_eeg_data = eeg_data.transpose()
  X = list()
  y = list()
  
  # trial duration is actually variable, but cannot be determined precisely anyway
  trial_frames = int(duration * sfreq)
  
  # (optional) start a bit earlier + extend
  prefix_frames = int(sfreq * prefix_time)
  affix_frames = int(sfreq * suffix_time)
  
  for event in events:
    start_i = event['start'] - prefix_frames
    
    # no trial data before time 0 (should be given implicit, but who knows)
    if start_i < 0:
      print("get_trials_x_and_y: skip trial (start_i < 0)")
      continue
    # assert start_i >= 0
    
    # stop_i = event['stop']
    stop_i = start_i + trial_frames + affix_frames
    
    # for ch_index in ch_picks:
      # transposed_eeg_data[ch_index][start_i:stop_i:downsample_step]
    # this is equal to:
    # for all picked channels, select all frames (with downsample step) from start to stop
    trial = np.array([transposed_eeg_data[ch_index][start_i:stop_i:downsample_step] for ch_index in ch_picks])
    
    X.append(trial)
    
    # event type (1-5)
    y.append(event['event'])
  
  return X, y


# pick 9 channels closest to the motor cortex
# ch_picks = [2, 3, 4, 5, 6, 7, 18, 19, 20]
# pick all channels (except reference)
# ch_picks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# pick all except O1 and O2 (back of the head)
# ch_picks = [0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# pick all except T5 and T6
# ch_picks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 18, 19, 20]
# pick all except F7 and F8
# ch_picks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18, 19, 20]

ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
            'A1', 'A2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz', 'X3']

# pick all channels except reference and Fp1, Fp2
ch_picks = [2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20]
print("Use EEG channels:")
print([ch_names[i] for i in ch_picks])

downsample_step = 1
if sample_frequency == 1000:
  downsample_step = 5
elif sample_frequency == 200:
  downsample_step = 1
else:
  raise RuntimeError("Unexpected target frequency:", sample_frequency)

trials, labels = get_trials_x_and_y(eeg_data, events, sample_frequency,
                                    downsample_step=downsample_step, ch_picks=ch_picks)

# X_raw = np.array(trials)  # use with "fake" labels (for 2 class problems)
# y = np.array(labels) - 1  # labels should range from 0-4 (?)

print("trial shape:", type(trials[0]), trials[0].shape)  # trials is a simple list
# print("y:", type(y), y.shape)

end_time_load_data = time.perf_counter()
print(f"Time to load EEG-data: {end_time_load_data-start_time_load_data:.2f}s")

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

print("Bandpass filter EEG data (4-40Hz)")
start_time_bandpass = time.perf_counter()
eeg_data = butter_bandpass_filter(eeg_data, 4.0, 40.0, sample_frequency, order=6, axis=1)

# TODO: apply bandpass filter before cutting trials from the eeg_data

end_time_bandpass = time.perf_counter()
print(f"Time to apply bandpass filter: {end_time_bandpass-start_time_bandpass:.2f}s")

###############################################################################

# generate the "images" per channel for all trials from 2 classes (1 vs 1)
# pick trials and labels to do a 1 vs 1 classification
# only pick trials for two classes, discard the rest
"""
num_classes = 2

# compare 0-1, 0-2, 0-3, 0-4, 1-2, 1-3, 1-4, 2-3, 2-4, 3-4
comparisons = [[0,1], [0,2], [0,3], [0,4], [1,2], [1,3], [1,4], [2,3], [2,4], [3,4]]
comparison_index = int(sys.argv[1])

first_class = comparisons[comparison_index][0]
second_class = comparisons[comparison_index][1]
print(f"Compare class {first_class} vs {second_class}")
print("Trials per class:", first_class, "=", (y==first_class).sum())
print("Trials per class:", second_class, "=", (y==second_class).sum())

list_of_trial_data = []
list_of_fake_labels = []

# CTW images
for trial, label in zip(X_raw, y):
  fake_label = 0
  if label == first_class:
    fake_label = 0
  elif label == second_class:
    fake_label = 1
  else:
    continue  # skip all other classes
  
  trial_data = []
  for ch_index in ch_picks:
    ch = trial[ch_index]
    cwt = create_ctw_for_channel(ch, widths_max=40)
    trial_data.append(cwt)
  list_of_trial_data.append(trial_data)
  list_of_fake_labels.append(fake_label)

X = np.array(list_of_trial_data)
print("X:", type(X), X.shape)
# should be (num_trials, num_channels, num_f, num_t)
y = np.array(list_of_fake_labels)
"""

###############################################################################

# generate the "images" per channel for all trials - normal (5 classes)
list_of_trial_data = []
list_of_labels = []

start_time_cwt = time.perf_counter()
# CTW images
num_bad_components = 0
num_bad_trials = 0
num_removed_trials = 0
for trial, label in zip(trials, labels):
  
  ica = FastICA(n_components=6, random_state=42)
  ica_sources = ica.fit_transform(trial)  # get the estimated sources
  sources_t = ica_sources.T
  bad_components_per_trial = 0
  for i, source in enumerate(sources_t):
    if kurtosis(source) > 8:
      sources_t[i][:] = 0
      num_bad_components += 1
      bad_components_per_trial += 1
  
  # allow one "bad" component per trial (that is removed by ICA repair anyway)
  if bad_components_per_trial > 0 :
    num_bad_trials += 1
  
  # skip this trial entirely if more than one "bad" component
  if bad_components_per_trial > 1:
    num_removed_trials += 1
    continue
  
  # after removing components that are considered "bad", reconstruct the mixed data
  # TODO maybe only do this repair if bad_components_per_trial > 0?
  trial = ica.inverse_transform(sources_t.T)
  
  trial_data = []
  for ch in trial:
  # for ch_index in ch_picks:
    # ch = trial[ch_index]
    cwt = create_ctw_for_channel(ch, widths_max=40)
    trial_data.append(cwt)
  
  list_of_trial_data.append(trial_data)
  list_of_labels.append(label)

X = np.array(list_of_trial_data)
print("X:", type(X), X.shape)
# should be (num_trials, num_channels, num_f, num_t) (this was for STFT images...)

# labels must match the trials after some might have been removed
y = np.array(list_of_labels) - 1  # 0-4 instead of 1-5
print("y:", type(y), y.shape)

end_time_cwt = time.perf_counter()
print(f"Time to generate CWTs: {end_time_cwt-start_time_cwt:.2f}s")

print(f"ICA detected {num_bad_components} bad components in {num_bad_trials} trials.")
print(f"Removed {num_removed_trials} trials with more than one bad component.")

###############################################################################

# some layers (conv, max pool) assume the input to have the channels as the
# last dimension (channels_last), e.g. (batch, dim1, dim2, channel)
X = tf.constant(X)
X = tf.transpose(X, perm=[0,2,3,1])
input_shape = X[0].shape

dataset = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))

num_trials = len(dataset)
print(f"Total number of trials: {num_trials}")


# to be called after the dataset has been preprocessed and split for k-fold cv
def configure_for_performance(ds):
  ds = ds.cache()
  # ds = ds.shuffle(buffer_size=len(ds))
  ds = ds.batch(16)
  ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
  return ds


# calculate precision, recall and f1 score from confusion matrix
def calculate_cm_scores(cm):
  tp = cm.diagonal()
  tp_and_fn = cm.sum(1)
  tp_and_fp = cm.sum(0)
  # note: might run into div by zero!
  precision = tp / tp_and_fp # how many predictions for this class are correct
  recall = tp / tp_and_fn # how many of the class trials have been found
  f_score = []
  # print("Precision:")
  # print(precision)
  # print("Recall:")
  # print(recall)
  for pr, re in zip(precision, recall):
    f1 = 0
    if pr > 0 or re > 0:
      f1 = 2 * (pr*re) / (pr+re)
    f_score.append(f1)
  # print("F1 score:")
  # print(f_score)
  return precision, recall, f_score


# split the dataset for k-fold cross-validation
k = 10
valid_size = int(num_trials / k)
print(f"Create {k} folds of size {valid_size}")

# calculate the average metrics
all_acc_train = []
all_acc_valid = []

# sum over all confusion matrices
cumulative_cm = None

start_time_train = time.perf_counter()

for i in range(k):
  print("-"*80)
  print(f"Fold {i+1}:")
  train_ds = dataset.take(i*valid_size).concatenate(dataset.skip((i+1)*valid_size))
  valid_ds = dataset.skip(i*valid_size).take(valid_size)
  train_ds = configure_for_performance(train_ds)
  valid_ds = configure_for_performance(valid_ds)
  
  # create the model
  model = Sequential()
  
  model.add(layers.Conv2D(30, 5, padding="same", activation="elu", input_shape=input_shape))
  # print(model.output_shape)
  model.add(layers.BatchNormalization())
  model.add(layers.MaxPooling2D(pool_size=(3,1)))
  model.add(layers.Dropout(0.3))
  model.add(layers.Conv2D(60, 7, padding="same", activation="elu"))
  model.add(layers.BatchNormalization())
  model.add(layers.MaxPooling2D(pool_size=(3,1)))
  model.add(layers.Dropout(0.3))
  model.add(layers.Conv2D(90, 7, padding="same", activation="elu"))
  model.add(layers.BatchNormalization())
  model.add(layers.MaxPooling2D(pool_size=(3,1)))
  model.add(layers.Dropout(0.3))
  model.add(layers.Flatten())
  # model.add(layers.Dense(64, activation="elu"))
  model.add(layers.Dense(32, activation="elu"))
  model.add(layers.Dense(num_classes, activation="softmax"))
  
  # instantiate an optimizer
  learn_rate = 0.001
  print(f"Learn rate: {learn_rate}")
  optimizer = tf.keras.optimizers.Adagrad(
    learning_rate=learn_rate, initial_accumulator_value=0.1, epsilon=1e-07,
    name='Adagrad')
  
  # compile the model
  model.compile(optimizer=optimizer,
                # from_logits=True (if not using a softmax activation as last layer)
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=["accuracy"])
  
  # early stopping
  # monitors the validation accuracy and stops training after [patience] epochs
  # that show no improvements
  es_callback = tf.keras.callbacks.EarlyStopping(monitor="accuracy",
                                                 patience=4,
                                                 restore_best_weights=True)
  
  # train the model
  num_epochs = 80
  print(f"Training for up to {num_epochs} epochs.")
  history = model.fit(train_ds, validation_data=valid_ds, epochs=num_epochs,
                      verbose=0, callbacks=[es_callback])
  
  # using early stopping, the actual number of epochs might be lower than num_epochs!
  true_num_epochs = len(history.history["loss"])
  if true_num_epochs < num_epochs:
    print(f"Early stop training after epoch {true_num_epochs}")
  
  # calculate average metrics
  # get the highest accuracy for validation + training acc for the same epoch
  best_valid_epoch = np.array(history.history['val_accuracy']).argmax()
  best_valid_acc = np.array(history.history['val_accuracy']).max()
  acc_valid = best_valid_acc
  # get the train acc for the same epoch as the valid acc
  acc_train = np.array(history.history['accuracy'])[best_valid_epoch]
  print(f"Highest validation accuracy ({best_valid_acc:.3f}) at epoch "\
        f"{best_valid_epoch+1} (training accuracy is {acc_train:.3f})")
  # print(history.history)
  all_acc_train.append(acc_train)
  all_acc_valid.append(acc_valid)
  
  true_labels = []
  for sample_batch, label_batch in valid_ds:
    for label in label_batch:
      true_labels.append(label.numpy())
  
  predicted_labels = []
  predictions = model.predict(valid_ds)
  for prediction in predictions:
    # score = tf.nn.softmax(prediction).numpy()
    # since the last layer already uses a softmax activation
    # there is no need to calculate the softmax again!
    predicted_labels.append(np.argmax(prediction))
  cm = confusion_matrix(true_labels, predicted_labels).numpy()
  # plot_confusion_matrix(cm, "Konfusionsmatrix, Fold "+str(i+1))
  print(cm)
  
  if cumulative_cm is None:
    cumulative_cm = cm
  else:
    cumulative_cm = cumulative_cm + cm
  
  precision, recall, f_score = calculate_cm_scores(cm)
  print("Precision:", precision, "Mean:", np.array(precision).mean())
  print("Recall:", recall, "Mean:", np.array(recall).mean())
  print("F1 Score:", f_score, "Mean:", np.array(f_score).mean())

end_time_train = time.perf_counter()
print(f"Time to train models: {end_time_train-start_time_train:.2f}s")

# only need to print this once
model.summary()

# get the mean accuracy and standard deviation from all folds
all_acc_train = np.array(all_acc_train)
all_acc_valid = np.array(all_acc_valid)
acc_mean_train = all_acc_train.mean()
acc_std_train = all_acc_train.std()
acc_mean_valid = all_acc_valid.mean()
acc_std_valid = all_acc_valid.std()
print(f"Mean accuracy (train) is {acc_mean_train:.3f} and STD is {acc_std_train:.3f}")
print(f"Mean accuracy (valid) is {acc_mean_valid:.3f} and STD is {acc_std_valid:.3f}")


print("Cumulative confusion matrix:")
print(cumulative_cm)

# precision and recall per class
precision, recall, f_score = calculate_cm_scores(cumulative_cm)
print("Mean precision per class:")
for i, p in enumerate(precision):
  print(f"{i}: {p:.2f}")
print("Mean recall per class:")
for i, r in enumerate(recall):
  print(f"{i}: {r:.2f}")
print("Mean F1 score per class:")
for i, f1 in enumerate(f_score):
  print(f"{i}: {f1:.2f}")

# use for 2 class problems
# print(f"first_class: {first_class}")
# print(f"second_class: {second_class}")

