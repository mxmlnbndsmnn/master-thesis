# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 20:23:12 2022

@author: User
"""

import numpy as np
# import tensorflow as tf
from tensorflow.math import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt


if False:
  true_labels = [1,2,3,4,5,1,2,3,4,5,1,2,3,4,5]
  predicted_labels = [1,2,3,4,4,2,1,3,4,5,3,4,1,4,2]
  
  # labels must be int (?)
  # using 6 classes instead of 5 here
  cm = confusion_matrix(true_labels,predicted_labels).numpy()
  # cut off the 0th row and column
  cm = cm[1:, 1:]
  print(cm)
  
  # cmap copper looks good, summer too
  sn.heatmap(cm, annot=True, cmap='summer', cbar=False)
  
  plt.show()


# test for kfold cv split
if True:
  X = [[1,1], [2,2], [3,3], [4,4], [5,5], [6,6], [7,7], [8,8], [9,9], [10,10]]
  y = [1,2,3,4,5,6,7,8,9,10]
  k = 5
  num_trials = len(X)
  valid_size = int(num_trials / k)
  for i in range(k):
    train_X = np.delete(X, np.s_[i*valid_size:(i+1)*valid_size], axis=0)
    train_y = np.delete(y, np.s_[i*valid_size:(i+1)*valid_size], axis=0)
    valid_X = X[i*valid_size:(i+1)*valid_size]
    valid_y = y[i*valid_size:(i+1)*valid_size]
    
    print(train_X)
    print(train_y)
    
    print(valid_X)
    print(valid_y)
  