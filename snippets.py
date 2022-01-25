# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 20:23:12 2022

@author: User
"""

import tensorflow as tf
from tensorflow.math import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt

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
