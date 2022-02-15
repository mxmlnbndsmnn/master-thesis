# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 11:20:09 2022

@author: User
"""

import matplotlib.pyplot as plt
import seaborn as sn
from tensorflow.math import confusion_matrix


def get_confusion_matrix(true_labels, predicted_labels):
  return confusion_matrix(true_labels, predicted_labels).numpy()


def plot_confusion_matrix(cm, title=None, class_names=None):
  if class_names is None:
    num_classes = cm.shape[0]
    class_names = [str(i) for i in range(num_classes)]
  ticklabels = [c for c in class_names]
  sn.heatmap(cm, annot=True, xticklabels=ticklabels, yticklabels=ticklabels,
             cmap='summer', cbar=False, fmt="d")
  plt.xlabel("Predicted labels")
  plt.ylabel("True labels")
  if title is not None:
    plt.title(title)
  plt.show()


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

