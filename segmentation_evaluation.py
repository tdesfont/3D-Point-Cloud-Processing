#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Build a metric on the segmentation done with S-Voxel
"""

#%% Imports

import time
from os import listdir
from os.path import exists, join
import os

import numpy as np

import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KDTree

from utils.ply import write_ply, read_ply

from features_computation.descriptors import local_PCA, compute_features
import tqdm

from collections import namedtuple
from collections import defaultdict

import pandas as pd

import matplotlib.pyplot as plt

from math import sqrt, pi

#%%

path = './segmentation_results/'

total_seg_points = np.empty((0, 3))
total_seg_labels = []

for file_id in range(-3, 3):
    seg_file = 'segmentation_slice_{}.ply'.format(file_id)
    cloud_ply = read_ply(path+seg_file)
    points = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T
    colors = np.vstack((cloud_ply['Red'], cloud_ply['Green'], cloud_ply['Blue'])).T
    seg_labels = np.sum(colors, axis=1)

    total_seg_points = np.vstack((total_seg_points, points))
    total_seg_labels += list(seg_labels)

hash_val = np.sum(total_seg_points, axis=1)

index = np.arange(len(total_seg_labels))
index = sorted(index, key = lambda x: hash_val[x])

total_seg_points = total_seg_points[index]

total_seg_labels = np.array(total_seg_labels)
total_seg_labels = total_seg_labels[index]

#%%

total_gt_points = np.empty((0, 3))
total_gt_labels = []

for file_id in range(-3, 3):
    seg_file = 'slice_{}.ply'.format(file_id)
    cloud_ply = read_ply(path+seg_file)
    points = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T
    labels = cloud_ply['class']

    total_gt_points = np.vstack((total_gt_points, points))
    total_gt_labels += list(labels)

hash_val = np.sum(total_gt_points, axis=1)

index = np.arange(len(total_gt_labels))
index = sorted(index, key = lambda x: hash_val[x])

total_gt_points = total_gt_points[index]

total_gt_labels = np.array(total_gt_labels)
total_gt_labels = total_gt_labels[index]

#%% Merge

points = total_gt_points
segmentation_labels = total_seg_labels
labels = total_gt_labels

#%%

from collections import Counter

predictions = np.ones(points.shape[0])*-1
for segment_id in np.unique(segmentation_labels):
    count = Counter(labels[segmentation_labels==segment_id])
    majority_label = count.most_common(1)[0][0]
    predictions[segmentation_labels == segment_id] = majority_label

assert -1 not in predictions

plt.figure()
plt.hist(labels, edgecolor='k', alpha=0.7)
plt.hist(predictions, edgecolor='k', alpha=0.7)
plt.show()

#%%
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

#%%

label_names= {0: 'Unclassified',
              1: 'Ground',
              2: 'Building',
              3: 'Poles',
              4: 'Pedestrians',
              5: 'Cars',
              6: 'Vegetation'}

class_names = np.array([label_names[i] for i in range(7)])

#%%

plot_confusion_matrix(labels, predictions, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

#%%

print(classification_report(labels, predictions))

#%%

write_ply('./segmentation_evaluation/prediction.ply',
      [points[labels!=0], predictions[labels!=0]], ['x', 'y', 'z', 'pred'])

write_ply('./segmentation_evaluation/ground_truth.ply',
      [points[labels!=0], labels[labels!=0]], ['x', 'y', 'z', 'ground_truth'])

# print(classification_report(labels, predictions))

#%%

accurate = (labels == predictions)*1

write_ply('./segmentation_evaluation/accurate.ply',
      [points[labels!=0], accurate[labels!=0].astype('float')], ['x', 'y', 'z', 'ground_truth'])
