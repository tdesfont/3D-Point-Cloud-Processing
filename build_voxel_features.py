#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Voxel Features
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

from collections import Counter

#%%

radius = 0.5
num_per_class = 500
label_names = {0: 'Unclassified',
               1: 'Ground',
               2: 'Building',
               3: 'Poles',
               4: 'Pedestrians',
               5: 'Cars',
               6: 'Vegetation'}

training_features = np.empty((0, 4))
training_labels = np.empty((0,))

training_path = './data_wo_ground/training'
file = os.listdir(training_path)[0]

cloud_ply = read_ply(join(training_path, file))
points = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T
labels = cloud_ply['class']

training_inds = np.empty(0, dtype=np.int32)

for label, name in label_names.items():
    if label==0 or label==1:
        continue
    label_inds = np.where(labels==label)[0]
    if len(label_inds) <= num_per_class:
        training_inds = np.hstack((training_inds, label_inds))
    else:
        random_choice = np.random.choice(len(label_inds), num_per_class, replace=False)
        training_inds = np.hstack((training_inds, label_inds[random_choice]))

training_points = points[training_inds, :]

vert, line, plan, sphe = compute_features(training_points, points, radius)
features = np.vstack((vert.ravel(), line.ravel(), plan.ravel(), sphe.ravel())).T

training_features = np.vstack((training_features, features))
training_labels = np.hstack((training_labels, labels[training_inds]))

#%% Train a model to do the prediction

clf = RandomForestClassifier()
X_train, X_valid, y_train, y_valid = train_test_split(training_features, training_labels, test_size=0.33)
clf.fit(X_train, y_train)
y_predict = clf.predict(X_valid)

print(classification_report(y_valid, y_predict))

#%%

validation_file = './sandbox/training_chunck.ply'

cloud_ply = read_ply(validation_file)
points = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T
labels = cloud_ply['class']

training_features = np.empty((0, 4))
training_labels = np.empty((0,))

vert, line, plan, sphe = compute_features(points, points, radius)
features = np.vstack((vert.ravel(), line.ravel(), plan.ravel(), sphe.ravel())).T
training_features = np.vstack((training_features, features))
training_labels = np.hstack((training_labels, labels))

y_predict = clf.predict(training_features)
proba_predict = clf.predict_proba(training_features)
confidence = np.max(proba_predict, axis=1)>0.95

write_ply('./sandbox/high_confidence.ply',
      [points[confidence], training_labels[confidence]], ['x', 'y', 'z', 'class'])

write_ply('./sandbox/high_confidence_predict.ply',
      [points[confidence], y_predict[confidence]], ['x', 'y', 'z', 'predict'])



#%%

points[labels==5]

training_features = np.empty((0, 4))

vert, line, plan, sphe = compute_features(points, points, radius)
features = np.vstack((vert.ravel(), line.ravel(), plan.ravel(), sphe.ravel())).T
training_features = np.vstack((training_features, features))

y_predict = clf.predict(training_features)
class_proba = clf.predict_proba(training_features)