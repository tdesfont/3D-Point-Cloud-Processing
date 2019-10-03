#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    S-Voxel Classifier
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

training_path = "./features_svoxel/training/"
available_files = sorted(os.listdir(training_path))

X = np.empty((0, 8))
for file in available_files:
    df = pd.read_csv(training_path+file)
    df = df[['Bx', 'By', 'Bz', 'verticality', 'linearity', 'planarity', 'sphericity', 'label']]
    X = np.vstack((X, df.as_matrix()))

X_blcd = np.empty((0, 8))
for label in [0, 1, 2, 5, 6]:
    data_per_class = X[np.where(X[:, -1]==label)]
    nbr_class = len(data_per_class)
    indexes = np.arange(nbr_class)
    indexes = np.random.choice(indexes, 3000, replace=False)
    X_blcd = np.vstack((X_blcd, data_per_class[indexes]))

#%%

clf = RandomForestClassifier()
X_train, X_valid, y_train, y_valid = train_test_split(X_blcd[:, :7], X_blcd[:, 7],
                                                      test_size=0.33, shuffle=True)
clf.fit(X_train, y_train)
y_predict = clf.predict(X_valid)

print(classification_report(y_valid, y_predict))
