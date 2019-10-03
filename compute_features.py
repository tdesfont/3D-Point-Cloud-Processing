#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Efficient implementation of features computation.
"""

#%% Imports

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from utils.ply import write_ply, read_ply

import time

import pickle

from os.path import exists

from features_computation.descriptors import local_PCA, compute_features

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import sys

#%% Training data for the ground classifier


training_path = sys.argv[1]
# training_path = 'data_points_cloud/training/MiniLille1.ply'

training_features = np.empty((0, 4))

cloud_ply = read_ply(training_path)
points = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

# data is very volumetric ! 2Mo points
print("Points shape:", points.shape)

#%% Compute SVD on the point cloud

n = points.shape[0]
d = points.shape[1]
barycenter = np.mean(points, axis=0)
Q = points - barycenter
cov_mat = (1/n)*Q.T@Q
assert cov_mat.shape == (d, d)
eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)

#%%

bucket_size = 20
bucket_residual = 3

# Projection on the principal axis
scalar_product = (points-barycenter)@eigenvectors[:, 2]
hash_index = scalar_product//bucket_size
chunck_ids = np.unique(hash_index)

selected_index = 0
scalar_product[np.where(hash_index==selected_index)]
upper_bound = (selected_index+1)*bucket_size
lower_bound = selected_index*bucket_size

interest_indexes = np.where(hash_index==selected_index)[0]
fuzzy_indexes = np.where((scalar_product <= upper_bound+bucket_residual)*\
         (scalar_product >= lower_bound-bucket_residual))[0]

# Store points with such voxelization
write_ply('./data_processing/interest_points.ply',
          [points[interest_indexes], np.ones(interest_indexes.shape)*10],
          ['x', 'y', 'z', 'color'])

write_ply('./data_processing/interest_and_boundary_points.ply',
          [points[fuzzy_indexes], np.ones(fuzzy_indexes.shape)*0],
          ['x', 'y', 'z', 'color'])

#%% Iterate for features computations

radius = 0.5

features = np.empty((0, 4))
features_index = []

feature_file = 'features/training/'+training_path.split('/')[-1].split('.')[0] + '_features.npy'
print(feature_file)

ii = 0
for selected_index in chunck_ids:
    ii += 1
    print('Compute features on chunck: {}/{}'.format(ii, len(chunck_ids)))

    scalar_product[np.where(hash_index==selected_index)]
    upper_bound = (selected_index+1)*bucket_size
    lower_bound = selected_index*bucket_size

    interest_indexes = np.where(hash_index==selected_index)[0]
    fuzzy_indexes = np.where((scalar_product < upper_bound+bucket_residual)*\
             (scalar_product >= lower_bound-bucket_residual))[0]

    print(fuzzy_indexes.shape)

    vert, line, plan, sphe = compute_features(points[interest_indexes],
                                              points[fuzzy_indexes], radius)

    slice_features = np.vstack((vert.ravel(), line.ravel(), plan.ravel(), sphe.ravel())).T

    features_index += list(interest_indexes)
    features = np.vstack((features, slice_features))

assert features.shape[0]==points.shape[0]

# sort on index to preserve initial order of points
df = pd.DataFrame(data=features, index=features_index)
features = df.sort_index().as_matrix()

np.save(feature_file, features)
