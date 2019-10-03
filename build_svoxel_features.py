#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Create the training features for the s-voxels
    Load data by chunck and store the features files
    Train a classifier on those s-voxels
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

# HPO
bucket_size = 20
bucket_residual = 3

fields = ['voxel_key',
          'density',
          'Vx', 'Vy', 'Vz',
          'Bx', 'By', 'Bz',
          'Sx', 'Sy', 'Sz',
          'verticality', 'linearity', 'planarity', 'sphericity',
          'label']


path = './data_wo_ground/training/'
for file in os.listdir(path):

    training_path = path + file
    print(training_path)

    # Retrieve a slice the point cloud wo ground
    cloud_ply = read_ply(training_path)
    points = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T
    labels = cloud_ply['class']

    # Compute SVD on the point cloud
    n = points.shape[0]
    d = points.shape[1]
    barycenter = np.mean(points, axis=0)
    Q = points - barycenter
    cov_mat = (1/n)*Q.T@Q
    assert cov_mat.shape == (d, d)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)

    # Projection on the principal axis
    scalar_product = (points-barycenter)@eigenvectors[:, 2]
    hash_index = scalar_product//bucket_size
    chunck_ids = np.unique(hash_index)

    ii = 0
    for selected_id in chunck_ids:

        s_voxel_features = np.empty((0, len(fields)))

        ii += 1
        print('Processing point cloud {}/{}'.format(ii, len(chunck_ids)))

        upper_bound = (selected_id+1)*bucket_size
        lower_bound = selected_id*bucket_size

        interest_indexes = np.where(hash_index==selected_id)[0]
        fuzzy_indexes = np.where((scalar_product <= upper_bound+bucket_residual)*\
             (scalar_product >= lower_bound-bucket_residual))[0]

        # Restrict the points
        points_ = points[interest_indexes]
        labels_ = labels[interest_indexes]

        # 3.1 Voxelisation of the data
        N = points_.shape[0]
        indexes = np.arange(N)

        t0 = time.time()
        print('Building KDTree...')
        kd = KDTree(points_, metric='minkowski')
        t1 = time.time()
        print('KDTree built in {} sec'.format(t1-t0))

        # Single-Shot query for cubical voxels
        radius=0.1

        t0 = time.time()
        neighborhoods_inner_sphere = kd.query_radius(points_, r=radius)
        t1 = time.time()
        print('Query time for computing neighborhoods on all points: {} sec'.format(t1-t0))

        t0 = time.time()
        neighborhoods_outer_sphere = kd.query_radius(points_, r=radius*sqrt(3))
        t1 = time.time()
        print('Query time for computing neighborhoods on all points: {} sec'.format(t1-t0))

        # Cubical voxel without replacement
        N = points_.shape[0]
        assert neighborhoods_inner_sphere.shape[0] == N
        assert neighborhoods_outer_sphere.shape[0] == N

        assignated = np.array([False for i in range(N)])
        assignated_voxel = np.array([-1 for i in range(N)])

        rec_nbr_created_voxels = []
        rec_nbr_assignated_points = []

        voxel_key = -1

        for index in tqdm.tqdm(range(N)):

            if assignated[index]:
                continue

            # 1/2 CUBIC VOXEL ASSIGNATION

            inner_index = neighborhoods_inner_sphere[index]
            outer_index = neighborhoods_outer_sphere[index]

            # Compute the points in the cubic voxel
            in_cubic = np.prod(points_[outer_index] <= np.max(points_[inner_index], axis=0), axis=1)*\
                       np.prod(points_[outer_index] >= np.min(points_[inner_index], axis=0), axis=1)

            # Select the points in cubic voxels
            in_cubic_index = outer_index[np.where(in_cubic)]
            # Filter on assignated or not
            non_assignated_in_cubic_index = in_cubic_index[~assignated[in_cubic_index]]

            # Update assignated points
            assignated[non_assignated_in_cubic_index] = True
            # Assign voxel
            voxel_key += 1
            #
            assignated_voxel[non_assignated_in_cubic_index] = voxel_key
            count = Counter(labels_[non_assignated_in_cubic_index])
            majority_label = count.most_common(1)[0][0]

            rec_nbr_created_voxels.append(voxel_key+1)
            rec_nbr_assignated_points.append(np.sum(assignated)/N)

            # 2/2 CUBIC VOXEL FEATURES COMPUTATION

            feature_row = []
            feature_row = np.concatenate((feature_row, [voxel_key]))

            voxel_points = points[non_assignated_in_cubic_index]

            density = len(voxel_points)
            feature_row = np.concatenate((feature_row, [density]))

            voxel_center = 0.5*(np.max(voxel_points, axis=0) + np.min(voxel_points, axis=0))
            feature_row = np.concatenate((feature_row, voxel_center))

            barycenter = np.mean(voxel_points, axis=0)
            feature_row = np.concatenate((feature_row, barycenter))

            voxel_size = np.max(voxel_points, axis=0) - np.min(voxel_points, axis=0)
            feature_row = np.concatenate((feature_row, voxel_size))

            # Extract features from voxel_points
            eigenvalues, eigenvectors = local_PCA(voxel_points)
            normal = eigenvectors[:, 2]
            verticality = 2*np.arcsin(np.abs(normal@np.array([0, 0, 1])))/pi
            eps = 1e-6
            linearity = 1- eigenvalues[1] / (eigenvalues[0]+eps)
            planarity = (eigenvalues[1] - eigenvalues[2])/(eigenvalues[0]+eps)
            sphericity = eigenvalues[2]/(eigenvalues[0]+eps)

            feature_row = np.concatenate((feature_row,
                      [verticality, linearity, planarity, sphericity]))

            feature_row = np.concatenate((feature_row, [majority_label]))
            s_voxel_features = np.vstack((s_voxel_features, feature_row))
        print('Shape of s-voxel features:', s_voxel_features.shape)

        df = pd.DataFrame(s_voxel_features, columns=fields)
        df.index = df['voxel_key']
        df = df.drop(['voxel_key'], axis=1)
        df.to_csv('./features_svoxel/training/'+file[:-4]+'_{}.csv'.format(ii))