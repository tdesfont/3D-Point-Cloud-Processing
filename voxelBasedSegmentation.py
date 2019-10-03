#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Voxel Based Segmentation
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

import sys

#%%

path = './data_wo_ground/training/'
file = 'MiniLille1_wo_ground.ply'

training_path = path + file
print(training_path)

# Retrieve a slice the point cloud wo ground
cloud_ply = read_ply(training_path)
points = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T
labels = cloud_ply['class']

#%% Compute SVD on the point cloud

n = points.shape[0]
d = points.shape[1]
barycenter = np.mean(points, axis=0)
Q = points - barycenter
cov_mat = (1/n)*Q.T@Q
assert cov_mat.shape == (d, d)
eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)

#%% Slice the point cloud

# HPO
bucket_size = 20
bucket_residual = 3

# Projection on the principal axis
scalar_product = (points-barycenter)@eigenvectors[:, 2]
hash_index = scalar_product//bucket_size
chunck_ids = np.unique(hash_index)
print('Slices identifiers: ', chunck_ids)

selected_id = int(sys.argv[1])
upper_bound = (selected_id+1)*bucket_size
lower_bound = selected_id*bucket_size

interest_indexes = np.where(hash_index==selected_id)[0]
fuzzy_indexes = np.where((scalar_product <= upper_bound+bucket_residual)*\
         (scalar_product >= lower_bound-bucket_residual))[0]

# Store slice and display using CloudCompare
write_ply('./segmentation_results/slice_{}.ply'.format(selected_id),
          [points[interest_indexes], labels[interest_indexes]],
          ['x', 'y', 'z', 'class'])

#%% Restrict the points

points = points[interest_indexes]
labels = labels[interest_indexes]

training_features = np.empty((0, 4))
training_labels = np.empty((0, ))

#%% 3.1 Voxelisation of the data

N = points.shape[0]
indexes = np.arange(N)

t0 = time.time()
print('Building KDTree...')
kd = KDTree(points, metric='minkowski')
t1 = time.time()
print('KDTree built in {} sec'.format(t1-t0))

#%% Single-Shot query for cubical voxels

radius=0.1

t0 = time.time()
neighborhoods_inner_sphere = kd.query_radius(points, r=radius)
t1 = time.time()
print('Query time for computing neighborhoods on all points: {} sec'.format(t1-t0))

t0 = time.time()
neighborhoods_outer_sphere = kd.query_radius(points, r=radius*sqrt(3))
t1 = time.time()
print('Query time for computing neighborhoods on all points: {} sec'.format(t1-t0))

#%% Cubic Voxel Assignation without replacement

N = points.shape[0]
assert neighborhoods_inner_sphere.shape[0] == N
assert neighborhoods_outer_sphere.shape[0] == N

assignated = np.array([False for i in range(N)])
assignated_voxel = np.array([-1 for i in range(N)])

rec_nbr_created_voxels = []
rec_nbr_assignated_points = []

s_voxel_features = np.empty((0, 15))

voxel_key = -1
for index in tqdm.tqdm(range(N)):

    if assignated[index]:
        continue

    # 1/2 CUBIC VOXEL ASSIGNATION

    inner_index = neighborhoods_inner_sphere[index]
    outer_index = neighborhoods_outer_sphere[index]

    # Compute the points in the cubic voxel
    in_cubic = np.prod(points[outer_index] <= np.max(points[inner_index], axis=0), axis=1)*\
               np.prod(points[outer_index] >= np.min(points[inner_index], axis=0), axis=1)

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

    s_voxel_features = np.vstack((s_voxel_features, feature_row))

print('Number of created s-voxels:', voxel_key+1)
print('Should be equal to 1:', np.mean(assignated))

#%%

fields = ['voxel_key',
          'density',
          'Vx', 'Vy', 'Vz',
          'Bx', 'By', 'Bz',
          'Sx', 'Sy', 'Sz',
          'verticality', 'linearity', 'planarity', 'sphericity']

df = pd.DataFrame(s_voxel_features, columns=fields)
df.index = df['voxel_key']
df = df.drop(['voxel_key'], axis=1)

#%%

V = df[['Vx', 'Vy', 'Vz']].as_matrix()
B = df[['Bx', 'By', 'Bz']].as_matrix()
S = df[['Sx', 'Sy', 'Sz']].as_matrix()

#%% Union-Find datastructure

# https://github.com/jilljenn/tryalgo

class UnionFind:

    def __init__(self, n):
        self.up = list(range(n))
        self.rank = [0]*n

    def find(self, x):
        if self.up[x] == x:
            return x
        else:
            self.up[x] = self.find(self.up[x])
            return self.up[x]

    def union(self, x, y):
        repr_x = self.find(x)
        repr_y = self.find(y)
        if repr_x == repr_y:
            return False
        if self.rank[repr_x] == self.rank[repr_y]:
            self.rank[repr_x] += 1
            self.up[repr_y] = repr_x
        elif self.rank[repr_x] > self.rank[repr_y]:
            self.up[repr_y] = repr_x
        else:
            self.up[repr_x] = repr_y
        return True

#%% 3.3 Clustering by Link Chain Method

uf = UnionFind(V.shape[0])

# Build a kdtree on the voxel centers
t0 = time.time()
print('Building KDTree...')
kd = KDTree(V, metric='chebyshev')
t1 = time.time()
print('KDTree built in {} sec'.format(t1-t0))

# Define the inter voxel distance
c_d = 0.05*np.array([1, 1, 1])

# Query by majoration
neighborhoods = kd.query_radius(V, r=radius+np.max(c_d))

for index in range(V.shape[0]):

    # Compute linkage bool
    inter_voxel_abs_dist = np.abs(V[index]-V[neighborhoods[index]])
    size_term = 0.5*(S[index]+S[neighborhoods[index]])
    cells_term = c_d
    # Selection of the secondary-links
    is_linked = np.prod(inter_voxel_abs_dist < size_term + cells_term, axis=1)
    secondary_links = neighborhoods[index][np.where(is_linked)]

    for secondary_voxel in secondary_links:
        uf.union(index, secondary_voxel)

#%%

from collections import Counter
count = Counter(uf.up)

segmented_objects = []
for point_index in range(points.shape[0]):
    segment_id = uf.up[assignated_voxel[point_index]]
    segmented_objects.append(segment_id)

segmented_objects = np.array(segmented_objects).astype('float')

high_confidence_index = []
ordered_count = count.most_common()
min_nbr_point_per_voxel = 20 # HPO

# initialize
segment_id, nbr_voxel = ordered_count[0]
i = 0

while nbr_voxel >= min_nbr_point_per_voxel:
    #
    point_indexes = np.where(segmented_objects==segment_id)[0]
    high_confidence_index += list(point_indexes)
    #
    i+=1
    segment_id, nbr_voxel = ordered_count[i]
print(i)

# remaining indexes are those with low confidences
low_confidence_index = set(np.arange(points.shape[0])).difference(set(high_confidence_index))
low_confidence_index = np.sort(list(low_confidence_index))

high_conf_points = points[high_confidence_index]
labels_high_conf = segmented_objects[high_confidence_index]

low_conf_points = points[low_confidence_index]

# Build KD-Tree on Known Labelled Points only

t0 = time.time()
print('Building KDTree...')
kd = KDTree(high_conf_points, metric='minkowski')
t1 = time.time()
print('KDTree built in {} sec'.format(t1-t0))

nn1 = kd.query(low_conf_points, k=1, return_distance=False)

labels_low_conf = labels_high_conf[nn1.ravel()]

output_points = np.vstack((high_conf_points, low_conf_points))
output_seg_labels = np.hstack((labels_high_conf, labels_low_conf))

assert output_points.shape[0] == len(output_seg_labels)

#%% Visualize the output point

color_dict = {}
for segment_id in np.unique(output_seg_labels):
    r, g, b = np.random.random(3)
    color_dict[segment_id] = [r, g, b]

segmentation_color = np.zeros((len(output_seg_labels), 3))
for segment_key in color_dict:
    segmentation_color[np.where(output_seg_labels == segment_key)[0]] = color_dict[segment_key]

write_ply('./segmentation_results/segmentation_slice_{}.ply'.format(selected_id),
          [output_points, segmentation_color],
          ['x', 'y', 'z', 'Red', 'Green', 'Blue'])
