#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Due to the method based on a link-chain, connected-components are likely
    to be connected. However, in our type of data, the ground connects
    every object.

    Therefore, we train a classifier on one of the training dataset, and
    then store it as a model to be reusable.

    The ground classifier is a two-class classification problem solved
    with random forest on the features obtained by local PCA.
"""

#%% Imports

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from utils.ply import write_ply, read_ply

import time

import pickle

import os
from os.path import exists

from features_computation.descriptors import local_PCA, compute_features


#%% Training data for the ground classifier

training_path = 'data_points_cloud/training/MiniLille1.ply'

#%% Choose radius

radius = 0.5 # HPO

num_training_data = 20000 # HPO
num_per_class = num_training_data//2 // 5

label_names= {0: 'Unclassified',
              1: 'Ground',
              2: 'Building',
              3: 'Poles',
              4: 'Pedestrians',
              5: 'Cars',
              6: 'Vegetation'}

#%% Build training data

training_features = np.empty((0, 4))
training_labels = np.empty((0, ))

cloud_ply = read_ply(training_path)
points = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T
labels = cloud_ply['class']

# Initiate training indices array
training_inds = np.empty(0, dtype=np.int32)

for label, name in label_names.items():
    if label==0 or label==1:
        continue
    label_inds = np.where(labels==label)[0]
    if len(label_inds) <= num_per_class:
        print('Data harvesting on class: {}'.format(name))
        training_inds = np.hstack((training_inds, label_inds))
    else:
        random_choice = np.random.choice(len(label_inds), num_per_class, replace=False)
        training_inds = np.hstack((training_inds, label_inds[random_choice]))

# Build ground training data
print(training_inds.shape)
N = training_inds.shape[0]

label=1
name='Ground'

label_inds = np.where(labels==label)[0]
random_choice = np.random.choice(len(label_inds), N, replace=False)
training_inds = np.hstack((training_inds, label_inds[random_choice]))

print('Final shape of training data: {}'.format(training_inds.shape[0]))

# Gather chosen points
training_points = points[training_inds, :]

#%% Compute features

t0 = time.time()
vert, line, plan, sphe = compute_features(training_points, points, radius)
features = np.vstack((vert.ravel(), line.ravel(), plan.ravel(), sphe.ravel())).T
# Concatenate features / labels of all clouds
training_features = np.vstack((training_features, features))
training_labels = np.hstack((training_labels, labels[training_inds]))
# transform into two class classification problem
training_labels[training_labels!=1] = 0

t1 = time.time()

#%% Classifier and validation data

X = training_features
y = training_labels

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

y_predict = clf.predict(X_val)

print('Ground classifier performance report:')
print(classification_report(y_true=y_val, y_pred=y_predict))

#%% Store the classifier for later use

pickle.dump( clf, open( "models/ground_clf.p", "wb" ) )

#%% Apply classifier on available feature file of point clouds

ground_clf = pickle.load( open( "models/ground_clf.p", "rb" ) )

#%% Training features

available_features = os.listdir('./features/training/')

for file_name in available_features:

    feature_file  = './features/training/'+file_name
    features = np.load(feature_file)

    pc_name = feature_file.split('/')[-1]
    pc_name = pc_name[:-13]
    id_name = pc_name
    pc_name = pc_name+'.ply'
    pc_name = './data_points_cloud/training/' + pc_name

    print(pc_name)

    cloud_ply = read_ply(pc_name)
    points = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T
    labels = cloud_ply['class']

    predictions = ground_clf.predict(features)

    # Select points on the learnt class 0 (predicted as not being ground)
    write_ply('./data_wo_ground/training/{}_wo_ground.ply'.format(id_name),
          [points[predictions==0], labels[predictions==0]], ['x', 'y', 'z', 'class'])

    print('./data_wo_ground/training/{}_wo_ground.ply'.format(id_name))

#%% Test files

available_features = os.listdir('./features/test/')

for file_name in available_features:

    feature_file  = './features/test/'+file_name
    features = np.load(feature_file)

    pc_name = feature_file.split('/')[-1]
    pc_name = pc_name[:-13]
    id_name = pc_name
    pc_name = pc_name+'.ply'
    pc_name = './data_points_cloud/test/' + pc_name

    print(pc_name)

    cloud_ply = read_ply(pc_name)
    points = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

    predictions = ground_clf.predict(features)

    # Select points on the learnt class 0 (predicted as not being ground)
    write_ply('./data_wo_ground/test/{}_wo_ground.ply'.format(id_name),
          [points[predictions==0]], ['x', 'y', 'z'])

    print('./data_wo_ground/test/{}_wo_ground.ply'.format(id_name))
