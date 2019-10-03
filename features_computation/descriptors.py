#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NPM3D Project:
Compute the descriptors for the points cloud.
Reminder:
    Evaluation on a spherical radius, can be on a voxel instead
"""

# Imports
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KDTree
from utils.ply import write_ply, read_ply
import time
from math import pi
import tqdm

def local_PCA(points):
    n = points.shape[0]
    d = points.shape[1]
    barycenter = np.mean(points, axis=0)
    Q = points - barycenter
    cov_mat = (1/n)*Q.T@Q
    assert cov_mat.shape == (d, d)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)
    return eigenvalues, eigenvectors

def neighborhood_PCA(query_points, cloud_points, radius):
    """
        Compute the normals and eigenvalues on a point cloud
    """
    # vectors initialisation
    all_eigenvalues = np.zeros((query_points.shape[0], 3))
    all_eigenvectors = np.zeros((query_points.shape[0], 3, 3))
    # data structure for neighborhood queries
    print('Compute kd tree...')
    kd_tree = KDTree(cloud_points, leaf_size=8, metric='minkowski')
    # compute neighborhoods at once
    print('Compute query...')
    neighborhoods = kd_tree.query_radius(query_points, r=radius)
    # compute eigenvalues and eigenvectors
    print('Compute eigenvalues and eigenvectors...')
    for i in tqdm.tqdm(range(len(neighborhoods))):
        neighborhood = neighborhoods[i]
        eigenvalues, eigenvectors = local_PCA(cloud_points[neighborhood])
        all_eigenvalues[i] = eigenvalues[::-1]
        all_eigenvectors[i] = eigenvectors[:, ::-1]
    return all_eigenvalues, all_eigenvectors

def compute_features(query_points, cloud_points, radius):
    # compute eigenvalues, eigenvectors and normals
    all_eigenvalues, all_eigenvectors = neighborhood_PCA(query_points, cloud_points, radius)
    print('Compute features from eigenvectors and eigenvalues...')
    normals = all_eigenvectors[:, :, 2]
    # compute features from eigenvectors
    verticality = 2*np.arcsin(np.abs(normals@np.array([0, 0, 1])))/pi
    # compute dimensional features
    eps = 1e-6
    linearity = 1- all_eigenvalues[:, 1] / (all_eigenvalues[:, 0]+eps)
    planarity = (all_eigenvalues[:, 1] - all_eigenvalues[:, 2])/(all_eigenvalues[:, 0]+eps)
    sphericity = all_eigenvalues[:, 2]/(all_eigenvalues[:, 0]+eps)
    return verticality, linearity, planarity, sphericity

