# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 15:34:05 2019

@author: NUS
"""
import numpy as np 
#####################################################K-means clustering###########################################
# randomly select the centroids
def randCent(data,k):
    """random gengerate the centroids
    parameters
    ------------
    data: <class 'numpy.ndarray'>, shape=[n_samples, n_features], input data to be randomly select centorids.
            
    k:    <class 'int'>   the number of the centroids
    ------------
    return
        centroids: <class 'numpy.ndarray'>, shape=[k, n_features]
    """
    n_samples = data.shape[0]
    n_features = data.shape[1]
    centroids = np.zeros((k, n_features))
    rand_range = n_samples - 1
    for i in range(k):
        random_index = random.randint(0, rand_range)
        centroids[i] = data[random_index]
    return centroids

# assign points to centroids
def assign(data, k, centroids):
    """ return assignment:  <class 'numpy.matrix'>, shape=[n_samples, 1]"""
    n_samples = data.shape[0]
    n_features = data.shape[1]
    assignments = np.zeros((n_samples, 1))

    # Loop through every data point
    for i in range(n_samples):

        # For every centroid, calculate distance
        distances = np.zeros(k)
        for j in range(k):
            # Add up for each feature
            for m in range(n_features):
                distances[j] += pow(data[i][m] - centroids[j][m], 2)
            # no need to divide since we are looking for the min

        min_dist = min(distances)
        for j in range(k):
            if distances[j] == min_dist:
                assignments[i] = j
                break

    return assignments

def KMeans(data,k):
    """ KMeans algorithm 
    parameters
    ------------
    data: <class 'numpy.ndarray'>, shape=[n_samples, n_features], input data to be randomly select centorids.
            
    k:    <class 'int'>   the number of the centroids
    ------------
    return
        centroids: <class 'numpy.ndarray'>, shape=[k, n_features]
        clusterAssment:  <class 'numpy.matrix'>, shape=[n_samples, 1]
    """
    centroids = randCent(data, k)
    n_samples = data.shape[0]
    n_features = data.shape[1]

    not_converged = True
    while(not_converged):
        clusterAssment = assign(data, k, centroids)

        not_converged = False

        # For each cluster, recalculate centroid
        for i in range(k):
            new_centroid = np.zeros(n_features)
            for j in range(n_features):
                new_feature = 0
                counter = 0
                for m in range(n_samples):
                    if clusterAssment[m] == i:
                        new_feature += data[m][j]
                        counter += 1
                if counter > 0:
                    new_feature /= counter
                new_centroid[j] = new_feature
            if not np.array_equal(new_centroid, centroids[i]):
                not_converged = True
            centroids[i] = new_centroid

    return centroids, clusterAssment


##############################################color #############################################################
import random
def colors(k):
    """ generate the color for the plt.scatter
    parameters
    ------------
    k:    <class 'int'>   the number of the centroids
    ------------
    return
        ret: <class 'list'>, len = k
    """    
    ret = []
    for i in range(k):
        ret.append((random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))
    return ret


############################################mean shift clustering##############################################
from collections import defaultdict
import warnings
from sklearn.neighbors import NearestNeighbors
from sklearn.utils._joblib import Parallel
from sklearn.utils._joblib import delayed
 
def _mean_shift_single_seed(my_mean, X, nbrs, max_iter):
    """mean shift cluster for single seed.
    Parameters
    ----------
    X : array-like, shape=[n_samples, n_features]
        Samples to cluster.
    nbrs: NearestNeighbors(radius=bandwidth, n_jobs=1).fit(X)
    max_iter: max interations 
    return:
        mean(center) and the total number of pixels which is in the sphere
    """
    # For each seed, climb gradient until convergence or max_iter
    n_features = X.shape[1]
    is_converged = False
    my_mean = seed_to_array(my_mean, n_features)

    for i in range(max_iter):
        if is_converged:
            break

        # Find neighbors
        neighbors = nbrs.radius_neighbors(my_mean, return_distance=False)[0]

        # Calculate centre of gravity
        new_mean = np.zeros((1, n_features))
        for neighbor in neighbors:
            new_mean += X[neighbor]
        new_mean /= len(neighbors)

        if np.array_equal(my_mean, new_mean):
            is_converged = True

        else:
            my_mean = new_mean

    neighbors = nbrs.radius_neighbors(my_mean, return_distance=False)[0]

    return my_mean, len(neighbors)

# Since the seed has tuple as key, we need to convert back
def seed_to_array(seed, n_features):
    seed_array = np.zeros((1, n_features))
    i = 0
    for num in seed:
        seed_array[0][i] = num
        i += 1
    return seed_array

def mean_shift(X, bandwidth=None, seeds=None, bin_seeding=False,min_bin_freq=1, cluster_all=True, max_iter=300,
               n_jobs=None):
    """pipline of mean shift clustering
    Parameters
    ----------
    X : array-like, shape=[n_samples, n_features]
    bandwidth: the radius of the sphere
    seeds: whether use the bin seed algorithm to generate the initial seeds
    bin_size:    bin_size = bandwidth.
    min_bin_freq: for each bin_seed, the minimize of the points should cover
    return:
        cluster_centers <class 'numpy.ndarray'> shape=[n_cluster, n_features] ,labels <class 'list'>, len = n_samples
    """
    n_samples = X.shape[0]
    n_features = X.shape[1]

    if bin_seeding:
        seeds = get_bin_seeds(X, bandwidth)

    # find the points within the sphere
    nbrs = NearestNeighbors(radius=bandwidth, n_jobs=1).fit(X)

    ##########################################parallel computing############################
    center_intensity_dict = {}
    all_res = Parallel(n_jobs=n_jobs)(
        delayed(_mean_shift_single_seed)
        (seed, X, nbrs, max_iter) for seed in seeds)#
    ##########################################parallel computing############################

    cluster_centers = np.zeros((len(all_res), n_features))
    for i in range(len(all_res)):
        cluster_centers[i] = np.asarray(all_res[i][0])

    labels = [0] * n_samples
    neighborhoods = nbrs.radius_neighbors(cluster_centers, return_distance=False)
    for i in range(len(neighborhoods)):
        for neighbor in neighborhoods[i]:
            labels[neighbor] = i

    # Replace the original label numbers with unique indices starting from 0
    i = 0
    unique_labels_map = {}
    unique_labels = np.unique(labels)
    for unique_label in unique_labels:
        unique_labels_map[unique_label] = i
        i += 1
    for i in range(len(labels)):
        labels[i] = unique_labels_map[labels[i]]

    return cluster_centers, labels

def get_bin_seeds(X, bin_size, min_bin_freq=1):
    """generate the initial seeds, in order to use the parallel computing 
    Parameters
    ----------
    X : array-like, shape=[n_samples, n_features]
    bin_size:    bin_size = bandwidth.
    min_bin_freq: for each bin_seed, the minimize of the points should cover
    return:
        bin_seeds: dict-like bin_seeds = {key=seed, key_value=he total number of pixels which is in the sphere }
    """
    # Bin points
    n_features = X.shape[1]

    X_compressed = np.round(X/bin_size)

    bin_seeds = {}
    for sample in X_compressed:
        tuple_sample = tuple(sample)
        if tuple_sample in bin_seeds:
            bin_seeds[tuple_sample] += 1
        else:
            bin_seeds[tuple_sample] = 0

    bin_seeds_resized = {}

    for key in bin_seeds.keys():
        new_key = np.zeros(n_features)
        value = bin_seeds.get(key)
        for i in range(n_features):
            new_key[i] = key[i] * bin_size
        new_key_tuple = tuple(new_key)
        bin_seeds_resized[new_key_tuple] = value

    return bin_seeds_resized
