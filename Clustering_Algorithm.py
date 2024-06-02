# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 14:26:29 2022

@author: Xinhao Lan
"""
# built-in modules
import os
import shutil

# third-party modules
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.cluster import AgglomerativeClustering, BisectingKMeans, SpectralBiclustering
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics import silhouette_samples
from scipy.cluster.hierarchy import linkage, dendrogram

# K-means Method 
def basic_k_means(x, kmin, kmax):
    """
    Use basic k-means method to get the list of k and list of silhouette score

    :param x: numpy.ndarray. Input matrix of the feature file.
    :param kmin: int. the min k value in the list.
    :param kmax: int. the max k value in the list.

    :return kl: list. The type of each item in the list is int. This is the list of different k ranging from kmin to kmax.
    :return sil: list. The type of each item in the list is float. This is the list of different silhouette score for different k ranging from kmin to kmax.
    """
    sil = []
    kl = []
    for k in range(kmin, kmax + 1):
        kMeans = KMeans(n_clusters=k).fit(x)
        labels = kMeans.labels_ 
        sil.append(silhouette_score(x, labels, metric='euclidean')) 
        kl.append(k)
    return kl, sil

def K_means(x, k):
    """
    Use the k-means method to get the prediction of labels for each image.

    :param x: numpy.ndarray. Input matrix of the feature file.
    :param k: int. The total number of classes.

    :return kPredictions: list. The type of each item in the list is int. This includes the prediction labels of the images.
    """
    kMeansModel = KMeans(n_clusters=k, random_state = 42)
    kMeansModel.fit(x)
    kPredictions = kMeansModel.predict(x)
    return kPredictions

# Improved k-means / Spectral Clustering
def Improved_k_means(x, k):
    """
    Use the spectral clustering to get the prediction for each image.

    :param x: numpy.ndarray. Input matrix of the feature file.
    :param k: int. The total number of classes.

    :return labels: list. The type of each item in the list is int. This includes the prediction labels of the images.
    """
    model = SpectralClustering(n_clusters = k, affinity='nearest_neighbors', assign_labels='kmeans')
    labels = model.fit_predict(x)
    return labels

# Hierarchical Clustering
def Hierarchical_Clustering(x):
    """
    Use the hierarchical clustering to get the prediction tree graph for images.

    This is just the graph of predicted labels, detailed labels should be the same as AgglomerativeClustering

    :param x: numpy.ndarray. Input matrix of the feature file.
    """
    link_matrix = linkage(x,'single')
    dendrogram(link_matrix)
    
# Gaussian Mixture Model
def Gaussian_Mixture_Model(x, k):
    """
    Use the Gaussian Mixture model to get the prediction for each image.

    :param x: numpy.ndarray. Input matrix of the feature file.
    :param k: int. The total number of classes.

    :return labels: list. The type of each item in the list is int. This includes the prediction labels of the images.
    """
    gmm = GMM(n_components=k, random_state=42)
    labels = gmm.fit_predict(x)
    return labels

# DBSCAN
def DBSCAN_Clustering(x, distance, min_samples):
    """
    Use the DBSCAN clustering model to get the prediction for each image.

    :param x: numpy.ndarray. Input matrix of the feature file.
    :param distance: float. The distance threshold between two classes.
    :param min_samples: int. The min value of the instace in one class. 

    :return dbscan.labels_: list. The type of each item in the list is int. This includes the prediction labels of the images.

    """
    dbscan = DBSCAN(eps = distance, min_samples = min_samples)
    dbscan.fit(x)
    return dbscan.labels_

# Agglo Merative Clustering
def get_agglo_cluster_model(num_clusters = 2):
    """
    Use the agglo clustering model to get the prediction for each image.

    :param num_clusters: int, optional. The total number of classes. The default is 2.

    :return agglo_cluster_model: The model based on the agglo clustering.
    """
    agglo_cluster_model = AgglomerativeClustering(n_clusters=num_clusters, linkage = "single", distance_threshold = None)
    return agglo_cluster_model

# Bisect k-means
def get_bisect_k_means_model(num_clusters = 2):
    """
    Use the bisect k-means model to get the prediction for each image.

    :param num_clusters: int, optional. The total number of classes. The default is 2.

    :return bisect_k_means_model: The model based on the bisect k-means.
    """
    bisect_k_means_model = BisectingKMeans(n_clusters=num_clusters, init="k-means++", random_state=42)
    return bisect_k_means_model

# SpectralBiclustering
def get_spectral_bi_cluster_model(num_clusters = 2):
    """
    Use the spectral bisect k-means model to get the prediction for each image.

    :param num_clusters: int, optional. The total number of classes. The default is 2.

    :return spectral_bi_cluster_model: The model based on the spectral bisect k-means.
    """
    spectral_bi_cluster_model = SpectralBiclustering(n_clusters=num_clusters, random_state = 42, method = "log", svd_method="arpack")
    return spectral_bi_cluster_model


def visualize_silhouette_score(x, labels):
    """
    Draw the graph of visualized silhouette score and coefficients.
    
    :param x: numpy.ndarray. Input matrix of the feature file.
    :param labels: list. The type of each item in the list is int. This includes the prediction labels of the images.
    """
    k = len(np.unique(labels))
    silhouette_coefficients = silhouette_samples(x, labels)
    padding = len(x) // 30
    pos = padding
    ticks = []
    for i in range(k):
        coeffs = silhouette_coefficients[labels == i]
        coeffs.sort()
        color = mpl.cm.Spectral(i / k)
        plt.fill_betweenx(np.arange(pos, pos + len(coeffs)), 0, coeffs, facecolor=color, edgecolor=color, alpha=0.7)
        ticks.append(pos + len(coeffs) // 2)
        pos += len(coeffs) + padding	
    
    plt.gca().yaxis.set_major_locator(FixedLocator(ticks))
    plt.gca().yaxis.set_major_formatter(FixedFormatter(range(k)))
    plt.ylabel("Cluster")
    plt.gca().set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.xlabel("Silhouette Coefficient") 
    plt.axvline(x=silhouette_score(x, labels, metric='euclidean'), color="red", linestyle="--")
    plt.title("$k={}$".format(k), fontsize=16)

def k_and_score(data, range_of_k=(2,15), metric = 'euclidean'):
    """
    Function to test all the clustering algorithm and get the silhouette score.

    :param data: numpy.ndarray. Input matrix of the feature file.
    :param range_of_k: tuple, optional. The min classes number and max classes number. The default is (2,15).
    """
    k_list = []
    score_list = []
    k_list, score_list = basic_k_means(data, range_of_k[0], range_of_k[1])
    print('Basic k-means:')
    print(k_list)
    print(score_list)
    
    k_list = []
    score_list = []
    for num_clusters in range (range_of_k[0], range_of_k[1] + 1):
        labels = Improved_k_means(data, num_clusters)
        score = silhouette_score(data, labels, metric = metric)
        k_list.append(num_clusters)
        score_list.append(score)
    print('Improved k-means:')
    print(k_list)
    print(score_list)
    
    k_list = []
    score_list = []
    for num_clusters in range (range_of_k[0], range_of_k[1] + 1):
        labels = Gaussian_Mixture_Model(data, num_clusters)
        score = silhouette_score(data, labels, metric = metric)
        k_list.append(num_clusters)
        score_list.append(score)
    print('Gaussian Mixture Model:')
    print(k_list)
    print(score_list)
    """
    print('DBSCAN:')
    k_list = []
    score_list = []
    labels = DBSCAN_Clustering(data, 15, 3)
    score = silhouette_score(data, labels, metric='euclidean')
    num_clusters = len(set(labels))
    print(num_clusters)
    print(score)
    """
    #Hierarchical_Clustering(data)
    k_list = []
    score_list = []
    for num_clusters in range (range_of_k[0], range_of_k[1] + 1):
        model = get_agglo_cluster_model(num_clusters)
        model.fit(data)
        labels = model.labels_
        score = silhouette_score(data, labels, metric = metric)
        k_list.append(num_clusters)
        score_list.append(score)
    print('agglo_cluster_model:')
    print(k_list)
    print(score_list)
    
    k_list = []
    score_list = []
    for num_clusters in range (range_of_k[0], range_of_k[1] + 1):
        model = get_bisect_k_means_model(num_clusters)
        model.fit(data)
        labels = model.labels_
        score = silhouette_score(data, labels, metric = metric)
        k_list.append(num_clusters)
        score_list.append(score)
    print('bisect_k_means_model:')
    print(k_list)
    print(score_list)
    
    k_list = []
    score_list = []
    for num_clusters in range (range_of_k[0], range_of_k[1] + 1):
        model = get_spectral_bi_cluster_model(num_clusters)
        model.fit(data)
        labels = model.row_labels_
        score = silhouette_score(data, labels, metric = metric)
        k_list.append(num_clusters)
        score_list.append(score)
    print('spectral_bi_cluster_model:')
    print(k_list)
    print(score_list)
    
def classify_image(k, paths, kPredictions, output_dir, input_dir):
    """
    Function to put images into their classes' folder
    
    :param k: int. Number of classes
    :param paths : list. The type of each item in the list is string. Each one is the direcatory of the image.
    :param kPredictions : list. The type of each item in the list is string. Each one is the predicted class name.
    :param output_dir : string. Output direcatory string for the classified images.
    """
    for i in range(1,k+1):
        name = input_dir + "/class" + str(i)
        if os.path.isdir(name):
            shutil.rmtree(name)
        os.mkdir(output_dir + "/class" + str(i))
    for i in range(len(paths)):
        for j in range(0,k):
            if kPredictions[i] == j:
                shutil.copy(paths[i], output_dir + "/class" + str(j+1))
    

data = np.loadtxt('D:/SteelImage/Final/Transformer_feature') # load your feature
range_of_k=(2,10)
k_and_score(data, range_of_k, "cosine")

