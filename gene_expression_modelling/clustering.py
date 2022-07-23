# TODO(mmd): Update this comment post changes.
# Ideas and code copied liberally from http://scikit-learn.org/stable/modules/clustering.html and linked
# examples.
#
# Skipped:
#   * Spectral Clustering
#   * Birch
#
# TODO(mmd):
#   * Variational Bayes Gaussian Mixture Models
#   * BIC Selection Criteria for Cluster Selection
#   * Try other distance metrics
#   * Subspace Methods (see SciKit Learn Manifold Learning Modules)

import numpy, pandas as pd

#from sklearn.preprocessing import scale
from sklearn import cluster, mixture#, metrics, decomposition

# To avoid can't find name error in tracebacks.
## Constants:


def get_labels(clusterer, data):
    if hasattr(clusterer, 'labels_'): return clusterer.labels_
    elif hasattr(clusterer, 'weights_'): return clusterer.predict(data)
    else: raise

def fit_and_label(clusterer, df):
    clusterer.fit(df.values)
    return clusterer, pd.DataFrame(get_labels(clusterer, df.values), index=df.index)

def k_means(df, n_clusters=3, init='k-means++', n_init=10):
    return fit_and_label(cluster.KMeans(init=init, n_clusters=n_clusters, n_init=n_init), df)

def gmm(df, n_clusters=3, covariance_type='full'):
    clusterer = mixture.GaussianMixture(n_components=n_clusters, covariance_type=covariance_type)
    return fit_and_label(clusterer, df)

def gmm_bayes(df, max_n_clusters=10, covariance_type='full', means_init=None):
    clusterer = mixture.BayesianGaussianMixture(n_components=max_n_clusters, covariance_type=covariance_type)
    return fit_and_label(clusterer, df)

def agglometrative(df, n_clusters=4, linkage='average'):
    return fit_and_label(cluster.AgglomerativeClustering(linkage=linkage, n_clusters=n_clusters), df)

def affinity_propagation(df): return fit_and_label(cluster.AffinityPropagation(), df)
def mean_shift(df): return fit_and_label(cluster.MeanShift(), df)
def dbscan(df): return fit_and_label(cluster.DBSCAN(eps=5), df)
