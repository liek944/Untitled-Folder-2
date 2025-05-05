"""
Clustering Module for Dengue Outbreak Detection System

This module implements various clustering algorithms to identify
potential dengue outbreak clusters based on symptom data.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.exceptions import NotFittedError


def perform_dbscan(data, features, eps=0.5, min_samples=5):
    """
    Perform DBSCAN clustering
    
    Args:
        data (pandas.DataFrame): Input data
        features (list): List of feature columns to use for clustering
        eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood
        min_samples (int): The minimum number of samples in a neighborhood for a point to be considered as a core point
        
    Returns:
        pandas.DataFrame: Data with cluster labels
    """
    # Create a copy of the data
    result = data.copy()
    
    # Extract features for clustering
    X = result[features].values
    
    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    result['cluster'] = dbscan.fit_predict(X)
    
    return result


def perform_kmeans(data, features, n_clusters=5):
    """
    Perform K-means clustering
    
    Args:
        data (pandas.DataFrame): Input data
        features (list): List of feature columns to use for clustering
        n_clusters (int): Number of clusters
        
    Returns:
        pandas.DataFrame: Data with cluster labels
    """
    # Create a copy of the data
    result = data.copy()
    
    # Extract features for clustering
    X = result[features].values
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    result['cluster'] = kmeans.fit_predict(X)
    
    return result


def perform_hierarchical_clustering(data, features, n_clusters=5, linkage='ward'):
    """
    Perform hierarchical clustering
    
    Args:
        data (pandas.DataFrame): Input data
        features (list): List of feature columns to use for clustering
        n_clusters (int): Number of clusters
        linkage (str): Linkage criterion
        
    Returns:
        pandas.DataFrame: Data with cluster labels
    """
    # Create a copy of the data
    result = data.copy()
    
    # Extract features for clustering
    X = result[features].values
    
    # Perform hierarchical clustering
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    result['cluster'] = hc.fit_predict(X)
    
    return result


def apply_kmeans_clustering(feature_matrix, n_clusters, random_state=None):
    """
    Apply K-Means clustering to symptom feature vectors
    
    This function applies K-Means clustering to binary symptom vectors to identify
    groups of cases with similar symptom patterns. It handles both pandas DataFrames
    and numpy arrays as input.
    
    Args:
        feature_matrix (numpy.ndarray or pandas.DataFrame): Binary symptom vectors
            where rows are cases and columns are symptoms (1=present, 0=absent)
        n_clusters (int): Number of clusters (K) to form
        random_state (int, optional): Random seed for reproducibility. Defaults to None for random results.
        
    Returns:
        tuple: (
            numpy.ndarray: Cluster labels for each case,
            sklearn.cluster.KMeans: Fitted KMeans model object
        )
        
    Raises:
        ValueError: If feature_matrix is empty or contains invalid values
    """
    # Input validation
    if isinstance(feature_matrix, pd.DataFrame):
        # Convert DataFrame to numpy array for clustering
        X = feature_matrix.values
    elif isinstance(feature_matrix, np.ndarray):
        X = feature_matrix
    else:
        raise ValueError("feature_matrix must be a numpy array or pandas DataFrame")
    
    # Check if the feature matrix is empty
    if X.size == 0:
        raise ValueError("Empty feature matrix provided")
    
    # Check for NaN values
    if np.isnan(X).any():
        raise ValueError("Feature matrix contains NaN values")
    
    # Check if n_clusters is valid
    if n_clusters < 2 or n_clusters >= X.shape[0]:
        raise ValueError(f"Invalid n_clusters value: {n_clusters}. Must be between 2 and {X.shape[0]-1}")
    
    try:
        # Initialize and fit KMeans model
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init='auto'  # Use 'auto' for newer scikit-learn versions
        )
        
        # Fit the model and get cluster labels
        labels = kmeans.fit_predict(X)
        
        return labels, kmeans
    
    except Exception as e:
        raise RuntimeError(f"Error during K-Means clustering: {str(e)}")


def get_cluster_centroids(kmeans_model, feature_names=None):
    """
    Get the centroids of clusters from a fitted KMeans model
    
    Args:
        kmeans_model: Fitted KMeans model object
        feature_names (list, optional): Names of features corresponding to columns in the feature matrix
        
    Returns:
        pandas.DataFrame: DataFrame where rows are clusters and columns are features,
                         values represent the centroid coordinates
    """
    try:
        # Check if the model is fitted
        centroids = kmeans_model.cluster_centers_
    except (AttributeError, NotFittedError):
        raise ValueError("The KMeans model is not fitted yet")
    
    # Convert centroids to DataFrame with feature names as columns
    if feature_names is not None:
        if len(feature_names) != centroids.shape[1]:
            raise ValueError(f"Length of feature_names ({len(feature_names)}) " 
                           f"does not match number of features ({centroids.shape[1]})")
        
        centroid_df = pd.DataFrame(centroids, columns=feature_names)
    else:
        centroid_df = pd.DataFrame(centroids)
    
    # Add cluster ID as index
    centroid_df.index.name = 'Cluster'
    centroid_df.index = [f"Cluster {i}" for i in range(len(centroid_df))]
    
    return centroid_df


def add_cluster_labels_to_data(data, labels, cluster_col_name='Cluster'):
    """
    Add cluster labels to the original data
    
    Args:
        data (pandas.DataFrame): Original data
        labels (numpy.ndarray): Cluster labels from clustering algorithm
        cluster_col_name (str, optional): Name for the cluster column. Defaults to 'Cluster'.
        
    Returns:
        pandas.DataFrame: Original data with cluster labels added
    """
    # Validate inputs
    if len(data) != len(labels):
        raise ValueError(f"Length of data ({len(data)}) does not match length of labels ({len(labels)})")
    
    # Create a copy of the data
    result = data.copy()
    
    # Add cluster labels
    result[cluster_col_name] = labels
    
    return result
