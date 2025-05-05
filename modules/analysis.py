"""
Analysis Module for Dengue Outbreak Detection System

This module provides functions to analyze clustering results and
identify potential dengue outbreak patterns.
"""

import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter


def identify_anomalous_clusters(data, cluster_col='cluster', threshold=2.0):
    """
    Identify clusters that may represent anomalous patterns (potential outbreaks)
    
    Args:
        data (pandas.DataFrame): Clustered data
        cluster_col (str): Column name containing cluster labels
        threshold (float): Z-score threshold for considering a cluster anomalous
        
    Returns:
        list: List of anomalous cluster IDs
    """
    # Get cluster statistics
    cluster_stats = data.groupby(cluster_col).size()
    
    # Calculate z-scores for cluster sizes
    z_scores = stats.zscore(cluster_stats)
    
    # Identify anomalous clusters (those with z-scores above threshold)
    anomalous_clusters = cluster_stats[abs(z_scores) > threshold].index.tolist()
    
    return anomalous_clusters


def analyze_cluster_characteristics(data, cluster_col='cluster', feature_cols=None):
    """
    Analyze the characteristics of each cluster
    
    Args:
        data (pandas.DataFrame): Clustered data
        cluster_col (str): Column name containing cluster labels
        feature_cols (list): Columns to analyze (if None, use all numeric columns)
        
    Returns:
        pandas.DataFrame: Cluster characteristics summary
    """
    if feature_cols is None:
        feature_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        # Remove cluster column if it's in the feature columns
        if cluster_col in feature_cols:
            feature_cols.remove(cluster_col)
    
    # Calculate mean values for each feature by cluster
    cluster_means = data.groupby(cluster_col)[feature_cols].mean()
    
    # Calculate standard deviations for each feature by cluster
    cluster_stds = data.groupby(cluster_col)[feature_cols].std()
    
    # Calculate sizes of each cluster
    cluster_sizes = data.groupby(cluster_col).size().rename('size')
    
    # Combine statistics into a single DataFrame
    cluster_summary = pd.concat([cluster_means, cluster_stds.add_suffix('_std'), 
                               cluster_sizes], axis=1)
    
    return cluster_summary


def calculate_temporal_trends(data, cluster_col='cluster', date_col='date'):
    """
    Calculate temporal trends for each cluster
    
    Args:
        data (pandas.DataFrame): Clustered data
        cluster_col (str): Column name containing cluster labels
        date_col (str): Column name containing dates
        
    Returns:
        pandas.DataFrame: Temporal trends by cluster
    """
    # Ensure date column is datetime type
    data = data.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    
    # Extract date components
    data['year'] = data[date_col].dt.year
    data['month'] = data[date_col].dt.month
    data['week'] = data[date_col].dt.isocalendar().week
    
    # Group by cluster and time periods
    weekly_trends = data.groupby([cluster_col, 'year', 'week']).size().unstack(level=0).fillna(0)
    monthly_trends = data.groupby([cluster_col, 'year', 'month']).size().unstack(level=0).fillna(0)
    
    return {
        'weekly': weekly_trends,
        'monthly': monthly_trends
    }


def profile_clusters(feature_matrix, labels, all_symptoms_list, as_percentage=True):
    """
    Analyze the characteristics of each cluster found by K-Means
    
    This function creates a profile of each cluster by calculating the prevalence
    of each symptom within the cluster. It helps identify which symptoms are
    dominant in each cluster, which is crucial for interpreting the clustering
    results in the context of dengue outbreak detection.
    
    Args:
        feature_matrix (numpy.ndarray or pandas.DataFrame): Binary symptom vectors
            where rows are cases and columns are symptoms (1=present, 0=absent)
        labels (numpy.ndarray): Cluster labels assigned by K-Means
        all_symptoms_list (list): List of all possible symptoms corresponding to
            the columns of feature_matrix
        as_percentage (bool, optional): If True, show symptom prevalence as percentages,
            otherwise show counts. Defaults to True.
        
    Returns:
        pandas.DataFrame: DataFrame where each row represents a cluster and columns show:
            - Cluster: Cluster ID
            - Size: Number of members (cases) in the cluster
            - Symptom columns: Prevalence (percentage or count) of each symptom
            
    Raises:
        ValueError: If inputs have inconsistent dimensions or are invalid
    """
    # Input validation
    if isinstance(feature_matrix, pd.DataFrame):
        # If feature_matrix is a DataFrame, ensure columns match all_symptoms_list
        if not all(col in feature_matrix.columns for col in all_symptoms_list):
            missing = [s for s in all_symptoms_list if s not in feature_matrix.columns]
            raise ValueError(f"The following symptoms are missing from feature_matrix columns: {missing}")
        
        # Extract the values for the specified symptoms in the correct order
        X = feature_matrix[all_symptoms_list].values
    elif isinstance(feature_matrix, np.ndarray):
        # If feature_matrix is a numpy array, ensure dimensions match
        if feature_matrix.shape[1] != len(all_symptoms_list):
            raise ValueError(f"feature_matrix has {feature_matrix.shape[1]} columns, " 
                           f"but all_symptoms_list has {len(all_symptoms_list)} items")
        X = feature_matrix
    else:
        raise ValueError("feature_matrix must be a numpy array or pandas DataFrame")
    
    # Check if labels and feature_matrix have the same number of rows
    if len(labels) != X.shape[0]:
        raise ValueError(f"labels has {len(labels)} items, but feature_matrix has {X.shape[0]} rows")
    
    # Get unique cluster labels
    unique_clusters = sorted(set(labels))
    
    # Initialize results dictionary
    cluster_profiles = {
        'Cluster': [],
        'Size': []
    }
    
    # Add columns for each symptom
    for symptom in all_symptoms_list:
        if as_percentage:
            cluster_profiles[f"{symptom} %"] = []
        else:
            cluster_profiles[symptom] = []
    
    # Calculate profiles for each cluster
    for cluster_id in unique_clusters:
        # Get indices of cases in this cluster
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_size = len(cluster_indices)
        
        # Add basic cluster info
        cluster_profiles['Cluster'].append(cluster_id)
        cluster_profiles['Size'].append(cluster_size)
        
        # Calculate symptom prevalence for this cluster
        if cluster_size > 0:  # Avoid division by zero
            for i, symptom in enumerate(all_symptoms_list):
                # Count cases with this symptom in the cluster
                symptom_count = np.sum(X[cluster_indices, i])
                
                # Add to profiles (as percentage or count)
                if as_percentage:
                    percentage = (symptom_count / cluster_size) * 100
                    cluster_profiles[f"{symptom} %"].append(round(percentage, 1))
                else:
                    cluster_profiles[symptom].append(int(symptom_count))
        else:
            # Handle empty clusters (should be rare)
            for symptom in all_symptoms_list:
                if as_percentage:
                    cluster_profiles[f"{symptom} %"].append(0.0)
                else:
                    cluster_profiles[symptom].append(0)
    
    # Convert to DataFrame
    profile_df = pd.DataFrame(cluster_profiles)
    
    # Set Cluster as index for better display
    profile_df = profile_df.set_index('Cluster')
    
    return profile_df


def identify_potential_dengue_clusters(cluster_profiles, dengue_symptoms, threshold=50.0):
    """
    Identify clusters that may represent potential dengue outbreaks
    
    This function analyzes cluster profiles to identify clusters with a high prevalence
    of dengue-specific symptoms, which may indicate potential dengue outbreaks.
    
    Args:
        cluster_profiles (pandas.DataFrame): Cluster profiles from profile_clusters function
        dengue_symptoms (list): List of symptoms specific to dengue fever
        threshold (float, optional): Minimum percentage threshold for key symptoms. Defaults to 50.0.
        
    Returns:
        list: List of cluster IDs that may represent potential dengue outbreaks
    """
    potential_dengue_clusters = []
    
    # Check if cluster_profiles has percentage columns
    percentage_cols = [col for col in cluster_profiles.columns if " %" in col]
    if not percentage_cols:
        raise ValueError("cluster_profiles must contain percentage columns (e.g., 'Fever %')")
    
    # Get the symptom percentage columns for dengue symptoms
    dengue_cols = [f"{symptom} %" for symptom in dengue_symptoms if f"{symptom} %" in percentage_cols]
    
    if not dengue_cols:
        raise ValueError("None of the specified dengue symptoms found in cluster profiles")
    
    # Iterate through clusters
    for cluster_id in cluster_profiles.index:
        # Count how many dengue symptoms exceed the threshold in this cluster
        high_symptom_count = sum(1 for col in dengue_cols if cluster_profiles.loc[cluster_id, col] >= threshold)
        
        # If at least 3 dengue symptoms exceed the threshold, consider it a potential dengue cluster
        if high_symptom_count >= 3:
            potential_dengue_clusters.append(cluster_id)
    
    return potential_dengue_clusters


def analyze_cluster_by_location(data, cluster_col='Cluster', location_col='Location'):
    """
    Analyze the distribution of clusters across different locations
    
    This function is particularly useful for identifying geographical patterns
    in the clustering results, which may indicate localized outbreaks.
    
    Args:
        data (pandas.DataFrame): Clustered data with location information
        cluster_col (str, optional): Column name containing cluster labels. Defaults to 'Cluster'.
        location_col (str, optional): Column name containing location information. Defaults to 'Location'.
        
    Returns:
        pandas.DataFrame: Cross-tabulation of clusters by location
    """
    # Validate inputs
    if cluster_col not in data.columns:
        raise ValueError(f"Column '{cluster_col}' not found in data")
    if location_col not in data.columns:
        raise ValueError(f"Column '{location_col}' not found in data")
    
    # Create a cross-tabulation of clusters by location
    location_distribution = pd.crosstab(
        data[location_col], 
        data[cluster_col], 
        margins=True, 
        margins_name='Total'
    )
    
    # Calculate percentages (what percentage of each location belongs to each cluster)
    location_pct = location_distribution.div(location_distribution['Total'], axis=0) * 100
    location_pct = location_pct.drop('Total', axis=1)
    location_pct = location_pct.round(1)
    
    return {
        'counts': location_distribution,
        'percentages': location_pct
    }


def identify_dengue_like_clusters(cluster_profiles, dengue_symptoms_list, thresholds):
    """
    Identify clusters potentially indicating a dengue outbreak based on symptom profiles
    
    This function analyzes cluster profiles to identify clusters that match dengue-like
    symptom patterns according to specified thresholds. It can detect clusters with both
    individual high-prevalence symptoms and specific symptom combinations characteristic
    of dengue fever.
    
    Args:
        cluster_profiles (pandas.DataFrame): Cluster profiles from profile_clusters function
        dengue_symptoms_list (list): List of key dengue symptoms to look for
        thresholds (dict): Dictionary defining criteria for dengue-like clusters, e.g.,
            {
                'min_dengue_symptoms': 3,  # Min number of key dengue symptoms needed
                'min_prevalence_pct': 60,  # Min % of cluster members with a key symptom
                'key_symptom_combo': ['Fever', 'Headache', 'Muscle Pain'],  # Specific combo to check
                'key_symptom_combo_pct': 50  # Min % for the specific combo
            }
        
    Returns:
        dict: Dictionary with two keys:
            'dengue_clusters': List of cluster IDs identified as dengue-like
            'cluster_reports': Dictionary mapping cluster IDs to reports explaining why they were flagged
    
    Note:
        This function operates on a single time window. For actual outbreak detection,
        you'll need to compare cluster characteristics across multiple time windows.
    """
    # Validate inputs
    if not isinstance(cluster_profiles, pd.DataFrame):
        raise ValueError("cluster_profiles must be a pandas DataFrame")
    
    if not isinstance(dengue_symptoms_list, list) or len(dengue_symptoms_list) == 0:
        raise ValueError("dengue_symptoms_list must be a non-empty list")
    
    required_threshold_keys = ['min_dengue_symptoms', 'min_prevalence_pct']
    if not isinstance(thresholds, dict) or not all(key in thresholds for key in required_threshold_keys):
        raise ValueError(f"thresholds must be a dictionary containing at least: {required_threshold_keys}")
    
    # Check if cluster_profiles has percentage columns
    percentage_cols = [col for col in cluster_profiles.columns if " %" in col]
    if not percentage_cols:
        raise ValueError("cluster_profiles must contain percentage columns (e.g., 'Fever %')")
    
    # Get the symptom percentage columns for dengue symptoms
    dengue_cols = [f"{symptom} %" for symptom in dengue_symptoms_list if f"{symptom} %" in percentage_cols]
    
    if not dengue_cols:
        raise ValueError("None of the specified dengue symptoms found in cluster profiles")
    
    # Extract threshold values
    min_dengue_symptoms = thresholds['min_dengue_symptoms']
    min_prevalence_pct = thresholds['min_prevalence_pct']
    
    # Check for key symptom combo if specified
    has_combo_check = 'key_symptom_combo' in thresholds and 'key_symptom_combo_pct' in thresholds
    if has_combo_check:
        key_combo = thresholds['key_symptom_combo']
        key_combo_pct = thresholds['key_symptom_combo_pct']
        # Validate combo symptoms exist in dengue_symptoms_list
        if not all(symptom in dengue_symptoms_list for symptom in key_combo):
            missing = [s for s in key_combo if s not in dengue_symptoms_list]
            raise ValueError(f"The following key_symptom_combo items are not in dengue_symptoms_list: {missing}")
    
    # Initialize results
    dengue_clusters = []
    cluster_reports = {}
    
    # Analyze each cluster
    for cluster_id in cluster_profiles.index:
        # Skip clusters with size 0 (should be rare)
        if cluster_profiles.loc[cluster_id, 'Size'] == 0:
            continue
        
        # Check which dengue symptoms exceed the minimum prevalence threshold
        high_prevalence_symptoms = []
        for symptom in dengue_symptoms_list:
            col = f"{symptom} %"
            if col in cluster_profiles.columns and cluster_profiles.loc[cluster_id, col] >= min_prevalence_pct:
                high_prevalence_symptoms.append(symptom)
        
        # Initialize report for this cluster
        report = []
        
        # Check if enough individual symptoms exceed the threshold
        individual_check_passed = len(high_prevalence_symptoms) >= min_dengue_symptoms
        if individual_check_passed:
            report.append(f"Cluster has {len(high_prevalence_symptoms)} dengue symptoms with prevalence >= {min_prevalence_pct}%: {', '.join(high_prevalence_symptoms)}")
        
        # Check for specific symptom combination if specified
        combo_check_passed = False
        if has_combo_check:
            # Check if all symptoms in the key combo exceed the combo threshold
            combo_symptoms_present = []
            for symptom in key_combo:
                col = f"{symptom} %"
                if col in cluster_profiles.columns and cluster_profiles.loc[cluster_id, col] >= key_combo_pct:
                    combo_symptoms_present.append(symptom)
            
            combo_check_passed = len(combo_symptoms_present) == len(key_combo)
            if combo_check_passed:
                report.append(f"Cluster has the key symptom combination with prevalence >= {key_combo_pct}%: {', '.join(key_combo)}")
        
        # Determine if this is a dengue-like cluster
        is_dengue_like = individual_check_passed
        if has_combo_check:
            # If combo check is specified, require either individual check OR combo check to pass
            is_dengue_like = individual_check_passed or combo_check_passed
        
        if is_dengue_like:
            dengue_clusters.append(cluster_id)
            cluster_reports[cluster_id] = report
    
    return {
        'dengue_clusters': dengue_clusters,
        'cluster_reports': cluster_reports
    }


def track_clusters_over_time(time_series_data, date_col, cluster_col, location_col=None, window='W'):
    """
    Track cluster sizes and distributions over time
    
    This function analyzes how clusters evolve over time, which is crucial for
    detecting emerging outbreaks. It can track both overall cluster sizes and
    location-specific patterns if location data is provided.
    
    Args:
        time_series_data (pandas.DataFrame): Data with time, cluster, and optional location columns
        date_col (str): Column name containing dates
        cluster_col (str): Column name containing cluster labels
        location_col (str, optional): Column name containing location information
        window (str, optional): Time window for resampling ('D' for daily, 'W' for weekly, etc.)
        
    Returns:
        dict: Dictionary containing time series analysis results
    """
    # Validate inputs
    if date_col not in time_series_data.columns:
        raise ValueError(f"Column '{date_col}' not found in data")
    if cluster_col not in time_series_data.columns:
        raise ValueError(f"Column '{cluster_col}' not found in data")
    
    # Ensure date column is datetime type
    data = time_series_data.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    
    # Set date as index for time series analysis
    data = data.set_index(date_col)
    
    # Track overall cluster sizes over time
    cluster_counts = data.groupby([pd.Grouper(freq=window), cluster_col]).size().unstack(fill_value=0)
    
    # Calculate the percentage of each cluster in each time window
    cluster_totals = cluster_counts.sum(axis=1)
    cluster_pcts = cluster_counts.div(cluster_totals, axis=0) * 100
    
    results = {
        'cluster_counts': cluster_counts,
        'cluster_percentages': cluster_pcts
    }
    
    # If location data is provided, track location-specific patterns
    if location_col is not None and location_col in time_series_data.columns:
        # Track clusters by location over time
        location_cluster_counts = data.groupby([pd.Grouper(freq=window), location_col, cluster_col]).size().unstack(fill_value=0)
        
        # Reshape to have time and location as multi-index, clusters as columns
        location_cluster_counts = location_cluster_counts.reset_index()
        location_cluster_counts = location_cluster_counts.pivot_table(
            index=[location_cluster_counts.index.get_level_values(0), location_col],
            values=location_cluster_counts.columns[2:],
            aggfunc='sum'
        )
        
        results['location_cluster_counts'] = location_cluster_counts
    
    return results
