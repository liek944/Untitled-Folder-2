"""
Visualization Module for Dengue Outbreak Detection System

This module provides functions to visualize clustering results and analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import Counter


def plot_clusters_2d(data, x_col, y_col, cluster_col='cluster', figsize=(10, 8)):
    """
    Create a 2D scatter plot of clusters
    
    Args:
        data (pandas.DataFrame): Clustered data
        x_col (str): Column name for x-axis
        y_col (str): Column name for y-axis
        cluster_col (str): Column name containing cluster labels
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each cluster with a different color
    sns.scatterplot(data=data, x=x_col, y=y_col, hue=cluster_col, palette='viridis', ax=ax)
    
    # Add labels and title
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'Cluster Distribution: {x_col} vs {y_col}')
    
    return fig


def plot_cluster_characteristics(cluster_summary, feature_cols=None, figsize=(12, 10)):
    """
    Create a heatmap of cluster characteristics
    
    Args:
        cluster_summary (pandas.DataFrame): Cluster characteristics summary
        feature_cols (list): Columns to visualize (if None, use all columns)
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    if feature_cols is None:
        feature_cols = cluster_summary.columns.tolist()
        # Remove size column if it exists
        if 'size' in feature_cols:
            feature_cols.remove('size')
            
    # Create a copy of the data with selected columns
    plot_data = cluster_summary[feature_cols].copy()
    
    # Normalize the data for better visualization
    normalized_data = (plot_data - plot_data.mean()) / plot_data.std()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(normalized_data, annot=True, cmap='coolwarm', center=0, ax=ax)
    
    plt.title('Cluster Characteristics (Z-score normalized)')
    plt.tight_layout()
    
    return fig


def plot_temporal_trends(temporal_data, cluster_ids=None, trend_type='weekly', figsize=(12, 6)):
    """
    Plot temporal trends for clusters
    
    Args:
        temporal_data (dict): Dictionary containing temporal trend data
        cluster_ids (list): List of cluster IDs to plot (if None, plot all)
        trend_type (str): Type of trend to plot ('weekly' or 'monthly')
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Select the appropriate trend data
    if trend_type not in temporal_data:
        raise ValueError(f"trend_type must be one of {list(temporal_data.keys())}")
        
    trend_data = temporal_data[trend_type]
    
    # Select clusters to plot
    if cluster_ids is None:
        cluster_ids = trend_data.columns.tolist()
    else:
        # Ensure all requested clusters exist in the data
        cluster_ids = [c for c in cluster_ids if c in trend_data.columns]
        
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    for cluster in cluster_ids:
        trend_data[cluster].plot(ax=ax, label=f'Cluster {cluster}')
    
    plt.title(f'{trend_type.capitalize()} Trends by Cluster')
    plt.xlabel('Time Period')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return fig


def plot_cluster_sizes(cluster_data, figsize=(10, 6), title='Cluster Sizes', color='skyblue', highlight_clusters=None, highlight_color='tomato'):
    """
    Create a bar chart showing the number of cases in each cluster
    
    Args:
        cluster_data: Either:
            - A list or array of cluster labels
            - A pandas DataFrame with a 'Size' column (e.g., from profile_clusters)
            - A pandas DataFrame with a cluster column
        figsize (tuple): Figure size
        title (str): Plot title
        color (str): Bar color
        highlight_clusters (list, optional): List of cluster IDs to highlight
        highlight_color (str): Color for highlighted clusters
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Process input data to get cluster counts
    if isinstance(cluster_data, pd.DataFrame):
        if 'Size' in cluster_data.columns:
            # Input is a cluster profile DataFrame with Size column
            cluster_counts = cluster_data['Size'].copy()
        else:
            # Assume the DataFrame has a cluster column, find it
            cluster_cols = [col for col in cluster_data.columns if 'cluster' in str(col).lower() or 'Cluster' in str(col)]
            if not cluster_cols:
                raise ValueError("DataFrame must contain a 'Size' column or a column with 'cluster' in its name")
            cluster_col = cluster_cols[0]  # Use the first matching column
            cluster_counts = cluster_data[cluster_col].value_counts().sort_index()
    else:
        # Input is a list/array of cluster labels
        cluster_counts = pd.Series(Counter(cluster_data)).sort_index()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine bar colors
    if highlight_clusters is not None:
        colors = [highlight_color if idx in highlight_clusters else color for idx in cluster_counts.index]
    else:
        colors = color
    
    # Create the bar chart
    bars = ax.bar(cluster_counts.index.astype(str), cluster_counts.values, color=colors)
    
    # Add labels and title
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Number of Cases')
    ax.set_title(title)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom')
    
    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig


def plot_symptom_prevalence(cluster_profiles, symptoms=None, plot_type='heatmap', figsize=(12, 8), cmap='YlOrRd', title=None):
    """
    Create a visualization showing the prevalence of symptoms across different clusters
    
    Args:
        cluster_profiles (pandas.DataFrame): Cluster profiles DataFrame from profile_clusters
        symptoms (list, optional): List of symptoms to include (if None, include all)
        plot_type (str): Type of plot to create ('heatmap' or 'barplot')
        figsize (tuple): Figure size
        cmap (str): Colormap for heatmap
        title (str, optional): Plot title
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Validate input
    if not isinstance(cluster_profiles, pd.DataFrame):
        raise ValueError("cluster_profiles must be a pandas DataFrame")
    
    # Check if the DataFrame has percentage columns
    percentage_cols = [col for col in cluster_profiles.columns if " %" in col]
    if not percentage_cols:
        raise ValueError("cluster_profiles must contain percentage columns (e.g., 'Fever %')")
    
    # Extract symptom names from column names
    all_symptoms = [col.replace(' %', '') for col in percentage_cols]
    
    # Filter symptoms if specified
    if symptoms is not None:
        valid_symptoms = [s for s in symptoms if f"{s} %" in percentage_cols]
        if not valid_symptoms:
            raise ValueError(f"None of the specified symptoms found in cluster profiles")
        symptom_cols = [f"{s} %" for s in valid_symptoms]
    else:
        symptom_cols = percentage_cols
        valid_symptoms = all_symptoms
    
    # Extract the data for plotting
    plot_data = cluster_profiles[symptom_cols].copy()
    
    # Rename columns to remove the ' %' suffix for cleaner labels
    plot_data.columns = [col.replace(' %', '') for col in plot_data.columns]
    
    # Create the plot based on the specified type
    if plot_type == 'heatmap':
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create the heatmap
        sns.heatmap(plot_data, annot=True, cmap=cmap, fmt='.1f', ax=ax, vmin=0, vmax=100)
        
        # Set title
        if title is None:
            title = 'Symptom Prevalence (%) by Cluster'
        ax.set_title(title)
        
    elif plot_type == 'barplot':
        # Reshape the data for grouped bar plot
        plot_data_melted = plot_data.reset_index().melt(
            id_vars='Cluster',
            var_name='Symptom',
            value_name='Prevalence (%)'
        )
        
        # Create the grouped bar plot
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(x='Symptom', y='Prevalence (%)', hue='Cluster', data=plot_data_melted, ax=ax)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Set title
        if title is None:
            title = 'Symptom Prevalence by Cluster'
        ax.set_title(title)
        
        # Add legend
        ax.legend(title='Cluster')
        
    else:
        raise ValueError("plot_type must be 'heatmap' or 'barplot'")
    
    plt.tight_layout()
    return fig


def plot_cluster_locations(location_data, cluster_id=None, figsize=(12, 8), cmap='Blues', title=None):
    """
    Create a visualization showing the distribution of clusters across locations
    
    Args:
        location_data (dict): Dictionary from analyze_cluster_by_location function
        cluster_id (int, optional): Specific cluster to highlight (if None, show all)
        figsize (tuple): Figure size
        cmap (str): Colormap for heatmap
        title (str, optional): Plot title
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Validate input
    if not isinstance(location_data, dict) or 'counts' not in location_data:
        raise ValueError("location_data must be a dictionary with a 'counts' key")
    
    # Get the location counts data
    location_counts = location_data['counts']
    
    # Remove the 'Total' column and row if present
    if 'Total' in location_counts.columns:
        plot_data = location_counts.drop('Total', axis=1)
    else:
        plot_data = location_counts.copy()
    
    if 'Total' in plot_data.index:
        plot_data = plot_data.drop('Total')
    
    # Filter for a specific cluster if requested
    if cluster_id is not None:
        if cluster_id not in plot_data.columns:
            raise ValueError(f"Cluster {cluster_id} not found in location data")
        
        # Create a bar chart for a single cluster
        fig, ax = plt.subplots(figsize=figsize)
        
        # Sort locations by cluster count
        sorted_data = plot_data[cluster_id].sort_values(ascending=False)
        
        # Create the bar chart
        bars = ax.bar(sorted_data.index, sorted_data.values, color='skyblue')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only add labels for non-zero values
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom')
        
        # Set title
        if title is None:
            title = f'Distribution of Cluster {cluster_id} Across Locations'
        ax.set_title(title)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        ax.set_xlabel('Location')
        ax.set_ylabel('Number of Cases')
        
    else:
        # Create a heatmap for all clusters
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create the heatmap
        sns.heatmap(plot_data, annot=True, cmap=cmap, fmt='d', ax=ax)
        
        # Set title
        if title is None:
            title = 'Distribution of Clusters Across Locations'
        ax.set_title(title)
    
    plt.tight_layout()
    return fig
