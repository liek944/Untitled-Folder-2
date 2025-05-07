"""
Main Module for Dengue Outbreak Detection System

This script orchestrates the entire workflow of the dengue outbreak detection system,
from data loading to analysis and visualization for early detection of dengue outbreaks
in Roxas, Oriental Mindoro, Philippines.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random

# Import modules
from modules import data_loader, preprocessing, clustering, analysis, visualization
from modules.data_simulator import simulate_data, save_simulated_data


def run_dengue_detection(simulate=True, data_path=None, output_dir='./output'):
    """
    Run the complete dengue outbreak detection pipeline
    
    This function orchestrates the entire process for dengue outbreak detection,
    from data loading/simulation to analysis and visualization.
    
    Args:
        simulate (bool): Whether to simulate data (True) or load from file (False)
        data_path (str, optional): Path to the input data file if simulate=False
        output_dir (str): Directory to save outputs
        
    Returns:
        dict: Results of the analysis including identified potential outbreaks
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # ===== STEP 1: DEFINE PARAMETERS =====
    print("\n===== STEP 1: DEFINING PARAMETERS =====")
    
    # Define all symptoms (both dengue and non-dengue)
    all_symptoms = [
        'Fever', 'Headache', 'Muscle Pain', 'Joint Pain', 'Rash', 
        'Nausea', 'Vomiting', 'Retro-orbital Pain', 'Mild Bleeding',
        'Cough', 'Sore Throat', 'Runny Nose', 'Fatigue', 'Diarrhea'
    ]
    
    # Define dengue-specific symptoms
    dengue_symptoms = [
        'Fever', 'Headache', 'Muscle Pain', 'Joint Pain', 'Rash',
        'Nausea', 'Vomiting', 'Retro-orbital Pain', 'Mild Bleeding'
    ]
    
    # Define clustering parameters
    n_clusters = 5  # Number of clusters for K-means
    random_state = random.randint(1, 10000)  # Random seed for different results each run
    
    # Define dengue detection thresholds
    detection_thresholds = {
        'min_dengue_symptoms': 3,  # At least 3 dengue symptoms with high prevalence
        'min_prevalence_pct': 60,  # At least 60% of cluster members have these symptoms
        'key_symptom_combo': ['Fever', 'Headache', 'Muscle Pain'],  # Classic dengue triad
        'key_symptom_combo_pct': 50  # At least 50% prevalence for the key combo
    }
    
    # ===== STEP 2: LOAD OR SIMULATE DATA =====
    print("\n===== STEP 2: LOADING/SIMULATING DATA =====")
    
    if simulate:
        print("Simulating symptom report data...")
        
        # List of barangays in Roxas, Oriental Mindoro, Philippines
        roxas_barangays = [
            'Bagumbayan', 'Cantil', 'Dangay', 'Happy Valley', 'Libertad',
            'Libtong', 'Little Tanauan', 'Maraska', 'Odiong', 'Paclasan',
            'Poblacion', 'San Aquilino', 'San Isidro', 'San Jose', 'San Mariano',
            'San Miguel', 'San Rafael', 'San Vicente', 'Victoria'
        ]
        
        # Simulation parameters
        start_date = '2024-01-01'
        end_date = '2024-04-30'
        outbreak_start_date = '2024-03-01'
        outbreak_end_date = '2024-03-25'
        outbreak_barangay = 'San Isidro'  # Location of simulated outbreak
        outbreak_factor = 1.5  # Intensity of the outbreak
        
        # Generate simulated data with randomized parameters
        data = simulate_data(
            num_records=random.randint(800, 1200),  # Random number of records
            start_date=start_date,
            end_date=end_date,
            barangays=roxas_barangays,
            outbreak_start_date=outbreak_start_date,
            outbreak_end_date=outbreak_end_date,
            outbreak_barangay=random.choice(roxas_barangays),  # Random outbreak location
            outbreak_symptom_increase_factor=round(random.uniform(1.2, 2.0), 1)  # Random outbreak intensity
        )
        
        # Save simulated data for reference
        sim_data_path = os.path.join(output_dir, 'simulated_data.csv')
        save_simulated_data(data, sim_data_path)
        print(f"Simulated data saved to {sim_data_path}")
        
        # Also save a version with binary symptom columns for reference
        sim_binary_path = os.path.join(output_dir, 'simulated_data_binary.csv')
        save_simulated_data(data, sim_binary_path, symptoms_as_columns=True)
        
    else:
        if data_path is None:
            raise ValueError("data_path must be provided when simulate=False")
            
        print(f"Loading data from {data_path}...")
        data = data_loader.load_data(data_path)
        if data is None:
            print("Error: Failed to load data")
            return None
    
    print(f"Data shape: {data.shape}")
    
    # ===== STEP 3: PREPROCESS DATA =====
    print("\n===== STEP 3: PREPROCESSING DATA =====")
    
    # Handle missing values
    print("Handling missing values...")
    data_cleaned = preprocessing.handle_missing_values(data, strategy='drop')
    print(f"Data shape after cleaning: {data_cleaned.shape}")
    
    # Standardize symptoms
    print("Standardizing symptoms...")
    data_standardized = preprocessing.standardize_symptoms(data_cleaned)
    
    # ===== STEP 4: FEATURE ENGINEERING =====
    print("\n===== STEP 4: FEATURE ENGINEERING =====")
    
    # Create feature vectors for clustering
    print("Creating feature vectors...")
    feature_matrix = preprocessing.create_feature_vectors(data_standardized, all_symptoms)
    print(f"Feature matrix shape: {feature_matrix.shape}")
    
    # Save the feature matrix for reference
    feature_matrix_path = os.path.join(output_dir, 'feature_matrix.csv')
    feature_matrix.to_csv(feature_matrix_path)
    print(f"Feature matrix saved to {feature_matrix_path}")
    
    # ===== STEP 5: APPLY CLUSTERING =====
    print("\n===== STEP 5: APPLYING CLUSTERING =====")
    
    # Apply K-means clustering with random state
    print("Applying K-means clustering with random seed...")
    print(f"Using random seed: {random_state}")
    labels, kmeans_model = clustering.apply_kmeans_clustering(
        feature_matrix,
        n_clusters=n_clusters,
        random_state=random_state
    )
    
    # Get cluster centroids
    centroids = clustering.get_cluster_centroids(kmeans_model, feature_names=all_symptoms)
    print("Cluster centroids:")
    print(centroids)
    
    # Add cluster labels to original data
    data_with_clusters = clustering.add_cluster_labels_to_data(data_standardized, labels)
    
    # Save clustered data
    clustered_data_path = os.path.join(output_dir, 'clustered_data.csv')
    data_with_clusters.to_csv(clustered_data_path, index=False)
    print(f"Clustered data saved to {clustered_data_path}")
    
    # ===== STEP 6: ANALYZE CLUSTERS =====
    print("\n===== STEP 6: ANALYZING CLUSTERS =====")
    
    # Profile the clusters
    print("Profiling clusters...")
    cluster_profiles = analysis.profile_clusters(feature_matrix, labels, all_symptoms)
    print("Cluster profiles:")
    print(cluster_profiles)
    
    # Analyze clusters by location
    print("Analyzing geographical distribution...")
    location_analysis = analysis.analyze_cluster_by_location(
        data_with_clusters, 
        cluster_col='Cluster', 
        location_col='Location'
    )
    
    # Analyze temporal patterns if date column exists
    print("Analyzing temporal patterns...")
    time_analysis = analysis.track_clusters_over_time(
        data_with_clusters,
        date_col='ReportDateTime',
        cluster_col='Cluster',
        location_col='Location',
        window='W'  # Weekly analysis
    )
    
    # ===== STEP 7: IDENTIFY SUSPICIOUS CLUSTERS =====
    print("\n===== STEP 7: IDENTIFYING SUSPICIOUS CLUSTERS =====")
    
    # Identify potential dengue outbreak clusters
    print("Identifying potential dengue outbreak clusters...")
    dengue_results = analysis.identify_dengue_like_clusters(
        cluster_profiles, 
        dengue_symptoms, 
        detection_thresholds
    )
    
    # Print results
    dengue_clusters = dengue_results['dengue_clusters']
    if dengue_clusters:
        print(f"\nPOTENTIAL DENGUE OUTBREAK DETECTED in clusters: {dengue_clusters}")
        for cluster_id in dengue_clusters:
            print(f"\nCluster {cluster_id} flagged as potential dengue outbreak:")
            for reason in dengue_results['cluster_reports'][cluster_id]:
                print(f"  - {reason}")
            
            # Show this cluster's profile
            print(f"\nCluster {cluster_id} profile:")
            print(cluster_profiles.loc[[cluster_id]])
            
            # Show geographical distribution
            if 'Location' in data_with_clusters.columns:
                cluster_locations = data_with_clusters[data_with_clusters['Cluster'] == cluster_id]['Location'].value_counts()
                print(f"\nGeographical distribution of Cluster {cluster_id}:")
                print(cluster_locations.head(5))  # Show top 5 locations
    else:
        print("No potential dengue outbreaks detected.")
    
    # ===== STEP 8: VISUALIZE RESULTS =====
    print("\n===== STEP 8: VISUALIZING RESULTS =====")
    
    # Create visualizations directory if needed
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Plot cluster sizes (highlighting potential outbreak clusters)
    print("Plotting cluster sizes...")
    visualization.plot_cluster_sizes(
        cluster_profiles, 
        highlight_clusters=dengue_clusters,
        title='Cluster Sizes (Potential Outbreaks Highlighted)',
        save_path=os.path.join(output_dir, 'cluster_sizes.png')
    )
    
    # Plot symptom prevalence as a heatmap
    print("Plotting symptom prevalence...")
    visualization.plot_symptom_prevalence(
        cluster_profiles, 
        symptoms=dengue_symptoms,  # Focus on dengue symptoms
        plot_type='heatmap',
        title='Dengue Symptom Prevalence (%) by Cluster',
        save_path=os.path.join(output_dir, 'symptom_heatmap.png')
    )
    
    # Plot all symptoms as a heatmap too
    visualization.plot_symptom_prevalence(
        cluster_profiles,
        plot_type='heatmap',
        title='All Symptom Prevalence (%) by Cluster',
        save_path=os.path.join(output_dir, 'all_symptoms_heatmap.png')
    )
    
    # Plot geographical distribution of clusters
    print("Plotting geographical distribution...")
    visualization.plot_cluster_locations(
        location_analysis,
        title='Distribution of Clusters Across Locations',
        save_path=os.path.join(output_dir, 'cluster_locations.png')
    )
    
    # For each potential dengue cluster, plot its specific location distribution
    for cluster_id in dengue_clusters:
        visualization.plot_cluster_locations(
            location_analysis,
            cluster_id=cluster_id,
            title=f'Distribution of Cluster {cluster_id} Across Locations',
            save_path=os.path.join(output_dir, f'cluster_{cluster_id}_locations.png')
        )
    
    # Plot temporal trends if available
    if 'cluster_counts' in time_analysis:
        print("Plotting temporal trends...")
        # Create a custom temporal trends plot and save it directly
        fig4 = plt.figure(figsize=(12, 6))
        ax = fig4.add_subplot(111)
        time_analysis['cluster_counts'].plot(ax=ax)
        ax.set_title('Cluster Sizes Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Cases')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the figure directly to the output directory
        temporal_trends_path = os.path.join(output_dir, 'temporal_trends.png')
        plt.savefig(temporal_trends_path)
        plt.close(fig4)
    
    # ===== STEP 9: SUMMARY =====
    print("\n===== STEP 9: SUMMARY =====")
    
    print(f"Analysis complete. Results saved to {output_dir}")
    print(f"Visualizations saved to {viz_dir}")
    
    if dengue_clusters:
        print(f"\nPOTENTIAL DENGUE OUTBREAK DETECTED in clusters: {dengue_clusters}")
        print("Please review the detailed analysis and visualizations.")
    else:
        print("\nNo potential dengue outbreaks detected in this time window.")
        print("Continue monitoring for changes in symptom patterns.")
    
    # Return results dictionary
    results = {
        'data': data_with_clusters,
        'feature_matrix': feature_matrix,
        'kmeans_model': kmeans_model,
        'cluster_profiles': cluster_profiles,
        'dengue_results': dengue_results,
        'location_analysis': location_analysis,
        'time_analysis': time_analysis,
        'detection_thresholds': detection_thresholds
    }
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Dengue Outbreak Detection System for Roxas, Oriental Mindoro')
    parser.add_argument('--simulate', action='store_true', default=True,
                       help='Simulate data instead of loading from file (default: True)')
    parser.add_argument('--data', type=str, required=False, 
                       help='Path to the input data file (required if --simulate is False)')
    parser.add_argument('--output', type=str, default='./output', 
                       help='Directory to save outputs (default: ./output)')
    
    args = parser.parse_args()
    
    # Run the dengue detection pipeline
    if args.simulate:
        print("Running dengue detection with simulated data...")
        results = run_dengue_detection(simulate=True, output_dir=args.output)
    else:
        if args.data is None:
            parser.error("--data is required when --simulate is False")
        print(f"Running dengue detection with data from {args.data}...")
        results = run_dengue_detection(simulate=False, data_path=args.data, output_dir=args.output)
