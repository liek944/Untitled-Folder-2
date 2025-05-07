from flask import Flask, render_template, jsonify
import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import shutil

# Import the main function from main.py
from main import run_dengue_detection

# Initialize Flask application
app = Flask(__name__)

def run_analysis_pipeline():
    """
    Execute the dengue analysis pipeline and prepare results for display.
    
    Returns:
        dict: Dictionary containing key results needed for display
    """
    # Create a timestamp for this analysis run
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Run the dengue detection pipeline
    try:
        # Set output directory to static/images to save plots directly there
        output_dir = './static/images'
        
        # Clear previous images if they exist
        if os.path.exists(output_dir):
            # Keep the directory but remove all files in it
            for filename in os.listdir(output_dir):
                file_path = os.path.join(output_dir, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
        else:
            # Create the directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
        
        # Run the dengue detection pipeline
        results = run_dengue_detection(simulate=True, output_dir=output_dir)
        
        # Extract the clustered data
        data_with_clusters = results['data']
        
        # Extract cluster profiles
        cluster_profiles = results['cluster_profiles']
        
        # Extract dengue results
        dengue_results = results['dengue_results']
        
        # Extract suspicious clusters (potential dengue clusters)
        # The key might be 'potential_dengue_clusters' or just 'dengue_clusters' depending on the implementation
        suspicious_clusters = dengue_results.get('potential_dengue_clusters', [])
        if not suspicious_clusters and 'dengue_clusters' in dengue_results:
            suspicious_clusters = dengue_results['dengue_clusters']
        
        # Prepare results dictionary for the template
        # Convert cluster profiles DataFrame to HTML for proper display in the template
        cluster_profiles_html = ''
        if isinstance(cluster_profiles, pd.DataFrame):
            cluster_profiles_html = cluster_profiles.to_html(classes='table table-striped table-hover', index=True)
            
        results_dict = {
            'timestamp': timestamp,
            'total_cases': len(data_with_clusters),
            'cluster_profiles': cluster_profiles.to_dict() if isinstance(cluster_profiles, pd.DataFrame) else cluster_profiles,
            'cluster_profiles_html': cluster_profiles_html,
            'suspicious_clusters': suspicious_clusters,
            'cluster_size_plot': 'cluster_sizes.png',
            'symptom_plot': 'symptom_heatmap.png',
            'all_symptoms_plot': 'all_symptoms_heatmap.png',
            'cluster_locations_plot': 'cluster_locations.png',
            'temporal_trends_plot': 'temporal_trends.png' if os.path.exists(os.path.join(output_dir, 'temporal_trends.png')) else None
        }
        
        return results_dict
    except Exception as e:
        # Log the error
        print(f"Error in analysis pipeline: {str(e)}")
        raise e

@app.route('/')
def home():
    """Route for the homepage"""
    try:
        # Run the analysis pipeline
        results = run_analysis_pipeline()
        
        # Render the template with the results
        return render_template('index.html', results=results)
    except Exception as e:
        # Handle errors and display an error message
        error_message = f"An error occurred while running the analysis: {str(e)}"
        return render_template('index.html', error=error_message)

@app.route('/api/results')
def api_results():
    """API endpoint to get analysis results as JSON"""
    try:
        results = run_analysis_pipeline()
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
