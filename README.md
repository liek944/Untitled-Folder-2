# Dengue Outbreak Early Detection System

## 1. Project Goal

The primary objective of this project is to develop a system capable of **detecting potential early-stage dengue fever outbreaks** by analyzing reported symptoms, potentially combined with location (e.g., Barangays in Roxas, Oriental Mindoro, Philippines) and time data. The system aims to identify unusual clusters of dengue-like symptoms *before* widespread clinical testing or official public health declarations confirm an outbreak.

This system is intended to serve as an analytical tool for **public health analysts and epidemiologists**.

## 2. Core Technique

The system uses unsupervised clustering algorithms to identify unusual patterns in symptom data that may indicate early dengue outbreaks. Multiple clustering approaches (DBSCAN, K-means, Hierarchical) are implemented to provide robust analysis.

## 3. Project Structure

```
dengue-outbreak-detection/
├── main.py                 # Main script to run the pipeline
├── modules/                # Core modules
│   ├── data_loader.py      # Functions for loading and saving data
│   ├── preprocessing.py    # Data cleaning and feature engineering
│   ├── clustering.py       # Clustering algorithms implementation
│   ├── analysis.py         # Analysis of clustering results
│   └── visualization.py    # Visualization functions
├── requirements.txt        # Python dependencies
└── README.md               # This documentation
```

## 4. Installation

1. Clone this repository
2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## 5. Usage

Run the main script with your data file and specify which features to use for clustering:

```bash
python main.py --data path/to/your/data.csv --features fever headache joint_pain rash --output ./results
```

## 6. Data Requirements

The system expects a CSV file with the following minimum columns:
- Symptom data (e.g., fever, headache, joint_pain, rash)
- Date information (optional but recommended for temporal analysis)
- Location data (optional but recommended for spatial analysis)

## 7. Output

The system generates:
- Cluster visualizations
- Cluster characteristic analysis
- Temporal trend analysis (if date data is available)
- Identified anomalous clusters that may represent potential outbreaks

## 8. Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 9. License

This project is licensed under the MIT License - see the LICENSE file for details.
The core methodology relies on **unsupervised machine learning**, specifically **clustering algorithms**. The system groups individuals based on the similarity of their reported symptoms to identify patterns indicative of a potential outbreak.

* **Primary Algorithm:** K-Means clustering is the initial focus due to its simplicity, though other algorithms like DBSCAN might be explored.
* **Detection Logic:** The system analyzes the resulting clusters, profiling their symptom composition and monitoring their size, emergence, and geographic concentration over time to flag potential outbreaks based on predefined criteria and thresholds.

## 3. Scope

* **In Scope:** Data simulation/acquisition, data preprocessing (cleaning, symptom standardization, feature engineering), implementation of clustering algorithms, development of cluster interpretation logic for outbreak detection, basic result visualization and reporting.
* **Out of Scope (for initial versions):** Real-time integration with live hospital/clinic data feeds, development of a full-scale public health deployment platform, clinical diagnosis of individuals (the system suggests *potential outbreak clusters*, not individual diagnoses).

## 4. Technology Stack

* **Primary Language:** Python (v3.x)
* **Core Libraries:**
    * `pandas`: Data manipulation, preprocessing, and analysis.
    * `numpy`: Numerical operations, array handling.
    * `scikit-learn`: Implementation of clustering algorithms (KMeans, DBSCAN), preprocessing tools (e.g., MultiLabelBinarizer), model evaluation metrics (e.g., silhouette score).
    * `matplotlib` / `seaborn`: Data visualization (plotting cluster characteristics, trends).
    * `datetime`: Handling temporal data.
* **Potential Libraries (for extensions):**
    * `geopandas` / `folium`: Geospatial analysis and map-based visualization.
    * `Flask` / `Django`: Building a simple web interface/dashboard to display results (optional).
* **Environment Management:**`venv`
* **Version Control:** Git

## 5. Data Handling

* **Data Source:** Initially uses **simulated data** designed to mimic realistic symptom reporting patterns, including background noise (common cold/flu) and injected dengue outbreak characteristics within specific timeframes and locations (e.g., Barangays in Naga City). Can be adapted for historical datasets if available.
* **Data Points per Record:**
    * `CaseID`: Unique identifier.
    * `ReportDateTime`: Timestamp of the report.
    * `Location`: Granular location (e.g., Barangay, City/Municipality).
    * `Symptoms`: List of reported symptoms (strings).
    * `AgeGroup` (Optional).
* **Preprocessing Steps:**
    * **Cleaning:** Handling missing values.
    * **Symptom Standardization:** Mapping symptom variations (e.g., "high fever," "feverish") to a controlled vocabulary (e.g., "Fever"). A predefined list of relevant dengue and non-dengue symptoms is used.
    * **Feature Engineering:** Converting the list of standardized symptoms into **binary feature vectors (One-Hot Encoding)** suitable for clustering. Each vector position corresponds to a symptom in the master list (1 if present, 0 if absent).
* **Aggregation:** Data is typically processed in defined time windows (e.g., daily, weekly). Spatial aggregation might be used for location-specific analysis.

## 6. Code Structure Overview

The project follows a modular structure:

/dengue_detection|-- /data                # Placeholder for raw/processed data files (if applicable)|-- /notebooks           # Jupyter notebooks for exploration/prototyping|-- /src                 # Source code modules|   |-- init.py|   |-- data_loader.py   # Functions for data simulation or loading|   |-- preprocessing.py # Cleaning, standardization, feature engineering|   |-- clustering.py    # Clustering algorithm implementation (KMeans, etc.)|   |-- analysis.py      # Cluster profiling, outbreak detection logic|   |-- visualization.py # Plotting functions (using Matplotlib/Seaborn)|   |-- utils.py         # Common utility functions|   |-- config.py        # Configuration (paths, parameters, thresholds)|-- /static              # For web app: CSS, JS, saved images (if using Flask/Django)|   |-- /images|-- /templates           # For web app: HTML templates (if using Flask/Django)|-- app.py               # Flask/Django web application entry point (if applicable)|-- main.py              # Main script to run the end-to-end analysis pipeline|-- requirements.txt     # Python package dependencies|-- README.md            # This file*(Note: Structure may evolve. `/static`, `/templates`, `app.py` are only relevant if the optional web interface is built).*

## 7. Execution / Usage

The core analysis pipeline can typically be executed via the `main.py` script.

1.  **Setup:**
    * Clone the repository.
    * Set up the Python environment (e.g., `python -m venv venv && source venv/bin/activate` or `conda create --name dengue_env python=3.x`).
    * Install dependencies: `pip install -r requirements.txt`.
2.  **Configuration:** Adjust parameters (simulation settings, date ranges, file paths, clustering K value, detection thresholds) in `src/config.py` or pass as arguments if `main.py` is designed to accept them.
3.  **Run Analysis:** Execute the main script: `python main.py`.
4.  **Output:** Results (e.g., identified suspicious cluster IDs, summary statistics, generated plots) will be printed to the console and/or saved to specified output files/directories (like `static/images/` if plots are saved).

*(If a Flask/Django web interface exists, refer to specific instructions for running the web server, likely involving `flask run` or `python manage.py runserver`).*

## 8. Key Functionality for AI Assistant

* Understand the data flow from simulation/loading (`data_loader.py`) through preprocessing (`preprocessing.py`) to feature vectors.
* Recognize the use of `scikit-learn` for clustering (`clustering.py`) and the role of K-Means.
* Understand the logic for interpreting clusters (`analysis.py`), particularly how symptom profiles are generated and compared against dengue criteria.
* Assist with debugging issues in `pandas` data manipulation and `scikit-learn` model application.
* Help generate or interpret visualizations (`visualization.py`).
* Refactor code across modules (`src/`) for clarity or efficiency.
* If the web interface is being developed (`app.py`, `/templates`), assist with Flask/Django routing, template rendering, and passing data between the backend analysis and the frontend.

