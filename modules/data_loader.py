"""
Data Loader Module for Dengue Outbreak Detection System

This module handles loading data from various sources (CSV, API, etc.)
and provides a standardized format for the rest of the system.
"""

import pandas as pd
import numpy as np


def load_data(file_path):
    """
    Load data from a CSV file
    
    Args:
        file_path (str): Path to the data file
        
    Returns:
        pandas.DataFrame: Loaded data
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Successfully loaded data from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def save_data(data, file_path):
    """
    Save data to a CSV file
    
    Args:
        data (pandas.DataFrame): Data to save
        file_path (str): Path to save the data
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        data.to_csv(file_path, index=False)
        print(f"Successfully saved data to {file_path}")
        return True
    except Exception as e:
        print(f"Error saving data: {e}")
        return False
