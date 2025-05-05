"""
Preprocessing Module for Dengue Outbreak Detection System

This module handles data cleaning, normalization, and feature engineering
to prepare the data for clustering analysis.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from copy import deepcopy


def clean_data(data):
    """
    Clean the data by removing missing values, duplicates, etc.
    
    Args:
        data (pandas.DataFrame): Raw data
        
    Returns:
        pandas.DataFrame: Cleaned data
    """
    # Remove duplicates
    data = data.drop_duplicates()
    
    # Handle missing values (basic approach - can be enhanced)
    data = data.dropna()
    
    return data


def normalize_features(data, columns, method='standard'):
    """
    Normalize numerical features
    
    Args:
        data (pandas.DataFrame): Data to normalize
        columns (list): List of columns to normalize
        method (str): Normalization method ('standard' or 'minmax')
        
    Returns:
        pandas.DataFrame: Data with normalized features
    """
    data_copy = data.copy()
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Method must be 'standard' or 'minmax'")
    
    data_copy[columns] = scaler.fit_transform(data_copy[columns])
    
    return data_copy


def engineer_features(data):
    """
    Create new features from existing data
    
    Args:
        data (pandas.DataFrame): Input data
        
    Returns:
        pandas.DataFrame: Data with engineered features
    """
    # Placeholder for feature engineering
    # Example: Creating symptom count feature
    # data['symptom_count'] = data[['fever', 'headache', 'rash', ...]].sum(axis=1)
    
    return data


def standardize_symptoms(data, symptom_mappings=None):
    """
    Standardize symptom terms by replacing variations with standard terms
    
    This function processes the 'Symptoms' column in the DataFrame, which should
    contain lists of symptom strings. It replaces variations of symptom terms
    with standardized terms according to the provided mapping dictionary.
    
    Args:
        data (pandas.DataFrame): DataFrame containing a 'Symptoms' column with lists of symptoms
        symptom_mappings (dict, optional): Dictionary mapping symptom variations to standard terms.
                                         If None, a default mapping will be used.
        
    Returns:
        pandas.DataFrame: DataFrame with standardized symptom terms
    
    Example:
        >>> df = pd.DataFrame({'Symptoms': [['high fever', 'headache'], ['body aches', 'feverish']]})
        >>> mappings = {'high fever': 'Fever', 'feverish': 'Fever', 'body aches': 'Muscle Pain'}
        >>> standardize_symptoms(df, mappings)
        # Result: DataFrame with 'Symptoms' as [['Fever', 'headache'], ['Muscle Pain', 'Fever']]
    """
    # Create a copy of the data to avoid modifying the original
    result = data.copy()
    
    # Define default symptom mappings if none provided
    if symptom_mappings is None:
        symptom_mappings = {
            # Fever variations
            'high fever': 'Fever',
            'low-grade fever': 'Fever',
            'feverish': 'Fever',
            'elevated temperature': 'Fever',
            'high temperature': 'Fever',
            'fever': 'Fever',
            
            # Headache variations
            'head pain': 'Headache',
            'migraine': 'Headache',
            'head ache': 'Headache',
            'headache': 'Headache',
            
            # Muscle Pain variations
            'body aches': 'Muscle Pain',
            'myalgia': 'Muscle Pain',
            'muscle aches': 'Muscle Pain',
            'body pain': 'Muscle Pain',
            'muscle pain': 'Muscle Pain',
            
            # Joint Pain variations
            'arthralgia': 'Joint Pain',
            'joint aches': 'Joint Pain',
            'painful joints': 'Joint Pain',
            'joint pain': 'Joint Pain',
            
            # Rash variations
            'skin rash': 'Rash',
            'maculopapular rash': 'Rash',
            'petechial rash': 'Rash',
            'skin eruptions': 'Rash',
            'rash': 'Rash',
            
            # Nausea variations
            'feeling sick': 'Nausea',
            'queasy': 'Nausea',
            'nauseous': 'Nausea',
            'nausea': 'Nausea',
            
            # Vomiting variations
            'throwing up': 'Vomiting',
            'emesis': 'Vomiting',
            'vomiting': 'Vomiting',
            
            # Retro-orbital Pain variations
            'eye pain': 'Retro-orbital Pain',
            'pain behind eyes': 'Retro-orbital Pain',
            'orbital pain': 'Retro-orbital Pain',
            'retro-orbital pain': 'Retro-orbital Pain',
            
            # Mild Bleeding variations
            'petechiae': 'Mild Bleeding',
            'gum bleeding': 'Mild Bleeding',
            'nose bleeding': 'Mild Bleeding',
            'epistaxis': 'Mild Bleeding',
            'easy bruising': 'Mild Bleeding',
            'mild bleeding': 'Mild Bleeding',
            
            # Cough variations
            'dry cough': 'Cough',
            'productive cough': 'Cough',
            'coughing': 'Cough',
            'cough': 'Cough',
            
            # Sore Throat variations
            'pharyngitis': 'Sore Throat',
            'throat pain': 'Sore Throat',
            'painful throat': 'Sore Throat',
            'sore throat': 'Sore Throat',
            
            # Runny Nose variations
            'rhinorrhea': 'Runny Nose',
            'nasal discharge': 'Runny Nose',
            'nasal congestion': 'Runny Nose',
            'runny nose': 'Runny Nose',
            
            # Fatigue variations
            'tiredness': 'Fatigue',
            'malaise': 'Fatigue',
            'weakness': 'Fatigue',
            'lethargy': 'Fatigue',
            'exhaustion': 'Fatigue',
            'fatigue': 'Fatigue',
            
            # Diarrhea variations
            'loose stools': 'Diarrhea',
            'watery stool': 'Diarrhea',
            'frequent bowel movements': 'Diarrhea',
            'diarrhea': 'Diarrhea'
        }
    
    # Check if 'Symptoms' column exists and contains lists
    if 'Symptoms' not in result.columns:
        raise ValueError("DataFrame must contain a 'Symptoms' column")
    
    # Check if the first non-null entry is a list (to verify column format)
    first_valid_idx = result['Symptoms'].first_valid_index()
    if first_valid_idx is not None and not isinstance(result.loc[first_valid_idx, 'Symptoms'], list):
        # If symptoms are stored as strings (e.g., comma-separated), convert to lists
        if isinstance(result.loc[first_valid_idx, 'Symptoms'], str):
            result['Symptoms'] = result['Symptoms'].apply(lambda x: x.split(', ') if isinstance(x, str) else x)
        else:
            raise ValueError("'Symptoms' column must contain lists or comma-separated strings")
    
    # Standardize symptoms
    def standardize_symptom_list(symptom_list):
        if not isinstance(symptom_list, list):
            return []
        
        standardized = []
        for symptom in symptom_list:
            # Convert to lowercase for case-insensitive matching
            symptom_lower = symptom.lower() if isinstance(symptom, str) else str(symptom).lower()
            
            # Find matching standard term or keep original
            standard_term = symptom_mappings.get(symptom_lower, symptom)
            standardized.append(standard_term)
        
        return standardized
    
    # Apply standardization to each row
    result['Symptoms'] = result['Symptoms'].apply(standardize_symptom_list)
    
    return result


def handle_missing_values(data, strategy='drop'):
    """
    Handle missing values in the DataFrame
    
    Args:
        data (pandas.DataFrame): Input data with potential missing values
        strategy (str): Strategy for handling missing values
                       'drop': Drop rows with missing values in critical columns
                       'fill': Fill missing values with defaults
        
    Returns:
        pandas.DataFrame: DataFrame with missing values handled
    """
    # Create a copy of the data to avoid modifying the original
    result = data.copy()
    
    # Define critical columns that should not have missing values
    critical_columns = ['Symptoms', 'Location', 'ReportDateTime']
    
    if strategy == 'drop':
        # Drop rows with missing values in critical columns
        result = result.dropna(subset=critical_columns)
        
        # Also drop rows where Symptoms is an empty list
        if 'Symptoms' in result.columns:
            # Check if Symptoms column contains lists
            if isinstance(result['Symptoms'].iloc[0], list):
                result = result[result['Symptoms'].apply(lambda x: len(x) > 0)]
    
    elif strategy == 'fill':
        # Fill missing values with defaults
        if 'Location' in result.columns and result['Location'].isna().any():
            result['Location'] = result['Location'].fillna('Unknown')
        
        if 'AgeGroup' in result.columns and result['AgeGroup'].isna().any():
            result['AgeGroup'] = result['AgeGroup'].fillna('Unknown')
        
        if 'Symptoms' in result.columns:
            # Handle empty or missing symptom lists
            if isinstance(result['Symptoms'].iloc[0], list):
                result['Symptoms'] = result['Symptoms'].apply(lambda x: x if isinstance(x, list) and len(x) > 0 else ['Unknown'])
            else:
                result['Symptoms'] = result['Symptoms'].fillna('Unknown')
    
    else:
        raise ValueError("Strategy must be 'drop' or 'fill'")
    
    return result


def convert_symptoms_to_binary(data):
    """
    Convert the 'Symptoms' list column to binary indicator columns for each symptom
    
    Args:
        data (pandas.DataFrame): DataFrame with a 'Symptoms' column containing lists of symptoms
        
    Returns:
        pandas.DataFrame: DataFrame with binary columns for each symptom
    """
    # Create a copy of the data
    result = data.copy()
    
    # Check if 'Symptoms' column exists and contains lists
    if 'Symptoms' not in result.columns:
        raise ValueError("DataFrame must contain a 'Symptoms' column")
    
    # Get all unique symptoms
    all_symptoms = set()
    for symptom_list in result['Symptoms']:
        if isinstance(symptom_list, list):
            all_symptoms.update(symptom_list)
        elif isinstance(symptom_list, str):
            # Handle case where symptoms might be stored as comma-separated strings
            all_symptoms.update([s.strip() for s in symptom_list.split(',')])
    
    # Create binary columns for each symptom
    for symptom in sorted(all_symptoms):
        if isinstance(result['Symptoms'].iloc[0], list):
            result[symptom] = result['Symptoms'].apply(lambda x: 1 if symptom in x else 0)
        else:
            # Handle case where symptoms might be stored as comma-separated strings
            result[symptom] = result['Symptoms'].apply(
                lambda x: 1 if isinstance(x, str) and symptom in [s.strip() for s in x.split(',')] else 0
            )
    
    return result


def create_feature_vectors(df, all_symptoms_list):
    """
    Convert standardized symptom lists into binary feature vectors for clustering
    
    This function takes a DataFrame with a 'Symptoms' column containing lists of standardized
    symptoms and converts it into a binary feature matrix where each column represents
    a symptom and each row represents a case. The order of columns matches the provided
    all_symptoms_list.
    
    Args:
        df (pandas.DataFrame): DataFrame with a 'Symptoms' column containing lists of standardized symptoms
        all_symptoms_list (list): A fixed list containing all possible standardized symptoms in the desired order
        
    Returns:
        pandas.DataFrame: DataFrame where columns are symptoms and values are binary indicators (1=present, 0=absent)
                         The index of the returned DataFrame matches the index of the input DataFrame
    
    Example:
        >>> symptoms_list = ['Fever', 'Headache', 'Muscle Pain', 'Cough']
        >>> df = pd.DataFrame({
        ...     'CaseID': [1, 2, 3],
        ...     'Symptoms': [['Fever', 'Headache'], ['Muscle Pain'], ['Fever', 'Cough']]
        ... })
        >>> create_feature_vectors(df, symptoms_list)
        # Result: DataFrame with columns ['Fever', 'Headache', 'Muscle Pain', 'Cough'] and binary values
    """
    # Create a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Check if 'Symptoms' column exists
    if 'Symptoms' not in result_df.columns:
        raise ValueError("DataFrame must contain a 'Symptoms' column")
    
    # Check if the first non-null entry is a list (to verify column format)
    first_valid_idx = result_df['Symptoms'].first_valid_index()
    if first_valid_idx is not None and not isinstance(result_df.loc[first_valid_idx, 'Symptoms'], list):
        # If symptoms are stored as strings (e.g., comma-separated), convert to lists
        if isinstance(result_df.loc[first_valid_idx, 'Symptoms'], str):
            result_df['Symptoms'] = result_df['Symptoms'].apply(lambda x: x.split(', ') if isinstance(x, str) else x)
        else:
            raise ValueError("'Symptoms' column must contain lists or comma-separated strings")
    
    # Create an empty feature matrix filled with zeros
    feature_matrix = pd.DataFrame(0, index=result_df.index, columns=all_symptoms_list)
    
    # Fill the feature matrix based on symptoms presence
    for idx, row in result_df.iterrows():
        symptoms = row['Symptoms']
        if isinstance(symptoms, list):
            for symptom in symptoms:
                if symptom in all_symptoms_list:
                    feature_matrix.loc[idx, symptom] = 1
    
    return feature_matrix


def create_feature_vectors_with_sklearn(df, all_symptoms_list):
    """
    Convert standardized symptom lists into binary feature vectors using scikit-learn's MultiLabelBinarizer
    
    This is an alternative implementation using scikit-learn's MultiLabelBinarizer.
    It's more efficient for large datasets but requires an additional dependency.
    
    Args:
        df (pandas.DataFrame): DataFrame with a 'Symptoms' column containing lists of standardized symptoms
        all_symptoms_list (list): A fixed list containing all possible standardized symptoms in the desired order
        
    Returns:
        pandas.DataFrame: DataFrame where columns are symptoms and values are binary indicators (1=present, 0=absent)
    """
    from sklearn.preprocessing import MultiLabelBinarizer
    
    # Create a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Check if 'Symptoms' column exists
    if 'Symptoms' not in result_df.columns:
        raise ValueError("DataFrame must contain a 'Symptoms' column")
    
    # Convert string symptoms to lists if needed
    first_valid_idx = result_df['Symptoms'].first_valid_index()
    if first_valid_idx is not None and isinstance(result_df.loc[first_valid_idx, 'Symptoms'], str):
        result_df['Symptoms'] = result_df['Symptoms'].apply(lambda x: x.split(', ') if isinstance(x, str) else x)
    
    # Initialize the MultiLabelBinarizer with the predefined classes (symptoms)
    mlb = MultiLabelBinarizer(classes=all_symptoms_list)
    
    # Transform the symptoms lists into a binary matrix
    binary_matrix = mlb.fit_transform(result_df['Symptoms'])
    
    # Convert to DataFrame with proper column names
    feature_matrix = pd.DataFrame(binary_matrix, columns=mlb.classes_, index=result_df.index)
    
    # Reorder columns to match all_symptoms_list order
    feature_matrix = feature_matrix[all_symptoms_list]
    
    return feature_matrix
