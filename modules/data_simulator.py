"""
Data Simulator Module for Dengue Outbreak Detection System

This module provides functions to generate synthetic data for testing and
development of the dengue outbreak detection system, with a focus on
the Naga City, Philippines context.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


def simulate_data(num_records, start_date, end_date, barangays, 
                 outbreak_start_date, outbreak_end_date, outbreak_barangay,
                 outbreak_symptom_increase_factor):
    """
    Generate synthetic symptom report data for dengue outbreak detection analysis.
    
    This function creates a realistic dataset simulating symptom reports across different
    barangays in Naga City, Philippines, with an artificially injected outbreak in a
    specific barangay during a defined time period.
    
    Args:
        num_records (int): Total number of records to generate
        start_date (str or datetime): Simulation start date
        end_date (str or datetime): Simulation end date
        barangays (list): List of barangays in Naga City to include
        outbreak_start_date (str or datetime): Date when simulated outbreak begins
        outbreak_end_date (str or datetime): Date when simulated outbreak ends
        outbreak_barangay (str): The specific barangay where the outbreak is concentrated
        outbreak_symptom_increase_factor (float): Factor by which dengue-like symptoms 
                                                 increase during the outbreak period
    
    Returns:
        pandas.DataFrame: Simulated symptom report data with the following columns:
            - CaseID: Unique identifier for each case
            - ReportDateTime: When the symptoms were reported
            - Location: Barangay where the case was reported
            - Symptoms: List of reported symptoms
            - AgeGroup: Age group of the patient
    """
    # Convert string dates to datetime objects if necessary
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    if isinstance(outbreak_start_date, str):
        outbreak_start_date = datetime.strptime(outbreak_start_date, '%Y-%m-%d')
    if isinstance(outbreak_end_date, str):
        outbreak_end_date = datetime.strptime(outbreak_end_date, '%Y-%m-%d')
    
    # Define symptom lists
    dengue_symptoms = [
        'Fever', 'Headache', 'Muscle Pain', 'Joint Pain', 'Rash', 
        'Nausea', 'Vomiting', 'Retro-orbital Pain', 'Mild Bleeding'
    ]
    
    noise_symptoms = [
        'Cough', 'Sore Throat', 'Runny Nose', 'Fatigue', 'Diarrhea'
    ]
    
    # Define age groups
    age_groups = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60+']
    
    # Initialize data storage
    data = {
        'CaseID': [],
        'ReportDateTime': [],
        'Location': [],
        'Symptoms': [],
        'AgeGroup': []
    }
    
    # Generate records
    for i in range(1, num_records + 1):
        # Generate CaseID
        case_id = i
        
        # Generate random report date and time
        time_range = (end_date - start_date).total_seconds()
        random_seconds = random.randint(0, int(time_range))
        report_datetime = start_date + timedelta(seconds=random_seconds)
        
        # Randomly select a barangay, with higher probability for the outbreak barangay during outbreak
        if (outbreak_start_date <= report_datetime <= outbreak_end_date):
            # During outbreak period, increase probability of selecting the outbreak barangay
            if random.random() < 0.4:  # 40% chance to be in the outbreak barangay during outbreak
                location = outbreak_barangay
            else:
                location = random.choice(barangays)
        else:
            location = random.choice(barangays)
        
        # Determine if this is an outbreak case with some randomness
        # Even outside the outbreak area and time, there's a small chance of dengue-like cases
        # And within the outbreak, not all cases will show the pattern
        if outbreak_start_date <= report_datetime <= outbreak_end_date and location == outbreak_barangay:
            is_outbreak_case = random.random() < 0.85  # 85% chance during outbreak in the specific location
        elif outbreak_start_date <= report_datetime <= outbreak_end_date:
            is_outbreak_case = random.random() < 0.25  # 25% chance during outbreak in other locations
        else:
            is_outbreak_case = random.random() < 0.10  # 10% chance of dengue-like cases outside outbreak
        
        # Generate symptoms with more randomness
        if is_outbreak_case:
            # For outbreak cases, higher probability of dengue symptoms but with variability
            # Some outbreak cases might have more symptoms than others
            severity = random.random()  # Random severity factor
            if severity > 0.7:  # Severe cases (30%)
                num_dengue_symptoms = random.randint(5, len(dengue_symptoms))
                num_noise_symptoms = random.randint(0, 2)  # Few noise symptoms
            elif severity > 0.3:  # Moderate cases (40%)
                num_dengue_symptoms = random.randint(3, 5)
                num_noise_symptoms = random.randint(0, 3)
            else:  # Mild cases (30%)
                num_dengue_symptoms = random.randint(2, 4)
                num_noise_symptoms = random.randint(1, 3)
        else:
            # For non-outbreak cases, more variability
            case_type = random.random()
            if case_type < 0.1:  # 10% chance of dengue-like but not outbreak
                num_dengue_symptoms = random.randint(2, 4)
                num_noise_symptoms = random.randint(1, 4)
            elif case_type < 0.4:  # 30% chance of mixed symptoms
                num_dengue_symptoms = random.randint(1, 3)
                num_noise_symptoms = random.randint(1, 3)
            else:  # 60% chance of mostly non-dengue symptoms
                num_dengue_symptoms = random.randint(0, 2)
                num_noise_symptoms = random.randint(1, len(noise_symptoms))
        
        # Select dengue symptoms
        if is_outbreak_case and random.random() < 0.9:  # 90% chance of fever in outbreak cases
            # Ensure fever is included for most outbreak cases
            selected_dengue_symptoms = ['Fever']
            remaining_dengue = [s for s in dengue_symptoms if s != 'Fever']
            if num_dengue_symptoms > 1:
                additional_symptoms = random.sample(remaining_dengue, min(num_dengue_symptoms - 1, len(remaining_dengue)))
                selected_dengue_symptoms.extend(additional_symptoms)
        else:
            # Randomly select dengue symptoms
            selected_dengue_symptoms = random.sample(dengue_symptoms, min(num_dengue_symptoms, len(dengue_symptoms)))
        
        # Select noise symptoms
        selected_noise_symptoms = random.sample(noise_symptoms, min(num_noise_symptoms, len(noise_symptoms)))
        
        # Combine symptoms
        symptoms = selected_dengue_symptoms + selected_noise_symptoms
        random.shuffle(symptoms)  # Randomize order
        
        # Randomly select age group with variable distribution
        # Make age distribution vary based on whether it's an outbreak case
        if is_outbreak_case:
            # During outbreaks, children and elderly might be more affected
            age_weights = [0.20, 0.15, 0.10, 0.10, 0.10, 0.15, 0.20]  # Higher weights for young and elderly
        else:
            # Normal distribution with slight randomness each time
            base_weight = 1.0 / len(age_groups)
            age_weights = [base_weight + random.uniform(-0.03, 0.03) for _ in range(len(age_groups))]
            # Normalize weights to ensure they sum to 1
            total = sum(age_weights)
            age_weights = [w/total for w in age_weights]
        
        age_group = random.choices(age_groups, weights=age_weights)[0]
        
        # Add to data dictionary
        data['CaseID'].append(case_id)
        data['ReportDateTime'].append(report_datetime)
        data['Location'].append(location)
        data['Symptoms'].append(symptoms)
        data['AgeGroup'].append(age_group)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Sort by ReportDateTime to ensure chronological order
    df = df.sort_values('ReportDateTime').reset_index(drop=True)
    
    # Re-assign CaseIDs to maintain sequential order after sorting
    df['CaseID'] = range(1, len(df) + 1)
    
    return df


def save_simulated_data(df, output_path, symptoms_as_columns=False):
    """
    Save the simulated data to a CSV file.
    
    Args:
        df (pandas.DataFrame): The simulated data
        output_path (str): Path to save the CSV file
        symptoms_as_columns (bool): If True, convert symptoms list to binary columns
                                   for each symptom (useful for analysis)
    
    Returns:
        str: Path to the saved file
    """
    df_to_save = df.copy()
    
    # Convert datetime to string for CSV storage
    df_to_save['ReportDateTime'] = df_to_save['ReportDateTime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    if symptoms_as_columns:
        # Get all unique symptoms
        all_symptoms = set()
        for symptom_list in df_to_save['Symptoms']:
            all_symptoms.update(symptom_list)
        
        # Create binary columns for each symptom
        for symptom in sorted(all_symptoms):
            df_to_save[symptom] = df_to_save['Symptoms'].apply(lambda x: 1 if symptom in x else 0)
        
        # Drop the original Symptoms column
        df_to_save = df_to_save.drop('Symptoms', axis=1)
    else:
        # Convert symptom lists to strings for CSV storage
        df_to_save['Symptoms'] = df_to_save['Symptoms'].apply(lambda x: ', '.join(x))
    
    # Save to CSV
    df_to_save.to_csv(output_path, index=False)
    print(f"Simulated data saved to {output_path}")
    
    return output_path


def example_usage():
    """
    Example of how to use the simulate_data function.
    """
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
    outbreak_barangay = 'San Isidro'
    outbreak_factor = 1.5
    
    # Generate data
    simulated_data = simulate_data(
        num_records=1000,
        start_date=start_date,
        end_date=end_date,
        barangays=roxas_barangays,
        outbreak_start_date=outbreak_start_date,
        outbreak_end_date=outbreak_end_date,
        outbreak_barangay=outbreak_barangay,
        outbreak_symptom_increase_factor=outbreak_factor
    )
    
    # Save data
    save_simulated_data(simulated_data, 'simulated_dengue_data.csv')
    save_simulated_data(simulated_data, 'simulated_dengue_data_binary.csv', symptoms_as_columns=True)
    
    return simulated_data


if __name__ == "__main__":
    # Run the example when the script is executed directly
    example_data = example_usage()
    print(f"Generated {len(example_data)} simulated records")
    print(example_data.head())
