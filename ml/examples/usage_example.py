"""
Example usage of load_exoplanet_csv and preprocess_features functions.

This script demonstrates how to:
1. Load exoplanet data from a CSV file
2. Preprocess the features for machine learning
"""

import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import load_exoplanet_csv
from preprocess import preprocess_features
import numpy as np


def example_usage():
    """
    Example demonstrating the complete workflow.
    """
    print("=" * 80)
    print("Exoplanet Data Loading and Preprocessing Example")
    print("=" * 80)
    
    # Example 1: Load CSV data
    print("\n" + "=" * 80)
    print("Step 1: Load CSV Data")
    print("=" * 80)
    
    # For this example, we'll create a sample CSV first
    create_sample_csv('sample_exoplanet_data.csv')
    
    try:
        # Load the CSV file
        df = load_exoplanet_csv('sample_exoplanet_data.csv')
        print(f"\nLoaded DataFrame shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"\nFirst few rows:")
        print(df.head())
        
        # Example 2: Preprocess features
        print("\n" + "=" * 80)
        print("Step 2: Preprocess Features")
        print("=" * 80)
        
        X, y = preprocess_features(df, label_column='label', extract_flux_features=True)
        
        print(f"\nProcessed features shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Unique labels: {np.unique(y)}")
        print(f"\nFeature statistics:")
        print(f"  Mean: {X.mean():.6f}")
        print(f"  Std: {X.std():.6f}")
        print(f"  Min: {X.min():.6f}")
        print(f"  Max: {X.max():.6f}")
        
        print("\n" + "=" * 80)
        print("Example completed successfully!")
        print("=" * 80)
        
        return X, y
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def create_sample_csv(filename: str):
    """
    Create a sample CSV file for demonstration.
    """
    import pandas as pd
    import numpy as np
    
    print(f"Creating sample CSV file: {filename}")
    
    # Generate sample data
    n_samples = 100
    np.random.seed(42)
    
    data = {
        'Orbital Period': np.random.uniform(1, 100, n_samples),  # days
        'Transit Duration': np.random.uniform(0.5, 5, n_samples),  # hours
        'Planet Radius': np.random.uniform(0.5, 2.5, n_samples),  # Earth radii
        'Stellar Temp': np.random.uniform(4000, 7000, n_samples),  # Kelvin
        'flux': [np.random.randn(50).tolist() for _ in range(n_samples)],  # Light curve
        'label': np.random.choice([0, 1], n_samples, p=[0.9, 0.1])  # 0=no exoplanet, 1=exoplanet
    }
    
    # Add some missing values
    for i in np.random.choice(n_samples, 5, replace=False):
        data['Orbital Period'][i] = np.nan
    for i in np.random.choice(n_samples, 3, replace=False):
        data['Planet Radius'][i] = np.nan
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Created {filename} with {n_samples} samples")


if __name__ == "__main__":
    X, y = example_usage()
    
    # Clean up sample file
    import os
    if os.path.exists('sample_exoplanet_data.csv'):
        os.remove('sample_exoplanet_data.csv')
        print("\nCleaned up sample CSV file")

