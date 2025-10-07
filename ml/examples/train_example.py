"""
Example: Train models programmatically (without CLI)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import load_exoplanet_csv
from preprocess import preprocess_features
from train import train_random_forest, train_xgboost, save_model, save_metrics
from sklearn.model_selection import train_test_split


def create_sample_data():
    """Create sample exoplanet data for demonstration."""
    np.random.seed(42)
    n_samples = 500
    
    data = {
        'orbital_period': np.random.uniform(1, 100, n_samples),
        'transit_duration': np.random.uniform(0.5, 5, n_samples),
        'planet_radius': np.random.uniform(0.5, 2.5, n_samples),
        'stellar_temp': np.random.uniform(4000, 7000, n_samples),
        'flux': [np.random.randn(50).tolist() for _ in range(n_samples)],
        'label': np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    }
    
    df = pd.DataFrame(data)
    csv_path = 'sample_train_data.csv'
    df.to_csv(csv_path, index=False)
    print(f"Created sample data: {csv_path}")
    return csv_path


def main():
    print("=" * 80)
    print("PROGRAMMATIC TRAINING EXAMPLE")
    print("=" * 80)
    
    # Step 1: Create or load data
    csv_path = create_sample_data()
    
    # Step 2: Load CSV
    print("\nLoading data...")
    df = load_exoplanet_csv(csv_path)
    print(f"Loaded {len(df)} samples")
    
    # Step 3: Preprocess
    print("\nPreprocessing...")
    X, y = preprocess_features(df, label_column='label', extract_flux_features=True)
    print(f"Feature matrix: {X.shape}")
    
    # Step 4: Split
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Step 5: Train Random Forest
    print("\n" + "=" * 80)
    print("Training Random Forest...")
    print("=" * 80)
    rf_model, rf_metrics = train_random_forest(
        X_train, y_train, X_test, y_test, random_state=42
    )
    
    print("\nRandom Forest Metrics:")
    for metric, value in rf_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save Random Forest
    save_model(rf_model, '../models/example_rf.pkl', 'random_forest')
    save_metrics(rf_metrics, '../models/example_rf_metrics.json')
    
    # Step 6: Train XGBoost (if available)
    try:
        print("\n" + "=" * 80)
        print("Training XGBoost...")
        print("=" * 80)
        xgb_model, xgb_metrics = train_xgboost(
            X_train, y_train, X_test, y_test, random_state=42
        )
        
        print("\nXGBoost Metrics:")
        for metric, value in xgb_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Save XGBoost
        save_model(xgb_model, '../models/example_xgb.pkl', 'xgboost')
        save_metrics(xgb_metrics, '../models/example_xgb_metrics.json')
        
    except ImportError as e:
        print(f"\nSkipping XGBoost: {e}")
    
    # Cleanup
    import os
    if os.path.exists(csv_path):
        os.remove(csv_path)
        print(f"\nCleaned up {csv_path}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print("\nModels saved to ml/models/")


if __name__ == "__main__":
    main()

