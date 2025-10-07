"""
Unit tests for data loading and preprocessing functions.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import load_exoplanet_csv
from preprocess import preprocess_features


class TestLoadExoplanetCSV:
    """Tests for load_exoplanet_csv function."""
    
    def create_sample_csv(self, filename, data=None):
        """Helper to create sample CSV files."""
        if data is None:
            data = {
                'Orbital Period': [10.5, 25.3, 15.7, 8.2, np.nan],
                'Transit Duration': [2.5, 3.1, 1.8, 4.2, 2.9],
                'Planet Radius': [1.2, np.nan, 1.8, 0.9, 1.5],
                'Stellar Temp': [5778, 6200, 5500, np.nan, 5900],
                'flux': [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]],
                'label': [0, 1, 0, 1, 0]
            }
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        return filename
    
    def test_basic_loading(self):
        """Test basic CSV loading with normalized columns."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filename = f.name
        
        try:
            self.create_sample_csv(filename)
            df = load_exoplanet_csv(filename)
            
            # Check that file was loaded
            assert len(df) > 0, "DataFrame should not be empty"
            
            # Check required columns exist
            required_cols = ['orbital_period', 'transit_duration', 'planet_radius', 'stellar_temp', 'flux']
            for col in required_cols:
                assert col in df.columns, f"Required column '{col}' should exist"
            
            # Check column names are normalized (lowercase, underscores)
            for col in df.columns:
                assert col.islower(), f"Column '{col}' should be lowercase"
                assert ' ' not in col, f"Column '{col}' should not contain spaces"
        
        finally:
            if os.path.exists(filename):
                os.remove(filename)
    
    def test_missing_values_handling(self):
        """Test that missing values are handled correctly."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filename = f.name
        
        try:
            self.create_sample_csv(filename)
            df = load_exoplanet_csv(filename)
            
            # Check that missing values were handled
            required_cols = ['orbital_period', 'transit_duration', 'planet_radius', 'stellar_temp']
            for col in required_cols:
                assert df[col].isnull().sum() == 0, f"Column '{col}' should have no missing values"
        
        finally:
            if os.path.exists(filename):
                os.remove(filename)
    
    def test_column_aliases(self):
        """Test that common column name aliases are recognized."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filename = f.name
        
        try:
            # Use alias names
            data = {
                'Period': [10.5, 25.3, 15.7],
                'Duration': [2.5, 3.1, 1.8],
                'Radius': [1.2, 1.4, 1.8],
                'Temperature': [5778, 6200, 5500],
                'light_curve': [[1, 2], [3, 4], [5, 6]],
            }
            self.create_sample_csv(filename, data)
            df = load_exoplanet_csv(filename)
            
            # Should recognize aliases and rename to standard names
            assert 'orbital_period' in df.columns
            assert 'transit_duration' in df.columns
            assert 'planet_radius' in df.columns
            assert 'stellar_temp' in df.columns
            assert 'flux' in df.columns
        
        finally:
            if os.path.exists(filename):
                os.remove(filename)
    
    def test_missing_required_column(self):
        """Test that error is raised when required column is missing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filename = f.name
        
        try:
            # Missing 'flux' column
            data = {
                'Orbital Period': [10.5, 25.3],
                'Transit Duration': [2.5, 3.1],
                'Planet Radius': [1.2, 1.4],
                'Stellar Temp': [5778, 6200],
            }
            self.create_sample_csv(filename, data)
            
            with pytest.raises(ValueError, match="Required column.*not found"):
                load_exoplanet_csv(filename)
        
        finally:
            if os.path.exists(filename):
                os.remove(filename)
    
    def test_file_not_found(self):
        """Test that error is raised for non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_exoplanet_csv('non_existent_file.csv')


class TestPreprocessFeatures:
    """Tests for preprocess_features function."""
    
    def create_sample_dataframe(self):
        """Helper to create sample DataFrame."""
        np.random.seed(42)
        return pd.DataFrame({
            'orbital_period': [10.5, 25.3, 15.7, 8.2, 30.1],
            'transit_duration': [2.5, 3.1, 1.8, 4.2, 2.9],
            'planet_radius': [1.2, 1.4, 1.8, 0.9, 1.5],
            'stellar_temp': [5778, 6200, 5500, 5100, 5900],
            'flux': [np.random.randn(50) for _ in range(5)],
            'label': [0, 1, 0, 1, 0]
        })
    
    def test_basic_preprocessing(self):
        """Test basic preprocessing without flux features."""
        df = self.create_sample_dataframe()
        X, y = preprocess_features(df, label_column='label', extract_flux_features=False)
        
        # Check output types
        assert isinstance(X, np.ndarray), "X should be numpy array"
        assert isinstance(y, np.ndarray), "y should be numpy array"
        
        # Check shapes
        assert X.shape[0] == len(df), "Number of samples should match"
        assert len(y) == len(df), "Number of labels should match"
        
        # Check that features are scaled (mean ≈ 0, std ≈ 1)
        assert abs(X.mean()) < 0.5, "Mean should be close to 0"
        assert abs(X.std() - 1.0) < 0.5, "Std should be close to 1"
    
    def test_flux_feature_extraction(self):
        """Test that flux features are extracted correctly."""
        df = self.create_sample_dataframe()
        X_with_flux, y = preprocess_features(df, label_column='label', extract_flux_features=True)
        X_without_flux, _ = preprocess_features(df, label_column='label', extract_flux_features=False)
        
        # With flux features should have more columns
        assert X_with_flux.shape[1] > X_without_flux.shape[1], \
            "Feature extraction should add more features"
    
    def test_categorical_encoding(self):
        """Test that categorical columns are encoded."""
        df = self.create_sample_dataframe()
        df['category'] = ['A', 'B', 'A', 'C', 'B']
        
        X, y = preprocess_features(df, label_column='label', extract_flux_features=False)
        
        # Should not raise error and encode categorical column
        assert X.shape[0] == len(df)
    
    def test_no_label_column(self):
        """Test preprocessing when label column is missing."""
        df = self.create_sample_dataframe()
        df_no_label = df.drop(columns=['label'])
        
        X, y = preprocess_features(df_no_label, label_column='label', extract_flux_features=False)
        
        # Should return empty array for labels
        assert len(y) == 0, "Should return empty array when no label column"
        assert X.shape[0] == len(df_no_label), "Should still process features"
    
    def test_custom_feature_columns(self):
        """Test using custom feature columns."""
        df = self.create_sample_dataframe()
        
        custom_features = ['orbital_period', 'planet_radius']
        X, y = preprocess_features(
            df, 
            label_column='label', 
            extract_flux_features=False,
            feature_columns=custom_features
        )
        
        # Should only use specified columns (before scaling)
        # After extraction, the number should match
        assert X.shape[0] == len(df)
    
    def test_flux_formats(self):
        """Test handling different flux data formats."""
        df = pd.DataFrame({
            'orbital_period': [10.5, 25.3, 15.7],
            'transit_duration': [2.5, 3.1, 1.8],
            'planet_radius': [1.2, 1.4, 1.8],
            'stellar_temp': [5778, 6200, 5500],
            'flux': [
                [1, 2, 3, 4, 5],  # List
                np.array([6, 7, 8, 9, 10]),  # Numpy array
                "11,12,13,14,15"  # String
            ],
            'label': [0, 1, 0]
        })
        
        # Should handle different formats without error
        X, y = preprocess_features(df, label_column='label', extract_flux_features=True)
        assert X.shape[0] == len(df)


class TestIntegration:
    """Integration tests combining both functions."""
    
    def test_full_pipeline(self):
        """Test the complete pipeline from CSV to preprocessed features."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filename = f.name
        
        try:
            # Create sample CSV
            np.random.seed(42)
            data = {
                'Orbital Period': np.random.uniform(5, 50, 20),
                'Transit Duration': np.random.uniform(1, 5, 20),
                'Planet Radius': np.random.uniform(0.5, 2.5, 20),
                'Stellar Temp': np.random.uniform(4000, 7000, 20),
                'flux': [np.random.randn(30).tolist() for _ in range(20)],
                'label': np.random.choice([0, 1], 20)
            }
            
            # Add some missing values
            data['Orbital Period'][0] = np.nan
            data['Planet Radius'][1] = np.nan
            
            pd.DataFrame(data).to_csv(filename, index=False)
            
            # Load and preprocess
            df = load_exoplanet_csv(filename)
            X, y = preprocess_features(df, label_column='label', extract_flux_features=True)
            
            # Verify output
            assert X.shape[0] == len(df), "Should process all rows"
            assert len(y) == len(df), "Should have labels for all rows"
            assert X.shape[1] > 4, "Should have features from base columns + flux"
            
            # Check scaling
            assert abs(X.mean()) < 0.5, "Features should be scaled"
            
        finally:
            if os.path.exists(filename):
                os.remove(filename)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

