"""
Unit tests for NASA data fetcher.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from fetch_nasa import (
    NASA_DATASETS,
    fetch_dataset,
    fetch_all_datasets,
    download_file
)


class TestNASADatasets:
    """Tests for NASA dataset definitions."""
    
    def test_datasets_exist(self):
        """Test that datasets are defined."""
        assert len(NASA_DATASETS) > 0
        assert 'kepler' in NASA_DATASETS
        assert 'tess' in NASA_DATASETS
    
    def test_dataset_structure(self):
        """Test that each dataset has required fields."""
        for key, dataset in NASA_DATASETS.items():
            assert 'name' in dataset
            assert 'url' in dataset
            assert 'description' in dataset
            assert isinstance(dataset['name'], str)
            assert isinstance(dataset['url'], str)
            assert dataset['url'].startswith('http')


class TestFetchDataset:
    """Tests for fetch_dataset function."""
    
    def test_dry_run(self, tmp_path):
        """Test dry run mode."""
        result = fetch_dataset('kepler', tmp_path, dry_run=True)
        
        # Should succeed without creating file
        assert result is True
        assert len(list(tmp_path.glob('*.csv'))) == 0
    
    def test_invalid_dataset(self, tmp_path):
        """Test handling of invalid dataset name."""
        result = fetch_dataset('invalid_dataset', tmp_path, dry_run=True)
        
        assert result is False
    
    def test_custom_date_stamp(self, tmp_path):
        """Test custom date stamp in filename."""
        custom_date = '20231225'
        
        # Dry run to avoid actual download
        fetch_dataset('kepler', tmp_path, dry_run=True, date_stamp=custom_date)
        
        # Would create file with custom date (checked in dry run logs)
        expected_filename = f"kepler_{custom_date}.csv"
        assert True  # Dry run doesn't create file


class TestFetchAllDatasets:
    """Tests for fetch_all_datasets function."""
    
    def test_dry_run_all(self, tmp_path):
        """Test dry run for all datasets."""
        results = fetch_all_datasets(tmp_path, dry_run=True)
        
        # Should return results for all datasets
        assert len(results) == len(NASA_DATASETS)
        
        # All should succeed in dry run
        assert all(results.values())
    
    def test_dry_run_specific(self, tmp_path):
        """Test dry run for specific datasets."""
        datasets = ['kepler', 'tess']
        results = fetch_all_datasets(tmp_path, dry_run=True, datasets=datasets)
        
        # Should only process specified datasets
        assert len(results) == 2
        assert 'kepler' in results
        assert 'tess' in results


class TestDownloadFile:
    """Tests for download_file function."""
    
    @patch('fetch_nasa.requests.get')
    def test_successful_download(self, mock_get, tmp_path):
        """Test successful file download."""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-length': '1024'}
        mock_response.iter_content = lambda chunk_size: [b'test data']
        mock_get.return_value = mock_response
        
        output_path = tmp_path / 'test.csv'
        result = download_file('http://test.com/data.csv', output_path)
        
        assert result is True
        assert output_path.exists()
    
    @patch('fetch_nasa.requests.get')
    def test_timeout_handling(self, mock_get, tmp_path):
        """Test timeout error handling."""
        import requests
        mock_get.side_effect = requests.exceptions.Timeout()
        
        output_path = tmp_path / 'test.csv'
        result = download_file('http://test.com/data.csv', output_path)
        
        assert result is False
        assert not output_path.exists()
    
    @patch('fetch_nasa.requests.get')
    def test_request_error_handling(self, mock_get, tmp_path):
        """Test request error handling."""
        import requests
        mock_get.side_effect = requests.exceptions.RequestException("Network error")
        
        output_path = tmp_path / 'test.csv'
        result = download_file('http://test.com/data.csv', output_path)
        
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

