# NASA Data Fetcher Implementation

## Overview

Created `ml/src/fetch_nasa.py` - A command-line tool to download exoplanet datasets from NASA Exoplanet Archive with date-stamped filenames and comprehensive logging.

## ✅ Files Created

1. **`ml/src/fetch_nasa.py`** - Main data fetcher script (400+ lines)
2. **`ml/tests/test_fetch_nasa.py`** - Unit tests
3. **`ml/data/raw/.gitkeep`** - Placeholder for raw data directory
4. **`ml/FETCH_NASA_GUIDE.md`** - User documentation
5. **`ml/FETCH_NASA_IMPLEMENTATION.md`** - This file
6. **`.gitignore`** - Updated to ignore downloaded CSV files
7. **`ml/requirements.txt`** - Added `requests==2.31.0`

## Features Implemented

### 1. Multiple NASA Datasets

```python
NASA_DATASETS = {
    'kepler': 'Confirmed exoplanets from Kepler mission',
    'kepler_koi': 'All Kepler Objects of Interest (candidates)',
    'k2': 'K2 mission planet candidates',
    'tess': 'TESS Objects of Interest (TOI)',
    'confirmed': 'All confirmed exoplanets from all missions'
}
```

### 2. Date-Stamped Filenames

Files automatically named with current date:
- `kepler_20241007.csv`
- `tess_20241007.csv`

Format: `{dataset}_{YYYYMMDD}.csv`

### 3. Command-Line Interface

```bash
python fetch_nasa.py                      # Download all datasets
python fetch_nasa.py --dataset kepler     # Specific dataset
python fetch_nasa.py --dry-run            # Preview only
python fetch_nasa.py --list               # List datasets
python fetch_nasa.py --output-dir /path   # Custom directory
```

### 4. Progress Tracking

```
2024-10-07 14:23:01 - INFO - Downloading from: https://...
2024-10-07 14:23:02 - INFO - Progress: 25.3% (2.5 MB)
2024-10-07 14:23:04 - INFO - ✓ Downloaded successfully: 9.87 MB
```

### 5. Dry Run Mode

Preview downloads without actually downloading:
```bash
python fetch_nasa.py --dry-run
```

Output:
```
[DRY RUN] Would download from:
  URL: https://exoplanetarchive.ipac.caltech.edu/...
  To: ml/data/raw/kepler_20241007.csv
```

### 6. Error Handling

- ✅ Network timeouts (30s timeout)
- ✅ HTTP errors (404, 500, etc.)
- ✅ File permission errors
- ✅ Overwrite protection (asks confirmation)
- ✅ Graceful Ctrl+C interruption

### 7. Logging

Comprehensive logging with timestamps:
```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
```

### 8. Summary Reports

After download:
```
======================================================================
DOWNLOAD SUMMARY
======================================================================
✓ SUCCESS    - kepler
✓ SUCCESS    - tess
✗ FAILED     - k2
======================================================================
Total: 3 | Successful: 2 | Failed: 1
======================================================================
```

## Usage Examples

### Download All Datasets

```bash
cd ml/src
python fetch_nasa.py
```

### Download Specific Dataset

```bash
python fetch_nasa.py --dataset kepler
```

### Download Multiple Datasets

```bash
python fetch_nasa.py --dataset kepler --dataset tess
```

### List Available Datasets

```bash
python fetch_nasa.py --list
```

Output:
```
Available NASA Exoplanet Archive Datasets:
======================================================================

kepler
  Name: Kepler Confirmed Planets
  Description: Confirmed exoplanets from Kepler mission
  URL: https://exoplanetarchive.ipac.caltech.edu/...

tess
  Name: TESS Candidates
  Description: TESS Objects of Interest (TOI)
  URL: https://exoplanetarchive.ipac.caltech.edu/...
...
```

## Integration with Pipeline

### 1. Download Data

```bash
cd ml/src
python fetch_nasa.py --dataset kepler
```

### 2. Load and Preprocess

```python
from data_loader import load_exoplanet_csv
from preprocess import preprocess_features

# Load downloaded data
df = load_exoplanet_csv('../data/raw/kepler_20241007.csv')

# Preprocess
X, y = preprocess_features(df, label_column='koi_disposition')
```

### 3. Train Model

```bash
cd ml/src
python train.py ../data/raw/kepler_20241007.csv \
    --model random_forest \
    --output ../models/kepler_rf.pkl
```

## NASA Data Sources

All data from [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/):

### APIs Used

**TAP (Table Access Protocol)**:
```
https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=...&format=csv
```

**NStED API**:
```
https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=...&format=csv
```

### Approximate Dataset Sizes

| Dataset | Size | Rows | Update Frequency |
|---------|------|------|------------------|
| kepler | ~2 MB | ~3,000 | Monthly |
| kepler_koi | ~15 MB | ~10,000 | Monthly |
| k2 | ~5 MB | ~1,000 | Monthly |
| tess | ~8 MB | ~7,000 | Weekly |
| confirmed | ~10 MB | ~5,000 | Monthly |

## Command-Line Arguments

```
positional arguments:
  None

optional arguments:
  --dataset DATASET     Dataset to download (repeatable)
                       Choices: kepler, kepler_koi, k2, tess, confirmed
  
  --output-dir DIR      Output directory
                       Default: ../data/raw
  
  --dry-run            Preview without downloading
  
  --list               List available datasets and exit
  
  --verbose            Enable verbose logging
```

## Code Structure

### Main Functions

```python
def download_file(url, output_path, chunk_size=8192) -> bool
    """Download file with progress tracking."""

def fetch_dataset(dataset_key, output_dir, dry_run=False, date_stamp=None) -> bool
    """Fetch single dataset."""

def fetch_all_datasets(output_dir, dry_run=False, datasets=None) -> Dict[str, bool]
    """Fetch multiple datasets."""

def print_summary(results: Dict[str, bool])
    """Print download summary."""
```

### Error Handling

```python
try:
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
except requests.exceptions.Timeout:
    logger.error(f"✗ Timeout while downloading")
except requests.exceptions.RequestException as e:
    logger.error(f"✗ Error: {str(e)}")
```

## Testing

### Unit Tests

```bash
cd ml
pytest tests/test_fetch_nasa.py -v
```

Tests cover:
- Dataset definitions
- Dry run mode
- Invalid dataset handling
- Download success/failure
- Network error handling
- Timeout handling

### Manual Testing

```bash
# Test dry run
python fetch_nasa.py --dry-run

# Test list
python fetch_nasa.py --list

# Test single download
python fetch_nasa.py --dataset kepler --verbose
```

## Automation Examples

### Daily TESS Download (Cron)

```bash
# Add to crontab
0 2 * * * cd /path/to/ml/src && python fetch_nasa.py --dataset tess
```

### Python Script

```python
from fetch_nasa import fetch_all_datasets
from pathlib import Path

output_dir = Path('../data/raw')
results = fetch_all_datasets(
    output_dir,
    datasets=['kepler', 'tess']
)

if all(results.values()):
    print("✓ All downloads successful")
    # Continue with preprocessing...
```

### CI/CD Integration

```yaml
# .github/workflows/update-data.yml
- name: Download latest TESS data
  run: |
    cd ml/src
    python fetch_nasa.py --dataset tess
```

## Best Practices

### 1. Version Control

Date stamps track data versions:
```
kepler_20241001.csv  # October 1st download
kepler_20241007.csv  # October 7th download (new confirmations)
```

### 2. Validation

After download, validate:
```python
import pandas as pd

df = pd.read_csv('kepler_20241007.csv')
print(f"Rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
assert len(df) > 0, "Empty dataset"
```

### 3. Archiving

Keep previous versions:
```bash
# Archive old data
mkdir -p ml/data/archive
mv ml/data/raw/*_20241001.csv ml/data/archive/
```

### 4. Documentation

Track data source:
```python
metadata = {
    'download_date': '2024-10-07',
    'source': 'NASA Exoplanet Archive',
    'dataset': 'kepler',
    'url': NASA_DATASETS['kepler']['url'],
    'rows': len(df)
}
```

## Troubleshooting

### requests Not Found

```bash
pip install requests
```

### Network Timeout

Increase timeout in code:
```python
response = requests.get(url, stream=True, timeout=60)  # 60 seconds
```

### Permission Denied

```bash
chmod 755 ml/data/raw
# Or use custom directory
python fetch_nasa.py --output-dir ~/Downloads
```

### Large Files

For very large downloads, increase chunk size:
```python
download_file(url, path, chunk_size=16384)  # 16 KB chunks
```

## Future Enhancements

Potential improvements (not implemented):

- [ ] Resume interrupted downloads
- [ ] Parallel downloads for multiple datasets
- [ ] Compression (download as .csv.gz)
- [ ] Checksum verification
- [ ] Incremental updates (download only new rows)
- [ ] Database storage (SQLite/PostgreSQL)
- [ ] Notification on completion (email/Slack)
- [ ] Automatic retries with exponential backoff

## Performance

### Download Times (Approximate)

| Dataset | Size | Time (10 Mbps) | Time (100 Mbps) |
|---------|------|----------------|-----------------|
| kepler | 2 MB | ~2s | <1s |
| kepler_koi | 15 MB | ~12s | ~2s |
| k2 | 5 MB | ~4s | ~1s |
| tess | 8 MB | ~6s | ~1s |
| confirmed | 10 MB | ~8s | ~1s |

### Bandwidth Usage

Downloading all datasets: ~40 MB total

## Security Considerations

✅ **HTTPS Only**: All URLs use HTTPS  
✅ **Timeouts**: 30s timeout prevents hanging  
✅ **No Authentication**: Public NASA data, no credentials needed  
✅ **Path Validation**: Output paths are validated  
✅ **Overwrite Protection**: Asks before overwriting  

## Commit Message

```
feat(ml): add NASA Exoplanet Archive data fetcher with date-stamped downloads

• ml/src/fetch_nasa.py
  - CLI tool to download Kepler/K2/TESS datasets
  - 5 dataset sources (kepler, kepler_koi, k2, tess, confirmed)
  - Date-stamped filenames (dataset_YYYYMMDD.csv)
  - Dry-run mode, progress tracking, error handling
  - Overwrite protection, logging, summary reports

• ml/tests/test_fetch_nasa.py
  - Unit tests for download logic
  - Mock network requests
  - Error handling tests

• Documentation
  - ml/FETCH_NASA_GUIDE.md: usage guide
  - ml/FETCH_NASA_IMPLEMENTATION.md: technical details

• Dependencies
  - Added requests==2.31.0 to requirements.txt
  - Updated .gitignore for downloaded CSVs

Usage: python ml/src/fetch_nasa.py --dataset kepler --dry-run
```

