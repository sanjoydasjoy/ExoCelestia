# NASA Exoplanet Data Fetcher Guide

## Overview

`ml/src/fetch_nasa.py` downloads CSV datasets from NASA Exoplanet Archive with date-stamped filenames.

## Quick Start

```bash
cd ml/src

# Download all datasets
python fetch_nasa.py

# Download specific dataset
python fetch_nasa.py --dataset kepler

# Preview without downloading
python fetch_nasa.py --dry-run

# List available datasets
python fetch_nasa.py --list
```

## Available Datasets

| Dataset | Description |
|---------|-------------|
| `kepler` | Confirmed exoplanets from Kepler mission |
| `kepler_koi` | All Kepler Objects of Interest (candidates) |
| `k2` | K2 mission planet candidates |
| `tess` | TESS Objects of Interest (TOI) |
| `confirmed` | All confirmed exoplanets from all missions |

## Usage Examples

### Download All Datasets

```bash
python fetch_nasa.py
```

Output files:
```
ml/data/raw/
├── kepler_20241007.csv
├── kepler_koi_20241007.csv
├── k2_20241007.csv
├── tess_20241007.csv
└── confirmed_20241007.csv
```

### Download Specific Datasets

```bash
# Single dataset
python fetch_nasa.py --dataset kepler

# Multiple datasets
python fetch_nasa.py --dataset kepler --dataset tess
```

### Dry Run (Preview)

```bash
python fetch_nasa.py --dry-run
```

Shows what would be downloaded without actually downloading.

### Custom Output Directory

```bash
python fetch_nasa.py --output-dir /custom/path/data
```

### Verbose Logging

```bash
python fetch_nasa.py --verbose
```

## Command Line Arguments

```
--dataset DATASET     Specific dataset(s) to download (repeatable)
--output-dir DIR      Output directory (default: ../data/raw)
--dry-run            Preview without downloading
--list               List available datasets and exit
--verbose            Enable verbose logging
```

## Features

### Date-Stamped Filenames

Files are automatically named with current date:
- `kepler_20241007.csv`
- `tess_20241007.csv`

This prevents accidental overwrites and tracks when data was downloaded.

### Progress Tracking

Shows download progress:
```
2024-10-07 14:23:01 - INFO - Downloading from: https://...
2024-10-07 14:23:02 - INFO - Progress: 25.3% (2.5 MB)
2024-10-07 14:23:03 - INFO - Progress: 50.1% (5.0 MB)
2024-10-07 14:23:04 - INFO - ✓ Downloaded successfully: 9.87 MB
```

### Error Handling

- Network timeouts (30s timeout)
- HTTP errors (404, 500, etc.)
- File permission errors
- Graceful interruption (Ctrl+C)

### Overwrite Protection

If file exists, asks for confirmation:
```
File already exists: ml/data/raw/kepler_20241007.csv
Overwrite? (y/N):
```

## Integration with Data Pipeline

### 1. Download Data

```bash
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
python train.py ../data/raw/kepler_20241007.csv \
    --model random_forest \
    --output ../models/kepler_model.pkl
```

## Data Sources

All data from [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/):

- **Kepler**: Primary mission (2009-2013)
- **K2**: Extended mission (2014-2018)
- **TESS**: Current mission (2018-present)

### API Endpoints

The script uses NASA's TAP (Table Access Protocol) and API endpoints:

```python
# Example URLs
'https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+ps&format=csv'
'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=cumulative&format=csv'
```

## File Sizes (Approximate)

| Dataset | Size | Rows |
|---------|------|------|
| kepler | ~2 MB | ~3,000 |
| kepler_koi | ~15 MB | ~10,000 |
| k2 | ~5 MB | ~1,000 |
| tess | ~8 MB | ~7,000 |
| confirmed | ~10 MB | ~5,000 |

**Note**: Sizes vary as new planets are discovered and confirmed.

## Troubleshooting

### Import Error: requests

```bash
pip install requests
```

### Timeout Errors

Increase timeout in code or retry:
```python
response = requests.get(url, stream=True, timeout=60)  # 60 seconds
```

### Permission Denied

```bash
# Fix directory permissions
chmod 755 ml/data/raw

# Or use custom directory
python fetch_nasa.py --output-dir ~/Downloads
```

### Network Issues

- Check internet connection
- Check if NASA Exoplanet Archive is accessible
- Try downloading a single dataset first
- Use `--verbose` for detailed error logs

## Automation

### Download Daily (Cron)

```bash
# Add to crontab (daily at 2 AM)
0 2 * * * cd /path/to/ml/src && python fetch_nasa.py --dataset tess
```

### Python Script

```python
from fetch_nasa import fetch_all_datasets
from pathlib import Path

output_dir = Path('../data/raw')
results = fetch_all_datasets(output_dir, datasets=['kepler', 'tess'])

if all(results.values()):
    print("All downloads successful!")
else:
    print(f"Some downloads failed: {results}")
```

### CI/CD Integration

Add to GitHub Actions:

```yaml
- name: Download NASA data
  run: |
    cd ml/src
    python fetch_nasa.py --dataset kepler --dataset tess
```

## Data Update Frequency

NASA Exoplanet Archive updates:
- **Monthly**: Regular updates with new confirmations
- **Quarterly**: Major data releases
- **Ad-hoc**: Special announcements

Recommended download frequency:
- **TESS**: Weekly (active mission)
- **Kepler/K2**: Monthly (legacy data)

## Best Practices

1. **Version data**: Date stamps track when data was downloaded
2. **Validate after download**: Check row counts, column names
3. **Archive old data**: Keep previous versions for comparison
4. **Document source**: Note which URL/date data came from
5. **Check for updates**: NASA announces major updates

## Example Workflow

```bash
# 1. List available datasets
python fetch_nasa.py --list

# 2. Preview download
python fetch_nasa.py --dataset kepler --dry-run

# 3. Download
python fetch_nasa.py --dataset kepler

# 4. Verify
ls -lh ../data/raw/kepler_*.csv

# 5. Load and explore
python -c "
from data_loader import load_exoplanet_csv
df = load_exoplanet_csv('../data/raw/kepler_20241007.csv')
print(df.info())
print(df.head())
"
```

## Resources

- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
- [API Documentation](https://exoplanetarchive.ipac.caltech.edu/docs/program_interfaces.html)
- [Data Columns Guide](https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html)
- [Kepler Mission](https://www.nasa.gov/mission_pages/kepler/)
- [TESS Mission](https://www.nasa.gov/tess-transiting-exoplanet-survey-satellite/)

