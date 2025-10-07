"""
NASA Exoplanet Archive Data Fetcher

Downloads CSV files from NASA Exoplanet Archive for Kepler, K2, and TESS missions.
Saves files with date-stamped filenames to ml/data/raw directory.

Usage:
    python fetch_nasa.py                    # Download all datasets
    python fetch_nasa.py --dataset kepler   # Download specific dataset
    python fetch_nasa.py --dry-run          # Preview without downloading
    python fetch_nasa.py --output-dir /path # Custom output directory
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import time

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("ERROR: requests library not found. Install with: pip install requests")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# NASA Exoplanet Archive URLs
NASA_DATASETS: Dict[str, Dict[str, str]] = {
    'kepler': {
        'name': 'Kepler Confirmed Planets',
        'url': 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+ps&format=csv',
        'description': 'Confirmed exoplanets from Kepler mission'
    },
    'kepler_koi': {
        'name': 'Kepler Objects of Interest (KOI)',
        'url': 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=cumulative&format=csv',
        'description': 'All Kepler Objects of Interest including candidates'
    },
    'k2': {
        'name': 'K2 Candidates',
        'url': 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=k2candidates&format=csv',
        'description': 'K2 mission planet candidates'
    },
    'tess': {
        'name': 'TESS Candidates',
        'url': 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=TOI&format=csv',
        'description': 'TESS Objects of Interest (TOI)'
    },
    'confirmed': {
        'name': 'All Confirmed Planets',
        'url': 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+pscomppars&format=csv',
        'description': 'All confirmed exoplanets from all missions'
    }
}


def download_file(url: str, output_path: Path, chunk_size: int = 8192) -> bool:
    """
    Download a file from URL with progress tracking.
    
    Args:
        url: URL to download from
        output_path: Path to save the file
        chunk_size: Size of chunks to download (bytes)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Downloading from: {url}")
        
        # Send request with timeout
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Get file size if available
        total_size = int(response.headers.get('content-length', 0))
        
        # Download file in chunks
        downloaded = 0
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Log progress every MB
                    if downloaded % (1024 * 1024) == 0:
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            logger.info(f"Progress: {progress:.1f}% ({downloaded / (1024*1024):.1f} MB)")
                        else:
                            logger.info(f"Downloaded: {downloaded / (1024*1024):.1f} MB")
        
        file_size = output_path.stat().st_size
        logger.info(f"✓ Downloaded successfully: {file_size / (1024*1024):.2f} MB")
        
        return True
        
    except requests.exceptions.Timeout:
        logger.error(f"✗ Timeout while downloading from {url}")
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"✗ Error downloading from {url}: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"✗ Unexpected error: {str(e)}")
        return False


def fetch_dataset(
    dataset_key: str,
    output_dir: Path,
    dry_run: bool = False,
    date_stamp: Optional[str] = None
) -> bool:
    """
    Fetch a single dataset from NASA Exoplanet Archive.
    
    Args:
        dataset_key: Key identifying the dataset in NASA_DATASETS
        output_dir: Directory to save the file
        dry_run: If True, preview without downloading
        date_stamp: Date stamp to use in filename (default: today)
        
    Returns:
        True if successful (or dry run), False otherwise
    """
    if dataset_key not in NASA_DATASETS:
        logger.error(f"Unknown dataset: {dataset_key}")
        logger.info(f"Available datasets: {', '.join(NASA_DATASETS.keys())}")
        return False
    
    dataset = NASA_DATASETS[dataset_key]
    
    # Generate filename with date stamp
    if date_stamp is None:
        date_stamp = datetime.now().strftime('%Y%m%d')
    
    filename = f"{dataset_key}_{date_stamp}.csv"
    output_path = output_dir / filename
    
    logger.info("=" * 70)
    logger.info(f"Dataset: {dataset['name']}")
    logger.info(f"Description: {dataset['description']}")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 70)
    
    if dry_run:
        logger.info("[DRY RUN] Would download from:")
        logger.info(f"  URL: {dataset['url']}")
        logger.info(f"  To: {output_path}")
        return True
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if file already exists
    if output_path.exists():
        logger.warning(f"File already exists: {output_path}")
        response = input("Overwrite? (y/N): ").strip().lower()
        if response != 'y':
            logger.info("Skipped.")
            return True
    
    # Download the file
    success = download_file(dataset['url'], output_path)
    
    if success:
        logger.info(f"✓ Saved to: {output_path}")
    
    return success


def fetch_all_datasets(
    output_dir: Path,
    dry_run: bool = False,
    datasets: Optional[List[str]] = None
) -> Dict[str, bool]:
    """
    Fetch multiple datasets from NASA Exoplanet Archive.
    
    Args:
        output_dir: Directory to save files
        dry_run: If True, preview without downloading
        datasets: List of dataset keys to fetch (default: all)
        
    Returns:
        Dictionary mapping dataset names to success status
    """
    if datasets is None:
        datasets = list(NASA_DATASETS.keys())
    
    # Use same date stamp for all downloads in this session
    date_stamp = datetime.now().strftime('%Y%m%d')
    
    results = {}
    total = len(datasets)
    
    logger.info(f"\nFetching {total} dataset(s)...")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Date stamp: {date_stamp}")
    
    if dry_run:
        logger.info("[DRY RUN MODE - No files will be downloaded]")
    
    logger.info("")
    
    for i, dataset_key in enumerate(datasets, 1):
        logger.info(f"\n[{i}/{total}] Processing: {dataset_key}")
        
        success = fetch_dataset(dataset_key, output_dir, dry_run, date_stamp)
        results[dataset_key] = success
        
        # Small delay between downloads to be nice to the server
        if not dry_run and i < total:
            time.sleep(1)
    
    return results


def print_summary(results: Dict[str, bool]):
    """
    Print summary of download results.
    
    Args:
        results: Dictionary mapping dataset names to success status
    """
    logger.info("\n" + "=" * 70)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 70)
    
    successful = sum(1 for success in results.values() if success)
    failed = len(results) - successful
    
    for dataset, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"{status:12} - {dataset}")
    
    logger.info("=" * 70)
    logger.info(f"Total: {len(results)} | Successful: {successful} | Failed: {failed}")
    logger.info("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Download exoplanet data from NASA Exoplanet Archive',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available datasets:
  kepler      - Confirmed exoplanets from Kepler mission
  kepler_koi  - All Kepler Objects of Interest (candidates)
  k2          - K2 mission planet candidates
  tess        - TESS Objects of Interest
  confirmed   - All confirmed exoplanets from all missions

Examples:
  python fetch_nasa.py
  python fetch_nasa.py --dataset kepler
  python fetch_nasa.py --dataset kepler --dataset tess
  python fetch_nasa.py --dry-run
  python fetch_nasa.py --output-dir /custom/path
        """
    )
    
    parser.add_argument(
        '--dataset',
        action='append',
        choices=list(NASA_DATASETS.keys()),
        help='Specific dataset(s) to download (can specify multiple times)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../data/raw',
        help='Output directory for downloaded files (default: ../data/raw)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview what would be downloaded without actually downloading'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available datasets and exit'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # List datasets and exit
    if args.list:
        print("\nAvailable NASA Exoplanet Archive Datasets:")
        print("=" * 70)
        for key, info in NASA_DATASETS.items():
            print(f"\n{key}")
            print(f"  Name: {info['name']}")
            print(f"  Description: {info['description']}")
            print(f"  URL: {info['url'][:80]}...")
        print("\n" + "=" * 70)
        sys.exit(0)
    
    # Resolve output directory
    output_dir = Path(__file__).parent / args.output_dir
    output_dir = output_dir.resolve()
    
    # Fetch datasets
    datasets = args.dataset if args.dataset else None
    
    try:
        results = fetch_all_datasets(output_dir, args.dry_run, datasets)
        print_summary(results)
        
        # Exit with error code if any downloads failed
        if not all(results.values()):
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.warning("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\nFatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

