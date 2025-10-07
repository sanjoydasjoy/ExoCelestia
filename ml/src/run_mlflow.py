"""
MLflow experiment tracking for exoplanet detection models.

This script provides utilities for running experiments with MLflow tracking.
It can be used standalone or the training script can be called with --use-mlflow flag.

Usage:
    # Start MLflow UI
    mlflow ui --backend-store-uri ./mlruns
    # Then open http://localhost:5000 in browser

    # Run training with MLflow
    python train.py data.csv --model random_forest --use-mlflow

    # Run multiple experiments
    python run_mlflow.py --dataset data.csv --models random_forest xgboost nn

View MLflow UI:
    1. Run: mlflow ui --backend-store-uri ./mlruns
    2. Open: http://localhost:5000
    3. Compare: experiments, metrics, parameters, models
"""

import argparse
import sys
from pathlib import Path
import subprocess
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    print("ERROR: MLflow not available. Install with: pip install mlflow")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_experiment(
    dataset_path: str,
    model_type: str,
    experiment_name: str = 'exoplanet-detection',
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Run a single training experiment with MLflow tracking.
    
    Args:
        dataset_path: Path to CSV dataset
        model_type: Model type (random_forest, xgboost, nn)
        experiment_name: MLflow experiment name
        test_size: Test set fraction
        random_state: Random seed
    """
    logger.info(f"Running experiment: {model_type}")
    
    # Build command
    cmd = [
        'python', 'train.py',
        dataset_path,
        '--model', model_type,
        '--test-size', str(test_size),
        '--random-state', str(random_state),
        '--use-mlflow',
        '--experiment-name', experiment_name,
        '--output', f'../models/{model_type}_mlflow.pkl'
    ]
    
    # Run training
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"✓ {model_type} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {model_type} failed: {e}")
        logger.error(f"Output: {e.output}")
        return False


def run_multiple_experiments(
    dataset_path: str,
    models: list,
    experiment_name: str = 'exoplanet-detection',
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Run multiple experiments in sequence.
    
    Args:
        dataset_path: Path to CSV dataset
        models: List of model types to train
        experiment_name: MLflow experiment name
        test_size: Test set fraction
        random_state: Random seed
    """
    logger.info(f"Running {len(models)} experiments")
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Dataset: {dataset_path}")
    logger.info("")
    
    results = {}
    
    for model in models:
        success = run_experiment(
            dataset_path,
            model,
            experiment_name,
            test_size,
            random_state
        )
        results[model] = success
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 70)
    
    for model, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"{status:12} - {model}")
    
    logger.info("=" * 70)
    logger.info(f"Total: {len(results)} | Successful: {sum(results.values())}")
    logger.info("=" * 70)
    
    logger.info("\nView results:")
    logger.info("  mlflow ui --backend-store-uri ./mlruns")
    logger.info("  Open: http://localhost:5000")


def start_mlflow_ui(backend_store_uri: str = './mlruns', port: int = 5000):
    """
    Start MLflow UI server.
    
    Args:
        backend_store_uri: Path to MLflow tracking directory
        port: Port to run UI on
    """
    logger.info(f"Starting MLflow UI on port {port}")
    logger.info(f"Backend store: {backend_store_uri}")
    logger.info(f"Open: http://localhost:{port}")
    logger.info("Press Ctrl+C to stop")
    
    cmd = [
        'mlflow', 'ui',
        '--backend-store-uri', backend_store_uri,
        '--port', str(port)
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        logger.info("\nMLflow UI stopped")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error starting MLflow UI: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run MLflow experiments for exoplanet detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run multiple experiments
  python run_mlflow.py --dataset data.csv --models random_forest xgboost

  # Start MLflow UI
  python run_mlflow.py --ui

  # Custom experiment name
  python run_mlflow.py --dataset data.csv --models rf --experiment my-experiment
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        help='Path to dataset CSV file'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['random_forest', 'xgboost', 'nn'],
        help='Model types to train'
    )
    
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='exoplanet-detection',
        help='MLflow experiment name (default: exoplanet-detection)'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set size (default: 0.2)'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    parser.add_argument(
        '--ui',
        action='store_true',
        help='Start MLflow UI server'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='MLflow UI port (default: 5000)'
    )
    
    args = parser.parse_args()
    
    # Start UI
    if args.ui:
        start_mlflow_ui(port=args.port)
        return
    
    # Run experiments
    if args.dataset and args.models:
        run_multiple_experiments(
            args.dataset,
            args.models,
            args.experiment_name,
            args.test_size,
            args.random_state
        )
    else:
        parser.print_help()
        print("\n" + "=" * 70)
        print("Quick Start:")
        print("=" * 70)
        print("1. Run experiments:")
        print("   python run_mlflow.py --dataset data.csv --models random_forest xgboost")
        print("")
        print("2. Start MLflow UI:")
        print("   python run_mlflow.py --ui")
        print("   OR")
        print("   mlflow ui --backend-store-uri ./mlruns")
        print("")
        print("3. Open browser:")
        print("   http://localhost:5000")
        print("=" * 70)


if __name__ == "__main__":
    main()

