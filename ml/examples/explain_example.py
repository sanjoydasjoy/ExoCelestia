"""
Example: Using model explanations for predictions.

Demonstrates how to use explain.py to understand model predictions.
"""

import sys
from pathlib import Path
import numpy as np
import pickle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from explain import (
    explain_prediction,
    explain_batch_predictions,
    get_global_feature_importance
)
from sklearn.ensemble import RandomForestClassifier


def create_sample_model():
    """Create and train a sample model."""
    print("Training sample model...")
    
    np.random.seed(42)
    n_samples = 500
    n_features = 10
    
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    print(f"Model trained on {n_samples} samples")
    return model


def example_single_prediction():
    """Example 1: Explain a single prediction."""
    print("\n" + "=" * 80)
    print("Example 1: Single Prediction Explanation")
    print("=" * 80)
    
    # Create model
    model = create_sample_model()
    
    # Create sample
    np.random.seed(123)
    X_sample = np.random.randn(1, 10)
    
    # Define feature names (simulating exoplanet features)
    feature_names = [
        'orbital_period',
        'transit_duration',
        'planet_radius',
        'stellar_temp',
        'flux_mean',
        'flux_std',
        'flux_median',
        'flux_min',
        'flux_max',
        'flux_range'
    ]
    
    # Make prediction
    prediction = model.predict(X_sample)[0]
    confidence = model.predict_proba(X_sample)[0][1]
    
    print(f"\nPrediction: {'Exoplanet' if prediction == 1 else 'No Exoplanet'}")
    print(f"Confidence: {confidence:.2%}")
    
    # Get explanation
    print("\nGetting explanation...")
    explanation = explain_prediction(
        model,
        X_sample,
        feature_names,
        method='auto',
        top_k=5
    )
    
    print(f"\nExplanation Method: {explanation['method_used']}")
    print("\nTop 5 Contributing Features:")
    print("-" * 60)
    
    for i, feat in enumerate(explanation['top_features'], 1):
        impact_str = f"{feat['impact']:+.4f}"
        direction = "→ Exoplanet" if feat['impact'] > 0 else "→ No Exoplanet"
        bar_length = int(abs(feat['impact']) * 50)
        bar = '█' * bar_length
        
        print(f"{i}. {feat['name']:<20} {impact_str:>8}  {direction}")
        print(f"   {bar}")
    
    return model, feature_names


def example_batch_predictions(model, feature_names):
    """Example 2: Explain multiple predictions."""
    print("\n" + "=" * 80)
    print("Example 2: Batch Prediction Explanations")
    print("=" * 80)
    
    # Create batch of samples
    np.random.seed(456)
    X_batch = np.random.randn(3, 10)
    
    print(f"\nExplaining {len(X_batch)} predictions...")
    
    # Get explanations for all samples
    explanations = explain_batch_predictions(
        model,
        X_batch,
        feature_names,
        top_k=3
    )
    
    # Display results
    for i, explanation in enumerate(explanations):
        prediction = model.predict(X_batch[i:i+1])[0]
        
        print(f"\n--- Sample {i+1} ---")
        print(f"Prediction: {'Exoplanet' if prediction == 1 else 'No Exoplanet'}")
        print("Top 3 features:")
        
        for feat in explanation['top_features']:
            print(f"  • {feat['name']:<20} {feat['impact']:+.4f}")


def example_global_importance(model, feature_names):
    """Example 3: Global feature importance."""
    print("\n" + "=" * 80)
    print("Example 3: Global Feature Importance")
    print("=" * 80)
    
    print("\nGetting overall feature importance for the model...")
    
    importance = get_global_feature_importance(
        model,
        feature_names,
        top_k=10
    )
    
    print(f"\nMethod: {importance['method_used']}")
    print("\nMost Important Features (Overall):")
    print("-" * 60)
    
    max_importance = max(f['importance'] for f in importance['top_features'])
    
    for i, feat in enumerate(importance['top_features'], 1):
        # Create visual bar
        bar_length = int((feat['importance'] / max_importance) * 40)
        bar = '█' * bar_length
        
        print(f"{i:2}. {feat['name']:<20} {feat['importance']:.4f} {bar}")


def example_method_comparison(model, feature_names):
    """Example 4: Compare different explanation methods."""
    print("\n" + "=" * 80)
    print("Example 4: Comparing Explanation Methods")
    print("=" * 80)
    
    np.random.seed(789)
    X_sample = np.random.randn(1, 10)
    
    methods = ['auto', 'importance', 'permutation']
    
    print("\nComparing different explanation methods for same prediction:")
    print("-" * 60)
    
    for method in methods:
        try:
            explanation = explain_prediction(
                model,
                X_sample,
                feature_names,
                method=method,
                top_k=3
            )
            
            print(f"\nMethod: {explanation['method_used']}")
            for feat in explanation['top_features']:
                print(f"  {feat['name']:<20} {feat['impact']:+.4f}")
        
        except Exception as e:
            print(f"\nMethod '{method}' failed: {e}")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("MODEL EXPLANATION EXAMPLES")
    print("=" * 80)
    
    # Example 1: Single prediction
    model, feature_names = example_single_prediction()
    
    # Example 2: Batch predictions
    example_batch_predictions(model, feature_names)
    
    # Example 3: Global importance
    example_global_importance(model, feature_names)
    
    # Example 4: Method comparison
    example_method_comparison(model, feature_names)
    
    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETE!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  • Positive impact values push toward exoplanet prediction")
    print("  • Negative impact values push away from exoplanet prediction")
    print("  • Larger absolute values = stronger contribution")
    print("  • SHAP provides most accurate explanations (if installed)")
    print("  • Feature importances are fast but global (not sample-specific)")
    print("\n")


if __name__ == "__main__":
    main()

