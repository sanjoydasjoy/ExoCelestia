"""
Model explanation and interpretability utilities for exoplanet detection.

Provides SHAP-based explanations with fallbacks to permutation importance
and feature importances for sklearn-compatible models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
import warnings

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")

# Scikit-learn (should be available)
from sklearn.inspection import permutation_importance

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def explain_prediction(
    model: Any,
    X_sample: np.ndarray,
    feature_names: Optional[List[str]] = None,
    method: str = 'auto',
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Explain a model's prediction using SHAP values or fallback methods.
    
    This function provides interpretable explanations for model predictions by:
    1. Using SHAP (SHapley Additive exPlanations) if available
    2. Falling back to permutation importance
    3. Using feature_importances_ for tree-based models as last resort
    
    Args:
        model: Trained sklearn-compatible model
        X_sample: Single sample or batch of samples to explain (2D array)
        feature_names: List of feature names (optional, will use indices if not provided)
        method: Explanation method ('auto', 'shap', 'permutation', 'importance')
        top_k: Number of top features to return (default: 3)
        
    Returns:
        Dictionary containing:
            - 'top_features': List of top k features with their contributions
                             [{"name": str, "impact": float}, ...]
            - 'method_used': String indicating which explanation method was used
            - 'all_features': Complete feature importance/contribution (optional)
            
    Example:
        >>> result = explain_prediction(model, X_sample, feature_names=['temp', 'radius'])
        >>> print(result['top_features'])
        [{'name': 'stellar_temp', 'impact': 0.15}, 
         {'name': 'planet_radius', 'impact': -0.08},
         {'name': 'orbital_period', 'impact': 0.05}]
    """
    logger = logging.getLogger(__name__)
    
    # Ensure X_sample is 2D
    if X_sample.ndim == 1:
        X_sample = X_sample.reshape(1, -1)
    
    n_features = X_sample.shape[1]
    
    # Generate default feature names if not provided
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Validate feature names length
    if len(feature_names) != n_features:
        logger.warning(
            f"Feature names length ({len(feature_names)}) doesn't match "
            f"sample features ({n_features}). Using default names."
        )
        feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Determine method to use
    if method == 'auto':
        if SHAP_AVAILABLE:
            method = 'shap'
        elif hasattr(model, 'feature_importances_'):
            method = 'importance'
        else:
            method = 'permutation'
    
    # Try selected method with fallbacks
    explanation = None
    method_used = None
    
    # Try SHAP
    if method == 'shap' and SHAP_AVAILABLE:
        try:
            explanation = _explain_with_shap(model, X_sample, feature_names, top_k)
            method_used = 'shap'
            logger.info("Successfully generated SHAP explanations")
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {str(e)}. Trying fallback...")
    
    # Try permutation importance
    if explanation is None and method in ['permutation', 'auto']:
        try:
            explanation = _explain_with_permutation(model, X_sample, feature_names, top_k)
            method_used = 'permutation_importance'
            logger.info("Successfully generated permutation importance explanations")
        except Exception as e:
            logger.warning(f"Permutation importance failed: {str(e)}. Trying fallback...")
    
    # Try feature importances
    if explanation is None and hasattr(model, 'feature_importances_'):
        try:
            explanation = _explain_with_feature_importance(model, feature_names, top_k)
            method_used = 'feature_importances'
            logger.info("Successfully generated feature importance explanations")
        except Exception as e:
            logger.warning(f"Feature importance failed: {str(e)}")
    
    # Final fallback: uniform weights
    if explanation is None:
        logger.warning("All explanation methods failed. Returning uniform weights.")
        explanation = _explain_with_uniform(feature_names, top_k)
        method_used = 'uniform_fallback'
    
    explanation['method_used'] = method_used
    
    return explanation


def _explain_with_shap(
    model: Any,
    X_sample: np.ndarray,
    feature_names: List[str],
    top_k: int
) -> Dict[str, Any]:
    """
    Explain predictions using SHAP values.
    
    Args:
        model: Trained model
        X_sample: Sample(s) to explain
        feature_names: List of feature names
        top_k: Number of top features to return
        
    Returns:
        Dictionary with top features and their SHAP values
    """
    # Create SHAP explainer
    # Use TreeExplainer for tree-based models, otherwise use general Explainer
    if hasattr(model, 'feature_importances_'):
        # Tree-based model (RandomForest, XGBoost, etc.)
        explainer = shap.TreeExplainer(model)
    else:
        # For other models, use kernel explainer with sample background
        # Note: This requires background data, using X_sample as approximation
        explainer = shap.Explainer(model.predict, X_sample)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_sample)
    
    # Handle different SHAP value formats
    if isinstance(shap_values, list):
        # For binary classification, take the positive class
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    
    # If multiple samples, average SHAP values
    if shap_values.ndim == 2:
        shap_values_avg = np.mean(np.abs(shap_values), axis=0)
        shap_values_signed = np.mean(shap_values, axis=0)
    else:
        shap_values_avg = np.abs(shap_values)
        shap_values_signed = shap_values
    
    # Get top k features by absolute SHAP value
    top_indices = np.argsort(shap_values_avg)[::-1][:top_k]
    
    top_features = []
    for idx in top_indices:
        top_features.append({
            'name': feature_names[idx],
            'impact': float(shap_values_signed[idx])
        })
    
    # Include all feature contributions
    all_features = {
        feature_names[i]: float(shap_values_signed[i])
        for i in range(len(feature_names))
    }
    
    return {
        'top_features': top_features,
        'all_features': all_features
    }


def _explain_with_permutation(
    model: Any,
    X_sample: np.ndarray,
    feature_names: List[str],
    top_k: int
) -> Dict[str, Any]:
    """
    Explain predictions using permutation importance.
    
    Note: This requires creating dummy y values since we're explaining predictions.
    The importance shows how much prediction changes when feature is permuted.
    
    Args:
        model: Trained model
        X_sample: Sample(s) to explain
        feature_names: List of feature names
        top_k: Number of top features to return
        
    Returns:
        Dictionary with top features and their permutation importance
    """
    # Get predictions for all samples
    if hasattr(model, 'predict_proba'):
        # For classifiers, use probability predictions
        y_pred = model.predict_proba(X_sample)[:, 1]
    else:
        y_pred = model.predict(X_sample)
    
    # Create dummy y (use predictions as target)
    # This isn't ideal but allows us to measure feature importance for this sample
    y_dummy = y_pred
    
    # Calculate permutation importance
    perm_importance = permutation_importance(
        model, X_sample, y_dummy,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )
    
    # Get importances (mean across repeats)
    importances = perm_importance.importances_mean
    
    # Get top k features
    top_indices = np.argsort(np.abs(importances))[::-1][:top_k]
    
    top_features = []
    for idx in top_indices:
        top_features.append({
            'name': feature_names[idx],
            'impact': float(importances[idx])
        })
    
    all_features = {
        feature_names[i]: float(importances[i])
        for i in range(len(feature_names))
    }
    
    return {
        'top_features': top_features,
        'all_features': all_features
    }


def _explain_with_feature_importance(
    model: Any,
    feature_names: List[str],
    top_k: int
) -> Dict[str, Any]:
    """
    Explain using model's built-in feature_importances_.
    
    This is a global explanation (not specific to the sample) but works
    well for tree-based models.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_k: Number of top features to return
        
    Returns:
        Dictionary with top features and their importances
    """
    importances = model.feature_importances_
    
    # Get top k features
    top_indices = np.argsort(importances)[::-1][:top_k]
    
    top_features = []
    for idx in top_indices:
        top_features.append({
            'name': feature_names[idx],
            'impact': float(importances[idx])
        })
    
    all_features = {
        feature_names[i]: float(importances[i])
        for i in range(len(feature_names))
    }
    
    return {
        'top_features': top_features,
        'all_features': all_features
    }


def _explain_with_uniform(
    feature_names: List[str],
    top_k: int
) -> Dict[str, Any]:
    """
    Fallback: return uniform weights when all methods fail.
    
    Args:
        feature_names: List of feature names
        top_k: Number of top features to return
        
    Returns:
        Dictionary with uniform feature importances
    """
    uniform_weight = 1.0 / len(feature_names)
    
    top_features = [
        {'name': name, 'impact': uniform_weight}
        for name in feature_names[:top_k]
    ]
    
    all_features = {name: uniform_weight for name in feature_names}
    
    return {
        'top_features': top_features,
        'all_features': all_features
    }


def explain_batch_predictions(
    model: Any,
    X_batch: np.ndarray,
    feature_names: Optional[List[str]] = None,
    method: str = 'auto',
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Explain predictions for a batch of samples.
    
    Args:
        model: Trained model
        X_batch: Batch of samples (2D array)
        feature_names: List of feature names
        method: Explanation method
        top_k: Number of top features per sample
        
    Returns:
        List of explanation dictionaries, one per sample
    """
    explanations = []
    
    for i in range(len(X_batch)):
        X_sample = X_batch[i:i+1]  # Keep 2D shape
        explanation = explain_prediction(
            model, X_sample, feature_names, method, top_k
        )
        explanations.append(explanation)
    
    return explanations


def get_global_feature_importance(
    model: Any,
    feature_names: Optional[List[str]] = None,
    top_k: int = 10
) -> Dict[str, Any]:
    """
    Get global feature importance for the entire model.
    
    This is useful for understanding which features are generally most important,
    not specific to any particular prediction.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        top_k: Number of top features to return
        
    Returns:
        Dictionary with top features and their global importance
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model does not have feature_importances_ attribute")
    
    importances = model.feature_importances_
    
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(len(importances))]
    
    # Sort by importance
    sorted_indices = np.argsort(importances)[::-1]
    
    top_features = []
    for idx in sorted_indices[:top_k]:
        top_features.append({
            'name': feature_names[idx],
            'importance': float(importances[idx])
        })
    
    return {
        'top_features': top_features,
        'method_used': 'global_feature_importances'
    }


if __name__ == "__main__":
    # Example usage
    print("Model Explanation Utilities")
    print("=" * 60)
    
    # Create dummy model and data for demonstration
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    
    # Train a simple model
    np.random.seed(42)
    X_train = np.random.randn(100, 5)
    y_train = np.random.choice([0, 1], 100)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Create sample to explain
    X_sample = np.random.randn(1, 5)
    feature_names = ['orbital_period', 'transit_duration', 'planet_radius', 
                     'stellar_temp', 'flux_mean']
    
    # Get explanation
    explanation = explain_prediction(model, X_sample, feature_names, top_k=3)
    
    print(f"\nMethod used: {explanation['method_used']}")
    print(f"\nTop {len(explanation['top_features'])} features:")
    for feat in explanation['top_features']:
        print(f"  {feat['name']}: {feat['impact']:.4f}")
    
    print("\n" + "=" * 60)
    print("Explanation utilities ready!")

