"""Hyperparameter optimization utilities for PyroSense models."""

import warnings
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# Parameter grids for different model types and sources
RF_GRIDS = {
    'prithvi': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.1, 0.2]
    },
    'alphaearth': {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, 20],
        'max_features': ['sqrt', 0.2, 0.3],
        'min_samples_leaf': [1, 2, 4]
    },
    'weather': {
        'n_estimators': [50, 100, 150],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'max_features': ['sqrt', None]
    }
}

BOOSTING_GRID = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0]
}

# Default tuning plan (which models to tune for each source)
DEFAULT_TUNING_PLAN = {
    "prithvi": ["rf"],                  # RF is best for high-dimensional Prithvi
    "weather": ["rf", "xgb", "gb"],     # Low-dim weather benefits from boosting
    "alphaearth": ["rf", "xgb", "gb"]   # Mid-dim AlphaEarth handles both well
}


def optimize_base_models(X_train, y_train, sources, tuning_plan=None, verbose=True):
    """
    Optimize base models for each feature source using GridSearchCV.

    Args:
        X_train: Training feature DataFrame with source-prefixed columns
        y_train: Training labels
        sources: List of feature source names (e.g., ['prithvi', 'weather', 'alphaearth'])
        tuning_plan: Optional dict mapping source names to list of model types to tune.
                    If None, uses DEFAULT_TUNING_PLAN.
        verbose: Whether to print progress updates

    Returns:
        dict: {model_key: trained_model} where model_key is '{source}_{model_type}'
              e.g., {'prithvi_rf': RandomForestClassifier(...), 'weather_gb': GradientBoostingClassifier(...)}
    """
    # Suppress warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')

    # Use default plan if not provided
    if tuning_plan is None:
        tuning_plan = DEFAULT_TUNING_PLAN.copy()

    # Remove xgb from plan if not available
    if not XGBOOST_AVAILABLE:
        if verbose:
            print("XGBoost not installed. Install with: pip install xgboost")
            print("Continuing with RandomForest and GradientBoosting only...\n")
        for source in tuning_plan:
            tuning_plan[source] = [m for m in tuning_plan[source] if m != 'xgb']

    best_base_models = {}

    if verbose:
        print("Starting Hyperparameter Optimization")
        print("=" * 60)

    for source in sources:
        if source not in tuning_plan:
            continue

        models_to_tune = tuning_plan[source]

        # Isolate features for this specific source
        cols = [c for c in X_train.columns if c.startswith(f"{source}_")]

        if len(cols) == 0:
            if verbose:
                print(f"Warning: No features found for source '{source}'")
            continue

        X_source_train = X_train[cols].values

        # Preprocessing: Scale and handle NaNs
        scaler = StandardScaler()
        X_source_scaled = scaler.fit_transform(X_source_train)
        X_source_clean = np.nan_to_num(X_source_scaled, nan=0.0)

        for model_type in models_to_tune:
            if verbose:
                print(f"\nTuning {source.upper()} - {model_type.upper()}...")

            # Select estimator and the correct grid
            if model_type == 'rf':
                estimator = RandomForestClassifier(random_state=42, n_jobs=-1)
                param_grid = RF_GRIDS[source]
            elif model_type == 'xgb':
                estimator = XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1)
                param_grid = BOOSTING_GRID
            elif model_type == 'gb':
                estimator = GradientBoostingClassifier(random_state=42)
                param_grid = BOOSTING_GRID
            else:
                if verbose:
                    print(f"  Warning: Unknown model type '{model_type}', skipping")
                continue

            grid_search = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                cv=5,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )

            grid_search.fit(X_source_clean, y_train)

            # Save the winning model with a unique key
            model_key = f"{source}_{model_type}"
            best_base_models[model_key] = grid_search.best_estimator_

            if verbose:
                print(f"  -> Best CV AUC: {grid_search.best_score_:.4f}")
                print(f"  -> Best Params: {grid_search.best_params_}")

    if verbose:
        print("\n" + "=" * 60)
        print(f"Optimization Complete: {len(best_base_models)} models trained")

    return best_base_models
