"""
Stacking Ensemble
=================

Multi-source feature fusion using stacking ensemble architecture.

The StackingEnsemble combines features from multiple sources (Prithvi, Weather,
AlphaEarth) using a two-level architecture:

Level 1 - Base Models:
    - One classifier per feature source
    - Each base model learns from its source's features
    - Cross-validation generates out-of-fold predictions

Level 2 - Meta-Learner:
    - Takes base model predictions as input
    - Learns optimal weighting of sources
    - Produces final fire/no-fire prediction

Architecture:
    Prithvi features  --> RF Classifier --> P(fire|prithvi) --+
                                                               |
    Weather features  --> RF Classifier --> P(fire|weather) --+--> Meta-Learner --> Final P(fire)
                                                               |
    AlphaEarth features --> RF Classifier --> P(fire|earth) ---+

Training Flow:
    1. For each feature group, generate out-of-fold predictions via CV
    2. Stack predictions to form meta-features
    3. Train meta-learner on stacked predictions
    4. Refit base models on full training data

Inference Flow:
    1. Each base model predicts P(fire) from its features
    2. Stack base predictions
    3. Meta-learner produces final prediction
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd
from loguru import logger

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from pyrosense.models.classifiers import (
    ClassifierConfig,
    create_classifier,
    get_default_classifier,
)


@dataclass
class FeatureGroup:
    """
    Configuration for a feature source in the stacking ensemble.

    Attributes:
        name: Identifier for this feature group (e.g., "prithvi")
        column_prefix: Prefix for columns belonging to this group
        base_model: Base classifier for this group (None = use default)
        enabled: Whether this group is included in the ensemble
    """
    name: str
    column_prefix: str
    base_model: BaseEstimator | None = None
    enabled: bool = True

    def get_columns(self, df: pd.DataFrame) -> list[str]:
        """Get columns belonging to this feature group."""
        return [col for col in df.columns if col.startswith(self.column_prefix)]

    def get_model(self) -> BaseEstimator:
        """Get the base model, using default if not specified."""
        if self.base_model is not None:
            return clone(self.base_model)
        return get_default_classifier(for_source=self.name)


@dataclass
class StackingConfig:
    """
    Configuration for the stacking ensemble.

    Attributes:
        feature_groups: List of feature group configurations
        meta_model: Meta-learner classifier (None = LogisticRegression)
        cv_folds: Number of CV folds for generating meta-features
        use_probabilities: Use probabilities (True) or predictions (False)
        scale_features: Whether to standardize features before base models
        random_state: Random seed for reproducibility
    """
    feature_groups: list[FeatureGroup] = field(default_factory=list)
    meta_model: BaseEstimator | None = None
    cv_folds: int = 5
    use_probabilities: bool = True
    scale_features: bool = True
    random_state: int = 42

    @classmethod
    def default_wildfire(cls) -> StackingConfig:
        """Create default configuration for wildfire prediction."""
        return cls(
            feature_groups=[
                FeatureGroup(
                    name="prithvi",
                    column_prefix="prithvi_",
                    base_model=create_classifier(
                        ClassifierConfig.random_forest(n_estimators=100, max_depth=15)
                    ),
                ),
                FeatureGroup(
                    name="weather",
                    column_prefix="weather_",
                    base_model=create_classifier(
                        ClassifierConfig.random_forest(n_estimators=50, max_depth=10)
                    ),
                ),
                FeatureGroup(
                    name="alphaearth",
                    column_prefix="alphaearth_",
                    base_model=create_classifier(
                        ClassifierConfig.random_forest(n_estimators=100, max_depth=15)
                    ),
                ),
            ],
            meta_model=LogisticRegression(max_iter=1000, random_state=42),
            cv_folds=5,
            use_probabilities=True,
        )

    @classmethod
    def simple(cls, sources: list[str] | None = None) -> StackingConfig:
        """
        Create configuration with best hyperparameters from grid search.

        Uses optimized settings based on 5-fold CV on PyroSense dataset:
        - Prithvi: RF (AUC=0.9111, max_depth=10, max_features=0.1)
        - Weather: GB (AUC=0.9810, max_depth=3, learning_rate=0.1)
        - AlphaEarth: RF (AUC=0.7760, max_depth=10, max_features=0.3)
        """
        from sklearn.ensemble import GradientBoostingClassifier

        if sources is None:
            sources = ["prithvi", "weather"]

        feature_groups = []

        for source in sources:
            if source == "prithvi":
                # Best: RF with max_depth=10, max_features=0.1, n_estimators=100
                base_model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    max_features=0.1,
                    min_samples_leaf=1,
                    random_state=42,
                    n_jobs=-1
                )
            elif source == "weather":
                # Best: GB with learning_rate=0.1, max_depth=3, n_estimators=200
                base_model = GradientBoostingClassifier(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.1,
                    subsample=0.8,
                    random_state=42
                )
            elif source == "alphaearth":
                # Best: RF with max_depth=10, max_features=0.3, n_estimators=200
                base_model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    max_features=0.3,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                # Fallback: default RF for unknown sources
                base_model = None

            feature_groups.append(
                FeatureGroup(name=source, column_prefix=f"{source}_", base_model=base_model)
            )

        return cls(
            feature_groups=feature_groups,
            meta_model=LogisticRegression(max_iter=1000, random_state=42),
        )

    def get_enabled_groups(self) -> list[FeatureGroup]:
        """Get only enabled feature groups."""
        return [g for g in self.feature_groups if g.enabled]


class StackingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Stacking ensemble for multi-source feature fusion.

    Training:
        1. For each feature group, train base model with cross-validation
        2. Generate out-of-fold predictions (meta-features)
        3. Train meta-learner on stacked predictions
        4. Refit base models on full data

    Inference:
        1. Get predictions from each base model
        2. Stack predictions
        3. Meta-learner produces final prediction

    Usage:
        config = StackingConfig.default_wildfire()
        ensemble = StackingEnsemble(config)
        ensemble.fit(X_train, y_train)
        probas = ensemble.predict_proba(X_test)

    Attributes:
        config: StackingConfig with feature groups and meta-model
        base_models_: Fitted base models (dict: source -> model)
        meta_model_: Fitted meta-learner
        scalers_: Feature scalers (dict: source -> StandardScaler)
        feature_columns_: Columns used per feature group
    """

    # Explicitly mark as classifier for sklearn compatibility
    _estimator_type = "classifier"

    def __init__(self, config: StackingConfig | None = None) -> None:
        """
        Args:
            config: Stacking configuration. If None, uses default_wildfire()
        """
        self.config = config or StackingConfig.default_wildfire()

        # Fitted components (set during fit)
        self.base_models_: dict[str, BaseEstimator] = {}
        self.meta_model_: BaseEstimator | None = None
        self.scalers_: dict[str, StandardScaler] = {}
        self.feature_columns_: dict[str, list[str]] = {}
        self.classes_: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> StackingEnsemble:
        """
        Fit the stacking ensemble.

        Args:
            X: Feature DataFrame with columns for all feature groups
            y: Target labels (0 = no fire, 1 = fire)

        Returns:
            self
        """
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        enabled_groups = self.config.get_enabled_groups()
        if not enabled_groups:
            raise ValueError("No feature groups enabled")

        logger.info(f"Fitting stacking ensemble with {len(enabled_groups)} feature groups")

        # Step 1: Generate out-of-fold predictions for meta-features
        meta_features = []
        meta_feature_names = []

        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )

        for group in enabled_groups:
            # Get columns for this group
            columns = group.get_columns(X)
            if not columns:
                logger.warning(f"No columns found for group '{group.name}' with prefix '{group.column_prefix}'")
                continue

            self.feature_columns_[group.name] = columns
            X_group = X[columns].values

            # Scale features if configured
            if self.config.scale_features:
                scaler = StandardScaler()
                X_group = scaler.fit_transform(X_group)
                self.scalers_[group.name] = scaler

            # Handle NaN values
            X_group = np.nan_to_num(X_group, nan=0.0)

            # Get base model
            base_model = group.get_model()

            # Generate out-of-fold predictions
            logger.info(f"  [{group.name}] Generating OOF predictions ({len(columns)} features)")

            if self.config.use_probabilities:
                oof_preds = cross_val_predict(
                    base_model, X_group, y,
                    cv=cv, method="predict_proba",
                )
                # Use probability of positive class
                if oof_preds.ndim > 1:
                    oof_preds = oof_preds[:, 1]
            else:
                oof_preds = cross_val_predict(
                    base_model, X_group, y,
                    cv=cv, method="predict",
                )

            meta_features.append(oof_preds)
            meta_feature_names.append(f"pred_{group.name}")

            # Refit on full data
            base_model.fit(X_group, y)
            self.base_models_[group.name] = base_model

        # Step 2: Stack meta-features
        if not meta_features:
            raise ValueError("No valid feature groups found")

        X_meta = np.column_stack(meta_features)
        logger.info(f"  Meta-features shape: {X_meta.shape}")

        # Step 3: Train meta-learner
        meta_model = self.config.meta_model
        if meta_model is None:
            meta_model = LogisticRegression(max_iter=1000, random_state=self.config.random_state)
        else:
            meta_model = clone(meta_model)

        logger.info(f"  Training meta-learner: {type(meta_model).__name__}")
        meta_model.fit(X_meta, y)
        self.meta_model_ = meta_model

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Feature DataFrame

        Returns:
            Predicted labels (0 or 1)
        """
        probas = self.predict_proba(X)
        return (probas[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature DataFrame

        Returns:
            Array of shape (n_samples, 2) with [P(no fire), P(fire)]
        """
        if self.meta_model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Get base model predictions
        meta_features = []

        for group in self.config.get_enabled_groups():
            if group.name not in self.base_models_:
                continue

            columns = self.feature_columns_.get(group.name, [])
            if not columns:
                continue

            X_group = X[columns].values

            # Apply scaling
            if group.name in self.scalers_:
                X_group = self.scalers_[group.name].transform(X_group)

            # Handle NaN
            X_group = np.nan_to_num(X_group, nan=0.0)

            # Get predictions from base model
            base_model = self.base_models_[group.name]

            if self.config.use_probabilities:
                preds = base_model.predict_proba(X_group)
                if preds.ndim > 1:
                    preds = preds[:, 1]
            else:
                preds = base_model.predict(X_group)

            meta_features.append(preds)

        # Stack and predict with meta-learner
        X_meta = np.column_stack(meta_features)
        return self.meta_model_.predict_proba(X_meta)

    def get_feature_importances(self) -> dict[str, dict[str, float]]:
        """
        Get feature importances from base models (if available).

        Returns:
            Dict mapping group name to feature importance dict
        """
        importances = {}

        for group_name, model in self.base_models_.items():
            if hasattr(model, "feature_importances_"):
                columns = self.feature_columns_.get(group_name, [])
                importances[group_name] = {
                    col: float(imp)
                    for col, imp in zip(columns, model.feature_importances_)
                }

        return importances

    def get_source_weights(self) -> dict[str, float]:
        """
        Get the meta-learner weights for each source.

        Returns:
            Dict mapping source name to weight (only for linear meta-learners)
        """
        if self.meta_model_ is None:
            return {}

        if not hasattr(self.meta_model_, "coef_"):
            return {}

        coefs = self.meta_model_.coef_.flatten()
        groups = [g for g in self.config.get_enabled_groups() if g.name in self.base_models_]

        if len(coefs) != len(groups):
            return {}

        return {
            group.name: float(coef)
            for group, coef in zip(groups, coefs)
        }

    # Alias for convenience
    def source_weights(self) -> dict[str, float]:
        """Alias for get_source_weights()."""
        return self.get_source_weights()

    def summary(self) -> dict[str, Any]:
        """
        Get a summary of the fitted ensemble.

        Returns:
            Dict with ensemble configuration and statistics
        """
        return {
            "n_feature_groups": len(self.base_models_),
            "feature_groups": list(self.base_models_.keys()),
            "features_per_group": {
                name: len(cols)
                for name, cols in self.feature_columns_.items()
            },
            "meta_model": type(self.meta_model_).__name__ if self.meta_model_ else None,
            "source_weights": self.get_source_weights(),
            "config": {
                "cv_folds": self.config.cv_folds,
                "use_probabilities": self.config.use_probabilities,
                "scale_features": self.config.scale_features,
            },
        }
