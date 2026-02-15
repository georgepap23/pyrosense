"""
Classifier Factory
==================

Factory functions and configurations for creating scikit-learn classifiers.

Supported classifiers:
- Random Forest (rf)
- XGBoost (xgb) - requires xgboost package
- Logistic Regression (lr)
- Gradient Boosting (gb)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Any
from loguru import logger

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


# Check for XGBoost availability
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


ClassifierType = Literal["rf", "xgb", "lr", "gb"]


@dataclass
class ClassifierConfig:
    """
    Configuration for creating a classifier.

    Attributes:
        classifier_type: Type of classifier ("rf", "xgb", "lr", "gb")
        n_estimators: Number of trees for ensemble methods
        max_depth: Maximum tree depth (None for unlimited)
        learning_rate: Learning rate for boosting methods
        random_state: Random seed for reproducibility
        n_jobs: Number of parallel jobs (-1 for all cores)
        extra_params: Additional classifier-specific parameters
    """
    classifier_type: ClassifierType = "rf"
    n_estimators: int = 200
    max_depth: int | None = 15
    learning_rate: float = 0.1
    random_state: int = 42
    n_jobs: int = -1
    extra_params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def random_forest(
        cls,
        n_estimators: int = 200,
        max_depth: int | None = 15,
        **kwargs,
    ) -> ClassifierConfig:
        """Create a Random Forest configuration."""
        return cls(
            classifier_type="rf",
            n_estimators=n_estimators,
            max_depth=max_depth,
            **kwargs,
        )

    @classmethod
    def xgboost(
        cls,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        **kwargs,
    ) -> ClassifierConfig:
        """Create an XGBoost configuration."""
        return cls(
            classifier_type="xgb",
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            **kwargs,
        )

    @classmethod
    def logistic_regression(
        cls,
        max_iter: int = 1000,
        **kwargs,
    ) -> ClassifierConfig:
        """Create a Logistic Regression configuration."""
        return cls(
            classifier_type="lr",
            extra_params={"max_iter": max_iter},
            **kwargs,
        )

    @classmethod
    def gradient_boosting(
        cls,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        **kwargs,
    ) -> ClassifierConfig:
        """Create a Gradient Boosting configuration."""
        return cls(
            classifier_type="gb",
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            **kwargs,
        )


def create_classifier(config: ClassifierConfig) -> BaseEstimator:
    """
    Create a classifier from configuration.

    Args:
        config: ClassifierConfig specifying the classifier type and parameters

    Returns:
        Instantiated scikit-learn classifier
    """
    if config.classifier_type == "rf":
        return RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            random_state=config.random_state,
            n_jobs=config.n_jobs,
            **config.extra_params,
        )

    elif config.classifier_type == "xgb":
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available, falling back to GradientBoosting")
            return create_classifier(
                ClassifierConfig.gradient_boosting(
                    n_estimators=config.n_estimators,
                    max_depth=config.max_depth or 6,
                    learning_rate=config.learning_rate,
                    random_state=config.random_state,
                )
            )

        return XGBClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth or 6,
            learning_rate=config.learning_rate,
            random_state=config.random_state,
            n_jobs=config.n_jobs,
            use_label_encoder=False,
            eval_metric="logloss",
            **config.extra_params,
        )

    elif config.classifier_type == "lr":
        return LogisticRegression(
            random_state=config.random_state,
            n_jobs=config.n_jobs,
            max_iter=config.extra_params.get("max_iter", 1000),
            **{k: v for k, v in config.extra_params.items() if k != "max_iter"},
        )

    elif config.classifier_type == "gb":
        return GradientBoostingClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth or 5,
            learning_rate=config.learning_rate,
            random_state=config.random_state,
            **config.extra_params,
        )

    else:
        raise ValueError(f"Unknown classifier type: {config.classifier_type}")


def get_default_classifier(
    for_source: str | None = None,
) -> BaseEstimator:
    """
    Get a default classifier, optionally optimized for a specific feature source.

    Args:
        for_source: Optional source name to optimize for ("prithvi", "weather", etc.)

    Returns:
        Default classifier instance
    """
    if for_source == "prithvi":
        # Prithvi has 1024 features, RF works well
        return create_classifier(
            ClassifierConfig.random_forest(n_estimators=100, max_depth=15)
        )

    elif for_source == "weather":
        # Weather has few features, simpler model
        return create_classifier(
            ClassifierConfig.random_forest(n_estimators=50, max_depth=10)
        )

    elif for_source == "alphaearth":
        # AlphaEarth has 64 features
        return create_classifier(
            ClassifierConfig.random_forest(n_estimators=100, max_depth=15)
        )

    else:
        # General default
        return create_classifier(ClassifierConfig.random_forest())
