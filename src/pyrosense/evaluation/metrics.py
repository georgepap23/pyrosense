"""
Evaluation Metrics
==================

Metrics and evaluation utilities for wildfire prediction models.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
from typing import Any
from loguru import logger

from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


@dataclass
class EvaluationResult:
    """
    Container for model evaluation results.

    Attributes:
        auc: Area Under ROC Curve
        accuracy: Classification accuracy
        precision: Precision (positive predictive value)
        recall: Recall (sensitivity, true positive rate)
        f1: F1 score (harmonic mean of precision and recall)
        cv_auc_mean: Mean AUC from cross-validation
        cv_auc_std: Standard deviation of CV AUC
        confusion_matrix: 2x2 confusion matrix
        n_samples: Number of samples evaluated
        n_positive: Number of positive samples (fires)
        n_negative: Number of negative samples (no fires)
    """
    auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    cv_auc_mean: float | None = None
    cv_auc_std: float | None = None
    confusion_matrix: list[list[int]] | None = None
    n_samples: int = 0
    n_positive: int = 0
    n_negative: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def summary(self) -> str:
        """Get formatted summary string."""
        lines = [
            f"Evaluation Results (n={self.n_samples})",
            f"  Positive (fire): {self.n_positive}",
            f"  Negative (no fire): {self.n_negative}",
            f"",
            f"Metrics:",
            f"  AUC:       {self.auc:.4f}",
            f"  Accuracy:  {self.accuracy:.4f}",
            f"  Precision: {self.precision:.4f}",
            f"  Recall:    {self.recall:.4f}",
            f"  F1:        {self.f1:.4f}",
        ]

        if self.cv_auc_mean is not None:
            lines.extend([
                f"",
                f"Cross-Validation:",
                f"  AUC: {self.cv_auc_mean:.4f} ± {self.cv_auc_std:.4f}",
            ])

        if self.confusion_matrix is not None:
            cm = self.confusion_matrix
            lines.extend([
                f"",
                f"Confusion Matrix:",
                f"               Predicted",
                f"              No-Fire  Fire",
                f"  Actual No-Fire  {cm[0][0]:5d}  {cm[0][1]:5d}",
                f"  Actual Fire     {cm[1][0]:5d}  {cm[1][1]:5d}",
            ])

        return "\n".join(lines)


def evaluate_classifier(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_train: np.ndarray | None = None,
    y_train: np.ndarray | None = None,
    cv_folds: int = 5,
    random_state: int = 42,
) -> EvaluationResult:
    """
    Evaluate a fitted classifier on test data.

    Args:
        model: Fitted classifier with predict and predict_proba methods
        X_test: Test features
        y_test: Test labels
        X_train: Training features (for cross-validation, optional)
        y_train: Training labels (for cross-validation, optional)
        cv_folds: Number of CV folds
        random_state: Random seed for CV

    Returns:
        EvaluationResult with all metrics
    """
    y_test = np.asarray(y_test)

    # Get predictions
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
        if y_prob.ndim > 1:
            y_prob = y_prob[:, 1]
    else:
        y_prob = y_pred.astype(float)

    # Calculate metrics
    auc = roc_auc_score(y_test, y_prob)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Cross-validation (if training data provided)
    cv_auc_mean = None
    cv_auc_std = None

    if X_train is not None and y_train is not None:
        cv_result = cross_validate_model(
            model, X_train, y_train,
            cv_folds=cv_folds,
            random_state=random_state,
        )
        cv_auc_mean = cv_result["auc_mean"]
        cv_auc_std = cv_result["auc_std"]

    return EvaluationResult(
        auc=float(auc),
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        cv_auc_mean=cv_auc_mean,
        cv_auc_std=cv_auc_std,
        confusion_matrix=cm.tolist(),
        n_samples=len(y_test),
        n_positive=int(y_test.sum()),
        n_negative=int(len(y_test) - y_test.sum()),
    )


def cross_validate_model(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    scoring: str = "roc_auc",
    random_state: int = 42,
) -> dict[str, float]:
    """
    Perform cross-validation on a model.

    Args:
        model: Classifier to evaluate
        X: Feature matrix
        y: Labels
        cv_folds: Number of CV folds
        scoring: Metric to use for scoring
        random_state: Random seed

    Returns:
        Dict with mean and std of CV scores
    """
    from sklearn.base import clone

    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )

    scores = cross_val_score(
        clone(model), X, y,
        cv=cv,
        scoring=scoring,
    )

    logger.info(
        f"Cross-validation ({cv_folds}-fold): "
        f"{scoring}={scores.mean():.4f} ± {scores.std():.4f}"
    )

    return {
        f"{scoring}_mean": float(scores.mean()),
        f"{scoring}_std": float(scores.std()),
        f"{scoring}_scores": scores.tolist(),
        "auc_mean": float(scores.mean()),  # Alias for compatibility
        "auc_std": float(scores.std()),
    }


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str] | None = None,
) -> str:
    """
    Print sklearn classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Names for classes

    Returns:
        Classification report string
    """
    if target_names is None:
        target_names = ["No Fire", "Fire"]

    report = classification_report(y_true, y_pred, target_names=target_names)
    print(report)
    return report
