"""
Evaluation Utilities
====================

Metrics and evaluation utilities for wildfire prediction models.
"""

from pyrosense.evaluation.metrics import (
    evaluate_classifier,
    cross_validate_model,
    EvaluationResult,
)

__all__ = [
    "evaluate_classifier",
    "cross_validate_model",
    "EvaluationResult",
]
