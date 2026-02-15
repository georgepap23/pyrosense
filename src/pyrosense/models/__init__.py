"""
Machine Learning Models
=======================

Stacking ensemble and classifier utilities for wildfire prediction.

The main component is StackingEnsemble which combines features from
multiple sources (Prithvi, Weather, AlphaEarth) using a two-level
stacking architecture:

1. Base models: One model per feature source
2. Meta-learner: Combines base model predictions

This architecture allows each source to contribute its signal while
the meta-learner learns optimal weighting.
"""

from pyrosense.models.stacking import (
    StackingEnsemble,
    StackingConfig,
    FeatureGroup,
)
from pyrosense.models.classifiers import (
    ClassifierConfig,
    create_classifier,
    get_default_classifier,
)

__all__ = [
    "StackingEnsemble",
    "StackingConfig",
    "FeatureGroup",
    "ClassifierConfig",
    "create_classifier",
    "get_default_classifier",
]
