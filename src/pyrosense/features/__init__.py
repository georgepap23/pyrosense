"""
Feature Extraction
==================

Feature extractors for multi-source wildfire prediction.

Extractors:
- PrithviExtractor: 1024-dim features from Prithvi-EO-2.0 foundation model
- WeatherExtractor: Weather features from Open-Meteo API
- AlphaEarthExtractor: 64-dim embeddings from Google Earth Engine

All extractors inherit from BaseFeatureExtractor and follow a common interface.
"""

from pyrosense.features.base import (
    BaseFeatureExtractor,
    FeatureResult,
    FEATURE_REGISTRY,
    get_extractor,
)
from pyrosense.features.prithvi import PrithviExtractor, MultiTemporalPrithviExtractor
from pyrosense.features.weather import WeatherExtractor
from pyrosense.features.store import FeatureStore

__all__ = [
    # Base
    "BaseFeatureExtractor",
    "FeatureResult",
    "FEATURE_REGISTRY",
    "get_extractor",
    # Extractors
    "PrithviExtractor",
    "MultiTemporalPrithviExtractor",
    "WeatherExtractor",
    # Storage
    "FeatureStore",
]

# Attempt to register AlphaEarth if earthengine-api is available
try:
    from pyrosense.features.alphaearth import AlphaEarthExtractor
    __all__.append("AlphaEarthExtractor")
except ImportError:
    pass  # earthengine-api not installed
