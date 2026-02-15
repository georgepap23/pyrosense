"""
PyroSense: Wildfire Prediction with Foundation Models
======================================================

PyroSense is a multi-source feature fusion framework for wildfire prediction,
combining satellite imagery features from NASA's Prithvi foundation model with
weather data and Google's AlphaEarth embeddings using stacking ensemble methods.

Quick Start
-----------
>>> from pyrosense.features import PrithviExtractor, WeatherExtractor
>>> from pyrosense.models import StackingEnsemble
>>>
>>> # Extract features
>>> prithvi = PrithviExtractor()
>>> features = prithvi.extract(image_path="data/hls/fire_0001/composite.tif")
>>>
>>> # Train stacking ensemble
>>> ensemble = StackingEnsemble.default_wildfire()
>>> ensemble.fit(X_train, y_train)

Modules
-------
- pyrosense.data: Data loading and downloading (Mesogeos, HLS)
- pyrosense.features: Feature extraction (Prithvi, Weather, AlphaEarth)
- pyrosense.models: ML models (Stacking ensemble, classifiers)
- pyrosense.evaluation: Metrics and evaluation utilities
- pyrosense.cli: Command-line interface
"""

__version__ = "0.1.0"
__author__ = "PyroSense Team"

from pyrosense.data.mesogeos_loader import FireEvent, MesogeosLoader
from pyrosense.data.hls_downloader import HLSDownloader
from pyrosense.features.prithvi import PrithviExtractor, MultiTemporalPrithviExtractor
from pyrosense.features.weather import WeatherExtractor

__all__ = [
    "__version__",
    # Data
    "FireEvent",
    "MesogeosLoader",
    "HLSDownloader",
    # Features
    "PrithviExtractor",
    "MultiTemporalPrithviExtractor",
    "WeatherExtractor",
]
