# Changelog

All notable changes to PyroSense will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-02-15

### Added

- **EarthDial VLM Integration**: Vision-language model for fire area analysis
  - `EarthDialAssistant`: Analyze satellite imagery with natural language
  - CLI commands: `analyze` and `chat`
  - Automatic image cropping to 6.7km × 6.7km area matching Prithvi predictions

- **Stacking Ensemble**: Multi-source feature fusion using stacking architecture
  - Base models for each feature source (Prithvi, Weather, AlphaEarth)
  - Meta-learner for combining predictions
  - Cross-validation for generating meta-features

- **Feature Extractors**:
  - `PrithviExtractor`: 1024-dim features from Prithvi-EO-2.0 foundation model
  - `WeatherExtractor`: Weather features from Open-Meteo API
  - `AlphaEarthExtractor`: 64-dim embeddings from Google Earth Engine
  - Base class `BaseFeatureExtractor` for consistent interface

- **Feature Store**: Unified caching and persistence for extracted features
  - Parquet storage format
  - Version management
  - Multi-source combination

- **Data Loaders**:
  - `MesogeosLoader`: Fire event extraction from Mesogeos datacube
  - `HLSDownloader`: HLS satellite imagery download from NASA Earthdata

- **CLI**: Command-line interface for training, prediction, and analysis
  - `pyrosense train`: Train stacking ensemble
  - `pyrosense predict`: Make predictions
  - `pyrosense download`: Download HLS imagery
  - `pyrosense features`: Extract features
  - `pyrosense analyze`: EarthDial image analysis
  - `pyrosense chat`: Interactive EarthDial chat

- **Modern Packaging**: pyproject.toml with optional dependencies

### Dependencies

- Requires Python 3.10+
- Core: torch, numpy, pandas, scikit-learn, xarray, rasterio
- Optional: earthengine-api (AlphaEarth), transformers (EarthDial)

[Unreleased]: https://github.com/georgepap23/pyrosense/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/georgepap23/pyrosense/releases/tag/v0.1.0
