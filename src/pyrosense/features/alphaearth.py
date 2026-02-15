"""
AlphaEarth Feature Extractor
============================

Extracts 64-dimensional embeddings from Google's AlphaEarth satellite imagery
dataset via Google Earth Engine.

AlphaEarth provides pre-computed embeddings from Sentinel-2 imagery that capture
static land surface characteristics useful for understanding fire risk factors
like vegetation type, terrain, and land use patterns.

Requirements:
- earthengine-api package: pip install earthengine-api
- Earth Engine authentication: Run `earthengine authenticate` once

Dataset:
- Collection: GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL
- Resolution: ~10m (Sentinel-2 native)
- Bands: 64 embedding dimensions (A00-A63)
- Temporal: Annual composites
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import TYPE_CHECKING
from loguru import logger

from pyrosense.features.base import BaseFeatureExtractor, FeatureResult, register_extractor

# Lazy import of Earth Engine
EE_AVAILABLE = False
ee = None

if TYPE_CHECKING:
    from pyrosense.data.mesogeos_loader import FireEvent


def _init_earth_engine(project: str | None = None):
    """
    Initialize Earth Engine API with lazy import.

    Args:
        project: Google Cloud Project ID. Required for Earth Engine API.
                 Can also be set via GOOGLE_CLOUD_PROJECT environment variable.
    """
    global EE_AVAILABLE, ee
    if ee is not None:
        return EE_AVAILABLE

    try:
        import ee as earth_engine
        import os
        ee = earth_engine

        # Determine project ID
        project_id = project or os.environ.get("GOOGLE_CLOUD_PROJECT")

        # Try to initialize - this requires authentication
        try:
            if project_id:
                ee.Initialize(project=project_id)
                logger.info(f"Earth Engine initialized with project: {project_id}")
            else:
                ee.Initialize()
                logger.info("Earth Engine initialized successfully")
            EE_AVAILABLE = True
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Earth Engine initialization failed: {e}")
            if "project" in error_msg.lower():
                logger.info("Earth Engine requires a Cloud Project ID.")
                logger.info("Set via: ee.Initialize(project='your-project-id')")
                logger.info("Or environment variable: GOOGLE_CLOUD_PROJECT")
            else:
                logger.info("Run 'earthengine authenticate' to authenticate")
            EE_AVAILABLE = False
    except ImportError:
        logger.warning("earthengine-api not installed. Install with: pip install earthengine-api")
        EE_AVAILABLE = False

    return EE_AVAILABLE


# AlphaEarth embedding band names (64 dimensions)
ALPHAEARTH_BAND_NAMES = [f"A{i:02d}" for i in range(64)]  # A00, A01, ..., A63
ALPHAEARTH_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"


@register_extractor("alphaearth")
class AlphaEarthExtractor(BaseFeatureExtractor):
    """
    Extract 64-dim embeddings from Google AlphaEarth via Earth Engine.

    AlphaEarth embeddings capture static land surface characteristics from
    Sentinel-2 imagery, including vegetation patterns, land cover type,
    and terrain features that are relevant for wildfire risk assessment.

    Usage:
        extractor = AlphaEarthExtractor(cache_dir="data/alphaearth/")
        features = extractor.extract(event)

    Note:
        Requires Earth Engine authentication. Run:
        $ earthengine authenticate
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        buffer_meters: int = 30,
        scale: int = 10,
        project: str | None = None,
    ) -> None:
        """
        Args:
            cache_dir: Directory to cache extracted features
            buffer_meters: Buffer around point for sampling (meters)
            scale: Resolution for sampling (meters, default 10 = Sentinel-2 native)
            project: Google Cloud Project ID for Earth Engine. Can also be set
                     via GOOGLE_CLOUD_PROJECT environment variable.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.buffer_meters = buffer_meters
        self.scale = scale
        self.project = project
        self._ee_initialized = False

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_ee(self) -> bool:
        """Ensure Earth Engine is initialized."""
        if self._ee_initialized:
            return True

        self._ee_initialized = _init_earth_engine(project=self.project)
        return self._ee_initialized

    @property
    def feature_dim(self) -> int:
        """AlphaEarth embeddings have 64 dimensions."""
        return 64

    @property
    def feature_names(self) -> list[str]:
        """Feature names are alphaearth_b0 through alphaearth_b63."""
        return [f"b{i}" for i in range(64)]

    @property
    def source_name(self) -> str:
        """Source identifier for AlphaEarth features."""
        return "alphaearth"

    def _get_cache_path(self, event_id: str) -> Path | None:
        """Get cache file path for an event."""
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{event_id}.npy"

    def _load_from_cache(self, event_id: str) -> np.ndarray | None:
        """Try to load features from cache."""
        cache_path = self._get_cache_path(event_id)
        if cache_path and cache_path.exists():
            try:
                return np.load(cache_path)
            except Exception as e:
                logger.debug(f"Cache load failed for {event_id}: {e}")
        return None

    def _save_to_cache(self, event_id: str, features: np.ndarray) -> None:
        """Save features to cache."""
        cache_path = self._get_cache_path(event_id)
        if cache_path:
            try:
                np.save(cache_path, features)
            except Exception as e:
                logger.debug(f"Cache save failed for {event_id}: {e}")

    def extract(
        self,
        event: FireEvent,
        year: int | None = None,
        **kwargs,
    ) -> FeatureResult:
        """
        Extract AlphaEarth embedding for a single event.

        Args:
            event: FireEvent with location and date information
            year: Year for the annual embedding. If None, uses event year.
            **kwargs: Additional arguments (unused)

        Returns:
            FeatureResult with 64-dim embedding vector
        """
        # Check cache first
        cached = self._load_from_cache(event.event_id)
        if cached is not None:
            logger.debug(f"Loaded {event.event_id} from cache")
            return FeatureResult(
                event_id=event.event_id,
                features=cached,
                feature_names=self.feature_names,
                source=self.source_name,
                metadata={"from_cache": True},
            )

        # Initialize Earth Engine
        if not self._ensure_ee():
            logger.warning(f"Earth Engine not available for {event.event_id}")
            return self._empty_result(event.event_id)

        # Determine year for embedding
        if year is None:
            year = event.date.year

        try:
            features = self._extract_from_ee(
                event.latitude,
                event.longitude,
                year,
            )

            # Cache the result
            self._save_to_cache(event.event_id, features)

            return FeatureResult(
                event_id=event.event_id,
                features=features,
                feature_names=self.feature_names,
                source=self.source_name,
                metadata={
                    "year": year,
                    "latitude": event.latitude,
                    "longitude": event.longitude,
                },
            )

        except Exception as e:
            logger.warning(f"AlphaEarth extraction failed for {event.event_id}: {e}")
            return self._empty_result(event.event_id)

    def _extract_from_ee(
        self,
        lat: float,
        lon: float,
        year: int,
    ) -> np.ndarray:
        """
        Extract embedding from Earth Engine.

        Args:
            lat: Latitude
            lon: Longitude
            year: Year for annual embedding

        Returns:
            64-dimensional embedding as numpy array
        """
        # Create point geometry
        point = ee.Geometry.Point([lon, lat])

        # Get the annual embedding collection
        collection = ee.ImageCollection(ALPHAEARTH_COLLECTION)

        # Filter to the specific year
        # AlphaEarth data is available from 2017 onwards
        if year < 2017:
            raise ValueError(f"No AlphaEarth data available for year {year} (data starts from 2017)")

        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"

        # IMPORTANT: AlphaEarth is tiled, so we must filter by bounds first
        # to get the tile covering our point, not just any tile from that year
        filtered = collection.filterBounds(point).filterDate(start_date, end_date)

        # Check if collection has any images
        count = filtered.size().getInfo()
        if count == 0:
            raise ValueError(f"No AlphaEarth data found for year {year} at location ({lat}, {lon})")

        image = filtered.first()

        # Sample the embedding at the point location
        # Using a small buffer and reducing to get a single value per band
        sample = image.select(ALPHAEARTH_BAND_NAMES).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point.buffer(self.buffer_meters),
            scale=self.scale,
            maxPixels=10000,  # Increased from 1000 to handle 30m buffer at 10m scale
        )

        # Get the values as a dictionary
        values_dict = sample.getInfo()

        if values_dict is None:
            raise ValueError("No embedding values returned from Earth Engine")

        # Convert to numpy array in correct order
        features = np.array([
            values_dict.get(band, np.nan)
            for band in ALPHAEARTH_BAND_NAMES
        ], dtype=np.float32)

        # Replace None with NaN
        features = np.where(features == None, np.nan, features)

        return features

    def _empty_result(self, event_id: str) -> FeatureResult:
        """Create an empty result with NaN values for failed extractions."""
        return FeatureResult(
            event_id=event_id,
            features=np.full(self.feature_dim, np.nan),
            feature_names=self.feature_names,
            source=self.source_name,
            metadata={"error": "extraction_failed"},
        )

    def extract_batch(
        self,
        events: list[FireEvent],
        progress: bool = True,
        use_feature_collection: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Extract AlphaEarth features for multiple events.

        Args:
            events: List of FireEvent objects
            progress: Whether to log progress
            use_feature_collection: Use Earth Engine FeatureCollection for
                                    batch extraction (more efficient)
            **kwargs: Additional arguments

        Returns:
            DataFrame with event_id as index and embedding features as columns
        """
        if use_feature_collection and self._ensure_ee():
            try:
                return self._extract_batch_ee(events, progress)
            except Exception as e:
                logger.warning(f"Batch extraction failed: {e}, falling back to sequential")

        # Fall back to sequential extraction
        return super().extract_batch(events, progress=progress, **kwargs)

    def _extract_batch_ee(
        self,
        events: list[FireEvent],
        progress: bool = True,
    ) -> pd.DataFrame:
        """
        Batch extract using Earth Engine FeatureCollection.

        More efficient than individual queries for many events.
        """
        if not self._ensure_ee():
            return pd.DataFrame()

        # Check cache first, collect uncached events
        results: dict[str, np.ndarray] = {}
        uncached_events = []

        for event in events:
            cached = self._load_from_cache(event.event_id)
            if cached is not None:
                results[event.event_id] = cached
            else:
                uncached_events.append(event)

        logger.info(
            f"[{self.source_name}] {len(results)} cached, "
            f"{len(uncached_events)} to fetch"
        )

        if uncached_events:
            # Group events by year
            events_by_year: dict[int, list[FireEvent]] = {}
            for event in uncached_events:
                year = event.date.year
                if year not in events_by_year:
                    events_by_year[year] = []
                events_by_year[year].append(event)

            # Process each year
            for year, year_events in events_by_year.items():
                if progress:
                    logger.info(f"[{self.source_name}] Processing year {year}: {len(year_events)} events")

                try:
                    year_results = self._batch_sample_year(year_events, year)
                    for event_id, features in year_results.items():
                        results[event_id] = features
                        self._save_to_cache(event_id, features)
                except Exception as e:
                    logger.warning(f"Year {year} batch failed: {e}")
                    # Fall back to individual extraction
                    for event in year_events:
                        result = self.extract(event, year=year)
                        results[event.event_id] = result.features

        # Build DataFrame
        rows = []
        for event in events:
            features = results.get(event.event_id, np.full(64, np.nan))
            row = {"event_id": event.event_id}
            for i, name in enumerate(self.feature_names):
                row[f"{self.source_name}_{name}"] = features[i]
            rows.append(row)

        df = pd.DataFrame(rows)
        df.set_index("event_id", inplace=True)

        logger.info(
            f"[{self.source_name}] Extracted {len(df)} events, "
            f"{len(df.columns)} features"
        )
        return df

    def _batch_sample_year(
        self,
        events: list[FireEvent],
        year: int,
    ) -> dict[str, np.ndarray]:
        """
        Sample embeddings for multiple events in a single year.

        Uses Earth Engine FeatureCollection for efficient batch sampling.
        """
        # Create feature collection of points
        features = []
        for event in events:
            point = ee.Geometry.Point([event.longitude, event.latitude])
            feature = ee.Feature(point, {"event_id": event.event_id})
            features.append(feature)

        fc = ee.FeatureCollection(features)

        # Get the annual embedding - use mosaic to combine all tiles covering the points
        collection = ee.ImageCollection(ALPHAEARTH_COLLECTION)
        # Filter by bounds of all points, then by date, then mosaic
        image = collection.filterBounds(fc).filterDate(f"{year}-01-01", f"{year}-12-31").mosaic()

        if image is None:
            raise ValueError(f"No AlphaEarth data for year {year}")

        # Sample all points
        sampled = image.select(ALPHAEARTH_BAND_NAMES).sampleRegions(
            collection=fc,
            scale=self.scale,
            geometries=False,
        )

        # Get results
        sampled_list = sampled.getInfo()

        results: dict[str, np.ndarray] = {}
        for item in sampled_list.get("features", []):
            props = item.get("properties", {})
            event_id = props.get("event_id")
            if event_id:
                features_arr = np.array([
                    props.get(band, np.nan)
                    for band in ALPHAEARTH_BAND_NAMES
                ], dtype=np.float32)
                results[event_id] = features_arr

        return results
