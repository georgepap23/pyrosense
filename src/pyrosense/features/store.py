"""
Feature Store
=============

Unified storage and caching for extracted features.

The FeatureStore manages persistence of features from multiple sources
(Prithvi, Weather, AlphaEarth) and provides methods for combining them
into a single feature matrix for model training.

Storage Format:
- Parquet files for efficient columnar storage
- Separate files per source and version
- Metadata tracking for reproducibility
"""

from __future__ import annotations

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from pyrosense.features.base import BaseFeatureExtractor


class FeatureStore:
    """
    Unified feature storage with versioning and caching.

    Features are stored as Parquet files organized by source and version.
    The store handles merging features from multiple sources for model training.

    Directory Structure:
        store_dir/
        ├── prithvi/
        │   ├── v1.parquet
        │   ├── v2.parquet
        │   └── metadata.json
        ├── weather/
        │   ├── v1.parquet
        │   └── metadata.json
        └── alphaearth/
            ├── v1.parquet
            └── metadata.json

    Usage:
        store = FeatureStore("data/features/")

        # Save extracted features
        store.save("prithvi", prithvi_df, version="v1")
        store.save("weather", weather_df, version="v1")

        # Load features
        prithvi = store.load("prithvi", version="v1")

        # Combine multiple sources
        combined = store.get_combined(["prithvi", "weather"])
    """

    def __init__(self, store_dir: str | Path) -> None:
        """
        Args:
            store_dir: Directory for feature storage
        """
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def _get_source_dir(self, source: str) -> Path:
        """Get directory for a specific source."""
        source_dir = self.store_dir / source
        source_dir.mkdir(parents=True, exist_ok=True)
        return source_dir

    def _get_feature_path(self, source: str, version: str) -> Path:
        """Get path for a feature file."""
        return self._get_source_dir(source) / f"{version}.parquet"

    def _get_metadata_path(self, source: str) -> Path:
        """Get path for source metadata."""
        return self._get_source_dir(source) / "metadata.json"

    def save(
        self,
        source: str,
        features: pd.DataFrame,
        version: str = "v1",
        metadata: dict | None = None,
    ) -> Path:
        """
        Save features to store.

        Args:
            source: Feature source name (e.g., "prithvi", "weather")
            features: DataFrame with event_id as index
            version: Version identifier
            metadata: Optional metadata dict

        Returns:
            Path to saved file
        """
        feature_path = self._get_feature_path(source, version)

        # Ensure event_id is in index
        if "event_id" in features.columns:
            features = features.set_index("event_id")

        # Save as parquet
        features.to_parquet(feature_path, index=True)
        logger.info(
            f"Saved {len(features)} events x {len(features.columns)} features "
            f"to {feature_path}"
        )

        # Update metadata
        self._update_metadata(source, version, features, metadata)

        return feature_path

    def _update_metadata(
        self,
        source: str,
        version: str,
        features: pd.DataFrame,
        extra_metadata: dict | None = None,
    ) -> None:
        """Update source metadata file."""
        metadata_path = self._get_metadata_path(source)

        # Load existing metadata
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
        else:
            metadata = {"source": source, "versions": {}}

        # Update version info
        metadata["versions"][version] = {
            "n_events": len(features),
            "n_features": len(features.columns),
            "feature_names": features.columns.tolist(),
            "created_at": datetime.now().isoformat(),
            **(extra_metadata or {}),
        }

        # Save metadata
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def load(
        self,
        source: str,
        version: str = "v1",
    ) -> pd.DataFrame:
        """
        Load features from store.

        Args:
            source: Feature source name
            version: Version to load

        Returns:
            DataFrame with event_id as index
        """
        feature_path = self._get_feature_path(source, version)

        if not feature_path.exists():
            raise FileNotFoundError(f"Features not found: {feature_path}")

        features = pd.read_parquet(feature_path)
        logger.info(
            f"Loaded {len(features)} events x {len(features.columns)} features "
            f"from {feature_path}"
        )
        return features

    def load_if_exists(
        self,
        source: str,
        version: str = "v1",
    ) -> pd.DataFrame | None:
        """
        Load features if they exist, otherwise return None.

        Args:
            source: Feature source name
            version: Version to load

        Returns:
            DataFrame or None if not found
        """
        try:
            return self.load(source, version)
        except FileNotFoundError:
            return None

    def exists(self, source: str, version: str = "v1") -> bool:
        """Check if features exist in store."""
        return self._get_feature_path(source, version).exists()

    def list_sources(self) -> list[str]:
        """List all available sources."""
        sources = []
        for path in self.store_dir.iterdir():
            if path.is_dir() and (path / "metadata.json").exists():
                sources.append(path.name)
        return sources

    def list_versions(self, source: str) -> list[str]:
        """List all versions for a source."""
        source_dir = self._get_source_dir(source)
        versions = []
        for path in source_dir.glob("*.parquet"):
            versions.append(path.stem)
        return sorted(versions)

    def get_metadata(self, source: str) -> dict | None:
        """Get metadata for a source."""
        metadata_path = self._get_metadata_path(source)
        if metadata_path.exists():
            with open(metadata_path) as f:
                return json.load(f)
        return None

    def get_combined(
        self,
        sources: list[str],
        version: str = "v1",
        event_ids: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Combine features from multiple sources.

        Args:
            sources: List of source names to combine
            version: Version to load for each source
            event_ids: Optional filter to specific events

        Returns:
            Combined DataFrame with event_id as index
        """
        dfs = []

        for source in sources:
            try:
                df = self.load(source, version)
                dfs.append(df)
            except FileNotFoundError:
                logger.warning(f"Source '{source}' version '{version}' not found, skipping")

        if not dfs:
            raise ValueError(f"No features found for sources: {sources}")

        # Combine on index (event_id)
        combined = dfs[0]
        for df in dfs[1:]:
            combined = combined.join(df, how="inner")

        # Filter to specific events if requested
        if event_ids is not None:
            combined = combined.loc[combined.index.isin(event_ids)]

        logger.info(
            f"Combined {len(sources)} sources: "
            f"{len(combined)} events x {len(combined.columns)} features"
        )
        return combined

    def extract_and_save(
        self,
        extractor: BaseFeatureExtractor,
        events: list,
        version: str = "v1",
        force: bool = False,
        **extract_kwargs,
    ) -> pd.DataFrame:
        """
        Extract features using an extractor and save to store.

        Args:
            extractor: Feature extractor instance
            events: List of FireEvent objects
            version: Version identifier
            force: Overwrite existing features
            **extract_kwargs: Arguments passed to extractor.extract_batch()

        Returns:
            Extracted feature DataFrame
        """
        source = extractor.source_name

        # Check if already exists
        if not force and self.exists(source, version):
            logger.info(f"Features exist for {source}/{version}, loading from store")
            return self.load(source, version)

        # Extract features
        logger.info(f"Extracting {source} features for {len(events)} events")
        features = extractor.extract_batch(events, **extract_kwargs)

        # Save to store
        self.save(source, features, version)

        return features

    def delete(self, source: str, version: str) -> bool:
        """
        Delete a specific version.

        Args:
            source: Feature source name
            version: Version to delete

        Returns:
            True if deleted, False if not found
        """
        feature_path = self._get_feature_path(source, version)

        if feature_path.exists():
            feature_path.unlink()
            logger.info(f"Deleted {feature_path}")

            # Update metadata
            metadata = self.get_metadata(source)
            if metadata and version in metadata.get("versions", {}):
                del metadata["versions"][version]
                with open(self._get_metadata_path(source), "w") as f:
                    json.dump(metadata, f, indent=2)

            return True
        return False

    def to_numpy(
        self,
        sources: list[str],
        version: str = "v1",
    ) -> tuple[np.ndarray, list[str], list[str]]:
        """
        Get combined features as numpy array.

        Args:
            sources: List of source names
            version: Version to load

        Returns:
            Tuple of (feature_array, event_ids, feature_names)
        """
        combined = self.get_combined(sources, version)
        return (
            combined.values,
            combined.index.tolist(),
            combined.columns.tolist(),
        )
