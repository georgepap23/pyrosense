"""
Base Feature Extractor
======================

Abstract base class for all feature extractors in PyroSense.
Provides a consistent interface for extracting features from different sources.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
import numpy as np
import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from pyrosense.data.mesogeos_loader import FireEvent


@dataclass
class FeatureResult:
    """
    Container for extracted features from a single event.

    Attributes:
        event_id: Unique identifier linking back to the fire event
        features: Feature values as numpy array
        feature_names: Names of each feature dimension
        source: Name of the feature source (e.g., "prithvi", "weather")
        metadata: Optional additional information about extraction
    """
    event_id: str
    features: np.ndarray
    feature_names: list[str]
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float]:
        """Convert features to a dictionary with prefixed names."""
        return {
            f"{self.source}_{name}": float(val)
            for name, val in zip(self.feature_names, self.features)
        }

    def to_series(self) -> pd.Series:
        """Convert to pandas Series with prefixed column names."""
        return pd.Series(
            self.features,
            index=[f"{self.source}_{name}" for name in self.feature_names],
            name=self.event_id,
        )


class BaseFeatureExtractor(ABC):
    """
    Abstract base class for feature extractors.

    All extractors must implement:
    - feature_dim: Number of features extracted
    - feature_names: Names for each feature dimension
    - source_name: Identifier for this feature source
    - extract: Extract features from a single event
    - extract_batch: Extract features from multiple events

    Example:
        class MyExtractor(BaseFeatureExtractor):
            @property
            def feature_dim(self) -> int:
                return 10

            @property
            def feature_names(self) -> list[str]:
                return [f"feat_{i}" for i in range(10)]

            @property
            def source_name(self) -> str:
                return "my_source"

            def extract(self, event: FireEvent, **kwargs) -> FeatureResult:
                features = np.random.randn(10)
                return FeatureResult(
                    event_id=event.event_id,
                    features=features,
                    feature_names=self.feature_names,
                    source=self.source_name,
                )
    """

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Number of features extracted per event."""
        ...

    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        """Names for each feature dimension (without source prefix)."""
        ...

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Identifier for this feature source (e.g., 'prithvi', 'weather')."""
        ...

    @abstractmethod
    def extract(self, event: FireEvent, **kwargs) -> FeatureResult:
        """
        Extract features from a single event.

        Args:
            event: FireEvent with location and date information
            **kwargs: Extractor-specific arguments

        Returns:
            FeatureResult containing the extracted features
        """
        ...

    def extract_batch(
        self,
        events: list[FireEvent],
        progress: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Extract features from multiple events.

        Args:
            events: List of FireEvent objects
            progress: Whether to log progress
            **kwargs: Passed to extract()

        Returns:
            DataFrame with event_id as index, features as columns
            Column names are prefixed with source_name
        """
        results: list[FeatureResult] = []
        n_events = len(events)

        for i, event in enumerate(events):
            if progress and (i + 1) % 10 == 0:
                logger.info(f"[{self.source_name}] Extracting: {i + 1}/{n_events}")

            try:
                result = self.extract(event, **kwargs)
                results.append(result)
            except Exception as e:
                logger.warning(f"[{self.source_name}] Failed for {event.event_id}: {e}")

        if not results:
            return pd.DataFrame()

        # Build DataFrame from results
        rows = []
        for result in results:
            row = {"event_id": result.event_id}
            row.update(result.to_dict())
            rows.append(row)

        df = pd.DataFrame(rows)
        df.set_index("event_id", inplace=True)

        logger.info(
            f"[{self.source_name}] Extracted {len(df)} events, "
            f"{len(df.columns)} features"
        )
        return df

    def get_column_prefix(self) -> str:
        """Return the prefix used for column names."""
        return f"{self.source_name}_"


# Global registry of feature extractors
FEATURE_REGISTRY: dict[str, type[BaseFeatureExtractor]] = {}


def register_extractor(name: str):
    """Decorator to register a feature extractor class."""
    def decorator(cls: type[BaseFeatureExtractor]):
        FEATURE_REGISTRY[name] = cls
        return cls
    return decorator


def get_extractor(name: str, **kwargs) -> BaseFeatureExtractor:
    """
    Get a feature extractor by name.

    Args:
        name: Name of the registered extractor
        **kwargs: Arguments passed to extractor constructor

    Returns:
        Instantiated feature extractor
    """
    if name not in FEATURE_REGISTRY:
        available = ", ".join(FEATURE_REGISTRY.keys())
        raise ValueError(f"Unknown extractor '{name}'. Available: {available}")
    return FEATURE_REGISTRY[name](**kwargs)
