"""
Weather Feature Extractor
=========================

Extracts weather features from Open-Meteo Archive API.

For each fire event, retrieves daily weather data for the specified
number of days before the event and aggregates into features relevant
for wildfire prediction:
- Temperature (max, mean) - high temps increase fire risk
- Relative humidity (min, mean) - low humidity increases fire risk
- Wind speed (max, mean) - high winds spread fires
- Precipitation (sum, max) - recent rain reduces fire risk
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING
from loguru import logger

from pyrosense.features.base import BaseFeatureExtractor, FeatureResult, register_extractor

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests not installed. Weather fetching will not work.")

if TYPE_CHECKING:
    from pyrosense.data.mesogeos_loader import FireEvent


# Weather feature names in output order
WEATHER_FEATURE_NAMES = [
    "temp_max",
    "temp_min",
    "temp_mean",
    "humidity_max",
    "humidity_min",
    "humidity_mean",
    "wind_max",
    "wind_mean",
    "precip_sum",
    "precip_max",
]


@register_extractor("weather")
class WeatherExtractor(BaseFeatureExtractor):
    """
    Extracts weather features from Open-Meteo Archive API.

    For each fire event, retrieves daily weather data for the specified
    number of days before the event and aggregates into features.

    Usage:
        extractor = WeatherExtractor(days_before=7)
        features = extractor.extract(event)
    """

    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

    # Weather variables to fetch from API
    DAILY_VARS: list[str] = [
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "relative_humidity_2m_max",
        "relative_humidity_2m_min",
        "relative_humidity_2m_mean",
        "wind_speed_10m_max",
        "wind_speed_10m_mean",
        "precipitation_sum",
    ]

    def __init__(
        self,
        days_before: int = 7,
        cache_dir: str | Path | None = None,
    ) -> None:
        """
        Args:
            days_before: Number of days before event to fetch weather for
            cache_dir: Optional directory to cache weather responses
        """
        self.days_before = days_before
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def feature_dim(self) -> int:
        """Weather extractor produces 10 features."""
        return len(WEATHER_FEATURE_NAMES)

    @property
    def feature_names(self) -> list[str]:
        """Feature names without source prefix."""
        return WEATHER_FEATURE_NAMES.copy()

    @property
    def source_name(self) -> str:
        """Source identifier for weather features."""
        return "weather"

    def extract(
        self,
        event: FireEvent,
        **kwargs,
    ) -> FeatureResult:
        """
        Fetch and aggregate weather features for a single event.

        Args:
            event: FireEvent with lat, lon, date
            **kwargs: Additional arguments (unused)

        Returns:
            FeatureResult with aggregated weather statistics
        """
        if not REQUESTS_AVAILABLE:
            logger.error("requests library not available")
            return self._empty_result(event.event_id)

        # Calculate date range
        end_date = event.date - timedelta(days=1)  # Day before event
        start_date = end_date - timedelta(days=self.days_before - 1)

        params = {
            "latitude": event.latitude,
            "longitude": event.longitude,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "daily": ",".join(self.DAILY_VARS),
            "timezone": "UTC",
        }

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.warning(f"Weather fetch failed for {event.event_id}: {e}")
            return self._empty_result(event.event_id)

        if "daily" not in data:
            logger.warning(f"No daily data in response for {event.event_id}")
            return self._empty_result(event.event_id)

        # Aggregate daily values into features
        features = self._aggregate_weather(data["daily"])

        return FeatureResult(
            event_id=event.event_id,
            features=features,
            feature_names=self.feature_names,
            source=self.source_name,
            metadata={
                "days_before": self.days_before,
                "latitude": event.latitude,
                "longitude": event.longitude,
            },
        )

    def _aggregate_weather(self, daily_data: dict) -> np.ndarray:
        """
        Aggregate daily weather values into summary features.

        Returns features as numpy array in order of WEATHER_FEATURE_NAMES:
        - temp_max: Maximum temperature over the period
        - temp_min: Minimum temperature over the period
        - temp_mean: Mean temperature
        - humidity_max: Maximum humidity
        - humidity_min: Minimum humidity (dry = fire risk)
        - humidity_mean: Mean humidity
        - wind_max: Maximum wind speed
        - wind_mean: Mean wind speed
        - precip_sum: Total precipitation
        - precip_max: Maximum daily precipitation
        """
        features: dict[str, float] = {}

        # Temperature features
        if "temperature_2m_max" in daily_data:
            vals = [v for v in daily_data["temperature_2m_max"] if v is not None]
            features["temp_max"] = float(np.max(vals)) if vals else np.nan

        if "temperature_2m_min" in daily_data:
            vals = [v for v in daily_data["temperature_2m_min"] if v is not None]
            features["temp_min"] = float(np.min(vals)) if vals else np.nan

        if "temperature_2m_mean" in daily_data:
            vals = [v for v in daily_data["temperature_2m_mean"] if v is not None]
            features["temp_mean"] = float(np.mean(vals)) if vals else np.nan

        # Humidity features
        if "relative_humidity_2m_max" in daily_data:
            vals = [v for v in daily_data["relative_humidity_2m_max"] if v is not None]
            features["humidity_max"] = float(np.max(vals)) if vals else np.nan

        if "relative_humidity_2m_min" in daily_data:
            vals = [v for v in daily_data["relative_humidity_2m_min"] if v is not None]
            features["humidity_min"] = float(np.min(vals)) if vals else np.nan

        if "relative_humidity_2m_mean" in daily_data:
            vals = [v for v in daily_data["relative_humidity_2m_mean"] if v is not None]
            features["humidity_mean"] = float(np.mean(vals)) if vals else np.nan

        # Wind features
        if "wind_speed_10m_max" in daily_data:
            vals = [v for v in daily_data["wind_speed_10m_max"] if v is not None]
            features["wind_max"] = float(np.max(vals)) if vals else np.nan

        if "wind_speed_10m_mean" in daily_data:
            vals = [v for v in daily_data["wind_speed_10m_mean"] if v is not None]
            features["wind_mean"] = float(np.mean(vals)) if vals else np.nan

        # Precipitation features
        if "precipitation_sum" in daily_data:
            vals = [v for v in daily_data["precipitation_sum"] if v is not None]
            features["precip_sum"] = float(np.sum(vals)) if vals else np.nan
            features["precip_max"] = float(np.max(vals)) if vals else np.nan

        # Convert to array in correct order
        return np.array([
            features.get(name, np.nan)
            for name in WEATHER_FEATURE_NAMES
        ])

    def _empty_result(self, event_id: str) -> FeatureResult:
        """Create an empty result with NaN values for failed fetches."""
        return FeatureResult(
            event_id=event_id,
            features=np.full(self.feature_dim, np.nan),
            feature_names=self.feature_names,
            source=self.source_name,
            metadata={"error": "fetch_failed"},
        )

    def extract_batch(
        self,
        events: list[FireEvent],
        progress: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Extract weather features for multiple events.

        Args:
            events: List of FireEvent objects
            progress: Whether to log progress
            **kwargs: Additional arguments

        Returns:
            DataFrame with event_id as index and weather features as columns
        """
        results: list[FeatureResult] = []
        n_events = len(events)

        for i, event in enumerate(events):
            if progress and (i + 1) % 10 == 0:
                logger.info(f"[{self.source_name}] Fetching: {i + 1}/{n_events}")

            try:
                result = self.extract(event, **kwargs)
                results.append(result)
            except Exception as e:
                logger.warning(f"[{self.source_name}] Failed for {event.event_id}: {e}")
                results.append(self._empty_result(event.event_id))

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

        # Fill missing values with column medians
        for col in df.columns:
            if df[col].isna().any():
                median_val = df[col].median()
                if pd.notna(median_val):
                    df[col] = df[col].fillna(median_val)

        logger.info(
            f"[{self.source_name}] Extracted {len(df)} events, "
            f"{len(df.columns)} features"
        )
        return df
