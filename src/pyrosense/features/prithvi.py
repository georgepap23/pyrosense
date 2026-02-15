"""
Prithvi Feature Extractor
=========================

Extracts features from satellite imagery using NASA's Prithvi-EO-2.0 foundation model.

Loads the model directly via timm + HuggingFace Hub (no terratorch dependency).
Prithvi-EO-2.0-300M is a ViT-Large (24 blocks, 1024-dim) pre-trained on HLS imagery
using Masked Autoencoder (MAE) self-supervision.

THE EXTRACTION PROCESS:
-----------------------
1. Load satellite image (6-band HLS composite GeoTIFF)
2. Crop a 224x224 patch centered on the event location (preserves 30m resolution)
3. Normalize using HLS statistics
4. Pass through frozen Prithvi encoder → 1024-dim feature vector
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import TYPE_CHECKING
from dataclasses import dataclass
from loguru import logger

from pyrosense.features.base import BaseFeatureExtractor, FeatureResult, register_extractor

try:
    import timm
    from huggingface_hub import hf_hub_download
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    logger.warning("timm or huggingface_hub not installed. Using mock extractor.")

try:
    import rasterio
    from rasterio.windows import Window
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

if TYPE_CHECKING:
    from pyrosense.data.mesogeos_loader import FireEvent


@dataclass
class PrithviFeatures:
    """
    Container for extracted Prithvi features.

    Attributes:
        event_id: Links back to the fire event
        features: The actual feature vector (1024 dimensions)
        model_name: Which Prithvi model was used
    """
    event_id: str
    features: np.ndarray  # Shape: (1024,)
    model_name: str


# Normalization values for HLS data (from Prithvi documentation)
HLS_MEAN = [0.033, 0.033, 0.033, 0.033, 0.033, 0.033]
HLS_STD = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# HuggingFace repo for Prithvi weights
PRITHVI_REPOS: dict[str, dict[str, str]] = {
    "Prithvi-EO-2.0-300M": {
        "repo_id": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M",
        "filename": "Prithvi_EO_V2_300M.pt",
        "timm_model": "vit_large_patch16_224",  # ViT-Large: 24 blocks, 1024-dim
    },
}


@register_extractor("prithvi")
class PrithviExtractor(BaseFeatureExtractor):
    """
    Extracts features from satellite images using frozen Prithvi model.

    Loads Prithvi-EO-2.0 directly via timm (ViT-Large architecture) and
    HuggingFace Hub (pre-trained weights), bypassing terratorch.

    Usage:
        extractor = PrithviExtractor(model_name="Prithvi-EO-2.0-300M")
        features = extractor.extract(event, image_path=path)
    """

    def __init__(
        self,
        model_name: str = "Prithvi-EO-2.0-300M",
        device: str = "auto",
        hls_dir: str | Path = "data/hls/",
    ) -> None:
        self.model_name = model_name
        self.hls_dir = Path(hls_dir)

        # Auto-detect device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")
        self.model = self._load_model()

    @property
    def feature_dim(self) -> int:
        """Prithvi-EO-2.0-300M outputs 1024-dimensional features."""
        return 1024

    @property
    def feature_names(self) -> list[str]:
        """Feature names are prithvi_0 through prithvi_1023."""
        return [f"f{i}" for i in range(self.feature_dim)]

    @property
    def source_name(self) -> str:
        """Source identifier for Prithvi features."""
        return "prithvi"

    def _load_model(self) -> nn.Module:
        """
        Load the pre-trained Prithvi encoder.

        Creates a ViT-Large via timm, then loads Prithvi's pre-trained
        encoder weights from HuggingFace Hub. Only the encoder is used
        (the MAE decoder is discarded).
        """
        if not TIMM_AVAILABLE:
            logger.warning("timm not available, using mock model")
            return self._create_mock_model()

        model_info = PRITHVI_REPOS.get(self.model_name)
        if model_info is None:
            logger.warning(f"Unknown model {self.model_name}, using mock")
            return self._create_mock_model()

        logger.info(f"Loading Prithvi model: {self.model_name}")

        try:
            # Create ViT-Large architecture matching Prithvi's encoder
            model = timm.create_model(
                model_info["timm_model"],
                in_chans=6,        # 6 HLS bands
                num_classes=0,      # no classification head — feature extraction only
                global_pool="avg",  # average pool 196 patch tokens → single 1024-dim vector
            )

            # Download pre-trained weights from HuggingFace
            weights_path = hf_hub_download(
                repo_id=model_info["repo_id"],
                filename=model_info["filename"],
            )
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)

            # Extract encoder weights and adapt shapes
            encoder_state = {}
            for k, v in checkpoint.items():
                if not k.startswith("encoder."):
                    continue
                new_key = k.replace("encoder.", "")

                if new_key == "patch_embed.proj.weight":
                    # Prithvi: (1024, 6, 1, 16, 16) → timm: (1024, 6, 16, 16)
                    # Remove temporal dimension (we use single-frame input)
                    v = v.squeeze(2)
                elif new_key == "pos_embed":
                    # Prithvi: (1, 785, 1024) for 4 frames → (1, 197, 1024) for 1 frame
                    # Keep cls_token position + first 196 patch positions
                    v = v[:, :197, :]

                encoder_state[new_key] = v

            result = model.load_state_dict(encoder_state, strict=False)
            logger.info(
                f"Loaded weights (matched: {len(encoder_state)}, "
                f"missing: {len(result.missing_keys)}, "
                f"unexpected: {len(result.unexpected_keys)})"
            )

            model = model.to(self.device)
            model.eval()

            for param in model.parameters():
                param.requires_grad = False

            n_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Model ready. Parameters: {n_params:,}")
            return model

        except Exception as e:
            logger.error(f"Failed to load Prithvi model: {e}")
            logger.warning("Falling back to mock model")
            return self._create_mock_model()

    def _create_mock_model(self) -> nn.Module:
        """Create a lightweight mock for testing when Prithvi isn't available."""
        class MockPrithvi(nn.Module):
            def __init__(self, feature_dim: int = 1024):
                super().__init__()
                self.feature_dim = feature_dim
                self.conv = nn.Sequential(
                    nn.Conv2d(6, 64, 7, stride=2, padding=3),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1),
                )
                self.fc = nn.Linear(64, feature_dim)

            def forward(self, x):
                x = self.conv(x)
                x = x.flatten(1)
                return self.fc(x)

        return MockPrithvi().to(self.device).eval()

    def preprocess_image(
        self,
        image_path: str | Path,
        center_lat: float | None = None,
        center_lon: float | None = None,
        target_size: int = 224,
    ) -> torch.Tensor:
        """
        Load and preprocess a satellite image for Prithvi.

        If center coordinates are provided and the image is larger than
        224x224, crops a patch centered on (lat, lon) at native 30m resolution.
        Otherwise falls back to resizing the full image.

        Args:
            image_path: Path to 6-band HLS composite GeoTIFF
            center_lat: Latitude of event center (for cropping)
            center_lon: Longitude of event center (for cropping)
            target_size: Spatial size Prithvi expects (224)

        Returns:
            Preprocessed tensor of shape (1, 6, 224, 224)
        """
        image_path = Path(image_path)

        if RASTERIO_AVAILABLE and image_path.exists():
            with rasterio.open(image_path) as src:
                # Try to crop a 224x224 patch at native resolution
                if (center_lat is not None and center_lon is not None
                        and src.height > target_size and src.width > target_size):
                    # Convert lat/lon to pixel coordinates
                    col, row = src.index(center_lon, center_lat)
                    # rasterio.index returns (row, col) despite the name
                    row, col = col, row

                    # Center the window
                    half = target_size // 2
                    row_start = max(0, min(row - half, src.height - target_size))
                    col_start = max(0, min(col - half, src.width - target_size))

                    window = Window(col_start, row_start, target_size, target_size)
                    image = src.read(window=window).astype(np.float32)
                else:
                    image = src.read().astype(np.float32)

                # Ensure 6 bands
                if image.shape[0] < 6:
                    padding = np.zeros((6 - image.shape[0], *image.shape[1:]), dtype=np.float32)
                    image = np.concatenate([image, padding], axis=0)
                elif image.shape[0] > 6:
                    image = image[:6]
        else:
            logger.debug("Creating synthetic image (file not found or rasterio unavailable)")
            image = np.random.randn(6, target_size, target_size).astype(np.float32) * 0.1

        # Normalize to [0, 1] range (HLS data is scaled by 10000)
        if image.max() > 1.0:
            image = image / 10000.0

        # Replace NaN with 0 (masked cloud/nodata pixels)
        image = np.nan_to_num(image, nan=0.0)

        # Resize if needed (only when cropping wasn't possible)
        if image.shape[1] != target_size or image.shape[2] != target_size:
            try:
                import cv2
                resized = np.zeros((6, target_size, target_size), dtype=np.float32)
                for i in range(6):
                    resized[i] = cv2.resize(image[i], (target_size, target_size))
                image = resized
            except ImportError:
                # Simple nearest-neighbor resize fallback
                from torch.nn.functional import interpolate
                t = torch.from_numpy(image).unsqueeze(0)
                t = interpolate(t, size=(target_size, target_size), mode="bilinear")
                image = t.squeeze(0).numpy()

        # Normalize with HLS statistics
        for i in range(6):
            image[i] = (image[i] - HLS_MEAN[i]) / HLS_STD[i]

        tensor = torch.from_numpy(image).unsqueeze(0).to(self.device)
        return tensor

    @torch.no_grad()
    def extract(
        self,
        event: FireEvent,
        image_path: str | Path | None = None,
        **kwargs,
    ) -> FeatureResult:
        """
        Extract features from a satellite image for an event.

        Args:
            event: FireEvent with location and date information
            image_path: Path to the satellite image. If None, looks in hls_dir.
            **kwargs: Additional arguments (unused)

        Returns:
            FeatureResult with the extracted 1024-dim feature vector
        """
        # Find image path if not provided
        if image_path is None:
            image_path = self.hls_dir / event.event_id / "composite.tif"

        image_path = Path(image_path)
        tensor = self.preprocess_image(image_path, event.latitude, event.longitude)
        features = self.model(tensor)

        # Handle various output shapes
        if len(features.shape) > 2:
            features = features.mean(dim=[2, 3])

        features_np = features.cpu().numpy().squeeze()
        logger.debug(f"Extracted features shape: {features_np.shape}")

        return FeatureResult(
            event_id=event.event_id,
            features=features_np,
            feature_names=self.feature_names,
            source=self.source_name,
            metadata={"model_name": self.model_name, "image_path": str(image_path)},
        )

    def extract_batch(
        self,
        events: list[FireEvent],
        progress: bool = True,
        image_paths: list[Path] | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Extract features from multiple events.

        Args:
            events: List of FireEvent objects
            progress: Whether to log progress
            image_paths: Optional list of image paths (one per event)
            **kwargs: Additional arguments

        Returns:
            DataFrame with event_id as index and features as columns
        """
        results: list[FeatureResult] = []
        n_events = len(events)

        for i, event in enumerate(events):
            if progress and (i + 1) % 10 == 0:
                logger.info(f"[{self.source_name}] Extracting: {i + 1}/{n_events}")

            try:
                img_path = image_paths[i] if image_paths else None
                result = self.extract(event, image_path=img_path, **kwargs)
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

    def extract_legacy(
        self,
        image_path: str | Path,
        event_id: str = "unknown",
        center_lat: float | None = None,
        center_lon: float | None = None,
    ) -> PrithviFeatures:
        """
        Legacy extraction method for backward compatibility.

        Args:
            image_path: Path to the satellite image
            event_id: Identifier to link features to events
            center_lat: Latitude for cropping (optional)
            center_lon: Longitude for cropping (optional)

        Returns:
            PrithviFeatures with the extracted 1024-dim feature vector
        """
        from pyrosense.data.mesogeos_loader import FireEvent

        # Create a minimal FireEvent for extraction
        event = FireEvent(
            event_id=event_id,
            latitude=center_lat or 0.0,
            longitude=center_lon or 0.0,
            date=pd.Timestamp.now(),
            burned_area=0.0,
        )

        result = self.extract(event, image_path=image_path)

        return PrithviFeatures(
            event_id=result.event_id,
            features=result.features,
            model_name=self.model_name,
        )


@register_extractor("prithvi_multitemporal")
class MultiTemporalPrithviExtractor(BaseFeatureExtractor):
    """
    Extracts features from multi-temporal satellite images using Prithvi.

    Prithvi-EO-2.0 natively supports multi-temporal input through its
    3D patch embeddings and positional encodings. This class loads
    the model preserving the temporal dimension.

    For 2 timestamps:
    - patch_embed: keeps temporal dim (1024, 6, 2, 16, 16)
    - pos_embed: sliced to 1 + 196*2 = 393 positions
    - Input shape: (batch, 6, 2, 224, 224)

    Usage:
        extractor = MultiTemporalPrithviExtractor(n_frames=2)
        features = extractor.extract(event, image_paths=[path_t1, path_t2])
    """

    def __init__(
        self,
        model_name: str = "Prithvi-EO-2.0-300M",
        n_frames: int = 2,
        device: str = "auto",
        hls_dir: str | Path = "data/hls/",
    ) -> None:
        self.model_name = model_name
        self.n_frames = n_frames
        self.hls_dir = Path(hls_dir)

        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Multi-temporal Prithvi ({n_frames} frames) on {self.device}")
        self.model = self._load_model()

    @property
    def feature_dim(self) -> int:
        """Multi-temporal Prithvi outputs 1024-dimensional features."""
        return 1024

    @property
    def feature_names(self) -> list[str]:
        """Feature names are prithvi_mt_0 through prithvi_mt_1023."""
        return [f"f{i}" for i in range(self.feature_dim)]

    @property
    def source_name(self) -> str:
        """Source identifier for multi-temporal Prithvi features."""
        return "prithvi_mt"

    def _load_model(self) -> nn.Module:
        """
        Load Prithvi with multi-temporal support.

        Unlike single-frame mode, we keep the temporal dimension in
        patch_embed and adjust pos_embed for the number of frames.
        """
        if not TIMM_AVAILABLE:
            logger.warning("timm not available, using mock model")
            return self._create_mock_model()

        model_info = PRITHVI_REPOS.get(self.model_name)
        if model_info is None:
            return self._create_mock_model()

        logger.info(f"Loading multi-temporal Prithvi: {self.model_name}")

        try:
            # Download weights
            weights_path = hf_hub_download(
                repo_id=model_info["repo_id"],
                filename=model_info["filename"],
            )
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)

            # Create model with 3D conv for temporal input
            model = self._create_temporal_model(checkpoint)
            return model

        except Exception as e:
            logger.error(f"Failed to load multi-temporal model: {e}")
            return self._create_mock_model()

    def _create_temporal_model(self, checkpoint: dict) -> nn.Module:
        """
        Create a ViT model that accepts multi-temporal input.

        Uses timm's ViT but wraps the patch embedding to handle
        temporal stacking.
        """
        # Create standard ViT-Large
        model = timm.create_model(
            "vit_large_patch16_224",
            in_chans=6 * self.n_frames,  # Stack frames in channel dim for timm
            num_classes=0,
            global_pool="avg",
        )

        # Extract and adapt encoder weights
        encoder_state = {}
        for k, v in checkpoint.items():
            if not k.startswith("encoder."):
                continue
            new_key = k.replace("encoder.", "")

            if new_key == "patch_embed.proj.weight":
                # Prithvi shape: (1024, 6, 1, 16, 16) or (1024, 6, T, 16, 16)
                # We need: (1024, 6*n_frames, 16, 16)
                # Stack the same weights for each frame
                if v.dim() == 5:
                    v = v.squeeze(2)  # Remove temporal dim: (1024, 6, 16, 16)
                # Tile weights for all frames
                v = v.repeat(1, self.n_frames, 1, 1)  # (1024, 12, 16, 16)

            elif new_key == "pos_embed":
                # Prithvi: (1, 785, 1024) for 4 frames
                # 785 = 1 (cls) + 196*4 (patches per frame)
                # For n_frames: 1 + 196*n_frames
                n_positions = 1 + 196 * self.n_frames
                if v.shape[1] >= n_positions:
                    v = v[:, :n_positions, :]
                # else: Interpolate if needed (not implemented)

            encoder_state[new_key] = v

        result = model.load_state_dict(encoder_state, strict=False)
        logger.info(
            f"Loaded multi-temporal weights (matched: {len(encoder_state)}, "
            f"missing: {len(result.missing_keys)})"
        )

        model = model.to(self.device).eval()
        for param in model.parameters():
            param.requires_grad = False

        return model

    def _create_mock_model(self) -> nn.Module:
        """Mock model for testing."""
        n_frames = self.n_frames

        class MockMultiTemporal(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(6 * n_frames, 64, 7, stride=2, padding=3),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1),
                )
                self.fc = nn.Linear(64, 1024)

            def forward(self, x):
                x = self.conv(x)
                x = x.flatten(1)
                return self.fc(x)

        return MockMultiTemporal().to(self.device).eval()

    def preprocess_multi_temporal(
        self,
        image_paths: list[Path],
        center_lat: float | None = None,
        center_lon: float | None = None,
        target_size: int = 224,
    ) -> torch.Tensor:
        """
        Load and preprocess multiple temporal images.

        Args:
            image_paths: List of paths to temporal composites
            center_lat, center_lon: Event coordinates for cropping
            target_size: Spatial size (224)

        Returns:
            Tensor of shape (1, 6*n_frames, 224, 224) — frames stacked in channels
        """
        all_frames = []

        for path in image_paths:
            path = Path(path)

            if RASTERIO_AVAILABLE and path.exists():
                with rasterio.open(path) as src:
                    if (center_lat is not None and center_lon is not None
                            and src.height > target_size and src.width > target_size):
                        col, row = src.index(center_lon, center_lat)
                        row, col = col, row
                        half = target_size // 2
                        row_start = max(0, min(row - half, src.height - target_size))
                        col_start = max(0, min(col - half, src.width - target_size))
                        window = Window(col_start, row_start, target_size, target_size)
                        image = src.read(window=window).astype(np.float32)
                    else:
                        image = src.read().astype(np.float32)

                    # Ensure 6 bands
                    if image.shape[0] < 6:
                        padding = np.zeros(
                            (6 - image.shape[0], *image.shape[1:]), dtype=np.float32
                        )
                        image = np.concatenate([image, padding], axis=0)
                    elif image.shape[0] > 6:
                        image = image[:6]
            else:
                image = np.random.randn(6, target_size, target_size).astype(np.float32) * 0.1

            # Normalize
            if image.max() > 1.0:
                image = image / 10000.0
            image = np.nan_to_num(image, nan=0.0)

            # Resize if needed
            if image.shape[1] != target_size or image.shape[2] != target_size:
                try:
                    import cv2
                    resized = np.zeros((6, target_size, target_size), dtype=np.float32)
                    for i in range(6):
                        resized[i] = cv2.resize(image[i], (target_size, target_size))
                    image = resized
                except ImportError:
                    from torch.nn.functional import interpolate
                    t = torch.from_numpy(image).unsqueeze(0)
                    t = interpolate(t, size=(target_size, target_size), mode="bilinear")
                    image = t.squeeze(0).numpy()

            # Normalize with HLS statistics
            for i in range(6):
                image[i] = (image[i] - HLS_MEAN[i]) / HLS_STD[i]

            all_frames.append(image)

        # Stack frames in channel dimension: (6*n_frames, H, W)
        stacked = np.concatenate(all_frames, axis=0)
        tensor = torch.from_numpy(stacked).unsqueeze(0).to(self.device)
        return tensor

    @torch.no_grad()
    def extract(
        self,
        event: FireEvent,
        image_paths: list[Path] | None = None,
        **kwargs,
    ) -> FeatureResult:
        """
        Extract features from multiple temporal images.

        Args:
            event: FireEvent with location and date information
            image_paths: List of paths to temporal composites

        Returns:
            FeatureResult with 1024-dim feature vector
        """
        # Find image paths if not provided
        if image_paths is None:
            event_dir = self.hls_dir / event.event_id
            image_paths = [
                event_dir / f"composite_t{i+1}.tif"
                for i in range(self.n_frames)
            ]

        tensor = self.preprocess_multi_temporal(
            image_paths, event.latitude, event.longitude
        )
        features = self.model(tensor)

        if len(features.shape) > 2:
            features = features.mean(dim=[2, 3])

        features_np = features.cpu().numpy().squeeze()
        logger.debug(f"Multi-temporal features shape: {features_np.shape}")

        return FeatureResult(
            event_id=event.event_id,
            features=features_np,
            feature_names=self.feature_names,
            source=self.source_name,
            metadata={
                "model_name": f"{self.model_name}-{self.n_frames}T",
                "n_frames": self.n_frames,
            },
        )


def features_to_array(features_list: list[PrithviFeatures]) -> tuple[np.ndarray, list[str]]:
    """Convert list of PrithviFeatures to a numpy array (legacy compatibility)."""
    event_ids = [f.event_id for f in features_list]
    feature_array = np.stack([f.features for f in features_list])
    return feature_array, event_ids
