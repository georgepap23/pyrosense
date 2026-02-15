"""
PyroSense VLM Image Utilities
=============================

Convert HLS composites to RGB images for EarthDial VLM input.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from PIL import Image as PILImage


def hls_to_rgb(
    composite_path: str | Path,
    output_path: str | Path | None = None,
    scale_factor: float = 3000.0,
    crop_size: int = 224,
    center_lat: float | None = None,
    center_lon: float | None = None,
) -> Path:
    """
    Convert a 6-band HLS composite GeoTIFF to RGB image for EarthDial.

    Crops to a 224x224 area (6.7 km × 6.7 km at 30m resolution) to match
    the area used by Prithvi for fire probability prediction.

    HLS bands: 1=Blue, 2=Green, 3=Red, 4=NIR, 5=SWIR1, 6=SWIR2
    Output: RGB PNG with bands 3,2,1 (Red, Green, Blue)

    Args:
        composite_path: Path to HLS composite GeoTIFF
        output_path: Output path for RGB image (default: composite dir/rgb_preview.png)
        scale_factor: HLS reflectance scale factor (default: 3000)
        crop_size: Size of crop in pixels (default: 224, matching Prithvi)
        center_lat: Latitude to center crop on (default: center of image)
        center_lon: Longitude to center crop on (default: center of image)

    Returns:
        Path to the generated RGB image
    """
    import rasterio
    from rasterio.windows import Window
    from PIL import Image

    composite_path = Path(composite_path)

    if not composite_path.exists():
        raise FileNotFoundError(f"Composite not found: {composite_path}")

    with rasterio.open(composite_path) as src:
        # Determine crop window
        if center_lat is not None and center_lon is not None:
            # Transform lat/lon (WGS84) to image CRS (usually UTM)
            from pyproj import Transformer
            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            x, y = transformer.transform(center_lon, center_lat)
            # Convert projected coordinates to pixel coordinates
            row, col = src.index(x, y)
            logger.debug(f"Transformed ({center_lat}, {center_lon}) -> pixel ({row}, {col})")
        else:
            # Use center of image
            row, col = src.height // 2, src.width // 2

        # Calculate window bounds (centered on row, col)
        half_size = crop_size // 2
        row_start = max(0, row - half_size)
        col_start = max(0, col - half_size)

        # Ensure we don't exceed image bounds
        row_start = min(row_start, src.height - crop_size)
        col_start = min(col_start, src.width - crop_size)

        window = Window(col_start, row_start, crop_size, crop_size)

        # Read RGB bands with crop (Red=3, Green=2, Blue=1 in HLS)
        red = src.read(3, window=window).astype(np.float32)
        green = src.read(2, window=window).astype(np.float32)
        blue = src.read(1, window=window).astype(np.float32)

        logger.debug(
            f"Cropped {crop_size}x{crop_size} pixels "
            f"({crop_size * 30 / 1000:.1f} km × {crop_size * 30 / 1000:.1f} km) "
            f"from {src.width}x{src.height} image"
        )

    # Stack RGB
    rgb = np.stack([red, green, blue], axis=-1)

    # Handle nodata values (typically -9999 or very negative)
    valid_mask = (rgb > 0) & (rgb < scale_factor * 2)
    rgb = np.where(valid_mask, rgb, 0)

    # Normalize to 0-255 using reflectance scale
    rgb = np.clip(rgb / scale_factor * 255, 0, 255).astype(np.uint8)

    # Determine output path
    if output_path is None:
        output_path = composite_path.parent / "rgb_preview.png"
    else:
        output_path = Path(output_path)

    # Save as PNG
    Image.fromarray(rgb).save(output_path)
    logger.debug(f"RGB image saved: {output_path}")

    return output_path


def ensure_rgb_image(
    image_path: str | Path,
    center_lat: float | None = None,
    center_lon: float | None = None,
    crop_size: int = 224,
) -> Path:
    """
    Ensure an image is in RGB format for EarthDial.

    If the input is a GeoTIFF with multiple bands, convert to RGB and crop
    to match the area used by Prithvi (224×224 pixels = 6.7 km × 6.7 km).

    Args:
        image_path: Path to input image
        center_lat: Latitude to center crop on (default: center of image)
        center_lon: Longitude to center crop on (default: center of image)
        crop_size: Size of crop in pixels (default: 224)

    Returns:
        Path to RGB image ready for EarthDial
    """
    image_path = Path(image_path)

    # Check file extension
    suffix = image_path.suffix.lower()

    if suffix in (".png", ".jpg", ".jpeg"):
        # Already in standard image format
        return image_path

    if suffix in (".tif", ".tiff"):
        # Likely a GeoTIFF - check band count
        import rasterio

        with rasterio.open(image_path) as src:
            if src.count >= 3:
                # Multi-band - convert to RGB with crop
                return hls_to_rgb(
                    image_path,
                    center_lat=center_lat,
                    center_lon=center_lon,
                    crop_size=crop_size,
                )
            else:
                raise ValueError(f"GeoTIFF has {src.count} bands, need at least 3")

    raise ValueError(f"Unsupported image format: {suffix}")


def create_thumbnail(
    image_path: str | Path,
    max_size: tuple[int, int] = (512, 512),
    output_path: str | Path | None = None,
) -> Path:
    """
    Create a thumbnail of an image for faster VLM processing.

    Args:
        image_path: Path to input image
        max_size: Maximum thumbnail dimensions
        output_path: Output path (default: same dir with _thumb suffix)

    Returns:
        Path to thumbnail image
    """
    from PIL import Image

    image_path = Path(image_path)

    # Ensure RGB first
    rgb_path = ensure_rgb_image(image_path)

    # Open and resize
    img = Image.open(rgb_path)
    img.thumbnail(max_size, Image.Resampling.LANCZOS)

    # Determine output path
    if output_path is None:
        output_path = image_path.parent / f"{image_path.stem}_thumb.png"
    else:
        output_path = Path(output_path)

    img.save(output_path)
    logger.debug(f"Thumbnail saved: {output_path}")

    return output_path
