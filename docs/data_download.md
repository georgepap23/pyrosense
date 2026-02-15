# Data Download Guide

This guide explains how to obtain the datasets required for PyroSense.

## Overview

| Source | Purpose | Access |
|--------|---------|--------|
| Mesogeos | Fire labels | Google Drive |
| HLS | Satellite imagery | NASA Earthdata |
| AlphaEarth | Land embeddings | Google Earth Engine |

## 1. Mesogeos Datacube

Mediterranean wildfire datacube (2006-2022) with fire labels.

### Download via rclone

```bash
# Install rclone
brew install rclone  # macOS
# or: curl https://rclone.org/install.sh | sudo bash  # Linux

# Configure for Google Drive
rclone config
# Follow prompts to set up 'gdrive' remote

# Download
rclone sync gdrive:mesogeos/mesogeos.zarr data/mesogeos/mesogeos.zarr
```

Alternative: Download from [Mesogeos GitHub](https://github.com/Orion-AI-Lab/mesogeos)

## 2. HLS Satellite Imagery

Harmonized Landsat-Sentinel (HLS) imagery from NASA Earthdata.

### Prerequisites

1. Create a [NASA Earthdata account](https://urs.earthdata.nasa.gov/)

2. Configure credentials:
   ```bash
   cat > ~/.netrc << EOF
   machine urs.earthdata.nasa.gov
       login YOUR_USERNAME
       password YOUR_PASSWORD
   EOF
   chmod 600 ~/.netrc
   ```

### Download

```bash
pyrosense download --n-samples 100 --output data/
```

Or using Python:
```python
from pyrosense.data import MesogeosLoader, HLSDownloader

loader = MesogeosLoader("data/mesogeos/mesogeos.zarr")
events = loader.extract_fire_events(n_samples=100)

downloader = HLSDownloader(output_dir="data/hls/")
downloader.download_for_events(events)
```

## 3. AlphaEarth Embeddings (Optional)

64-dimensional land surface embeddings from Google Earth Engine.

### Setup

1. [Sign up for Earth Engine](https://earthengine.google.com/)
2. Create a [Google Cloud Project](https://console.cloud.google.com/) and enable Earth Engine API
3. Install and authenticate:
   ```bash
   pip install pyrosense[earth-engine]
   earthengine authenticate
   ```

### Usage

```python
from pyrosense.features import AlphaEarthExtractor

extractor = AlphaEarthExtractor(project="your-project-id")
features = extractor.extract_batch(events)
```

## Directory Structure

After downloading:
```
data/
├── mesogeos/mesogeos.zarr/    # Fire labels
├── hls/                        # Satellite composites
│   ├── fire_0000/composite.tif
│   └── ...
└── features/                   # Extracted features
```

## Troubleshooting

**HLS download fails**: Check NASA Earthdata credentials in `~/.netrc`

**Earth Engine authentication**: Run `earthengine authenticate --force` to re-authenticate

**rclone connection**: Test with `rclone lsd gdrive:` to verify Google Drive access
