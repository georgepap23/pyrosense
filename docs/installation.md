# Installation & CLI Guide

## Installation

```bash
# Clone the repository
git clone https://github.com/georgepap23/pyrosense.git
cd pyrosense

# Basic installation
pip install -e .

# With Earth Engine support (for AlphaEarth features)
pip install -e ".[earth-engine]"

# With EarthDial VLM support
pip install -e ".[earthdial]"
```

### EarthDial Setup

For EarthDial VLM, also clone the model repository:

```bash
cd /tmp && git clone https://github.com/hiyamdebary/EarthDial.git
```

Or use a custom location:
```bash
git clone https://github.com/hiyamdebary/EarthDial.git ~/models/EarthDial
export EARTHDIAL_SRC_PATH=~/models/EarthDial/src
```

## CLI Commands

### Training

```bash
# Train a stacking ensemble model
pyrosense train --config configs/stacking.yaml --output outputs/

# Skip download/extraction if data already exists
pyrosense train --config configs/stacking.yaml --skip-download --skip-extraction
```

### Prediction

```bash
# Predict fire probability for a location
pyrosense predict --model outputs/stacking_ensemble.pkl \
    --lat 38.0 --lon 23.5 --date 2023-07-15
```

### Data Download

```bash
# Download HLS imagery for fire events
pyrosense download --n-samples 100 --output data/

# Specify Mesogeos path
pyrosense download --n-samples 100 --mesogeos-path data/mesogeos/mesogeos.zarr
```

### Feature Extraction

```bash
# Extract all features
pyrosense features --events data/fire_events.csv --source all

# Extract specific source
pyrosense features --events data/fire_events.csv --source prithvi
pyrosense features --events data/fire_events.csv --source weather
pyrosense features --events data/fire_events.csv --source alphaearth
```

### EarthDial Analysis

```bash
# Single image analysis
pyrosense analyze -i data/hls/fire_0001/composite.tif -q "Assess fire risk"

# With coordinates for precise cropping
pyrosense analyze -i data/hls/fire_0001/composite.tif --lat 38.5 --lon 23.1

# Interactive chat session
pyrosense chat -i data/hls/fire_0001/composite.tif

# Specify device
pyrosense analyze -i image.tif --device mps    # Mac
pyrosense analyze -i image.tif --device cuda   # NVIDIA GPU
pyrosense analyze -i image.tif --device cpu    # CPU only
```

## Python API Quick Start

```python
from pyrosense.data import MesogeosLoader, HLSDownloader
from pyrosense.features import PrithviExtractor, WeatherExtractor, FeatureStore
from pyrosense.models import StackingEnsemble, StackingConfig

# 1. Load fire events
loader = MesogeosLoader("data/mesogeos/mesogeos.zarr")
fire_events = loader.extract_fire_events(n_samples=100)
negative_events = loader.sample_negative_events(fire_events)

# 2. Download HLS imagery
downloader = HLSDownloader(output_dir="data/hls/")
downloader.download_for_events(fire_events + negative_events)

# 3. Extract features
store = FeatureStore("data/features/")
prithvi = PrithviExtractor(hls_dir="data/hls/")
store.save("prithvi", prithvi.extract_batch(fire_events + negative_events))

# 4. Train model
X = store.get_combined(["prithvi", "weather"])
y = [1.0 if e.burned_area > 0 else 0.0 for e in fire_events + negative_events]

config = StackingConfig.simple(sources=["prithvi", "weather"])
ensemble = StackingEnsemble(config)
ensemble.fit(X, y)
```

## EarthDial Python API

```python
from pyrosense.vlm import EarthDialAssistant

# Initialize (auto-detects GPU/MPS/CPU)
assistant = EarthDialAssistant(device="auto")

# Single analysis
response = assistant.analyze_image(
    "data/hls/fire_0001/composite.tif",
    question="What vegetation do you see?"
)

# Generate full report
report = assistant.generate_report(
    "data/hls/fire_0001/composite.tif",
    fire_probability=0.85
)
print(report["summary"])
print(report["vegetation_analysis"])
print(report["recommended_strategies"])

# Interactive chat
history = None
response, history = assistant.chat("image.tif", "What terrain features are visible?", history)
response, history = assistant.chat("image.tif", "How might fire spread here?", history)
```
