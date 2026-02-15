# Project Structure

```
pyrosense/
├── pyproject.toml           # Package configuration
├── src/pyrosense/           # Main package
│   ├── data/                # Data loading
│   │   ├── mesogeos_loader.py   # Mesogeos datacube loader
│   │   ├── hls_downloader.py    # HLS satellite imagery
│   │   └── fire_event.py        # FireEvent dataclass
│   ├── features/            # Feature extractors
│   │   ├── base.py          # Abstract base class
│   │   ├── prithvi.py       # Prithvi foundation model (1024-dim)
│   │   ├── weather.py       # Weather features (10-dim)
│   │   ├── alphaearth.py    # AlphaEarth embeddings (64-dim)
│   │   └── store.py         # Feature caching (Parquet)
│   ├── models/              # ML models
│   │   ├── stacking.py      # Stacking ensemble
│   │   └── classifiers.py   # Classifier factory (RF, XGB, LR)
│   ├── vlm/                 # Vision-Language Model
│   │   ├── earthdial.py     # EarthDial integration
│   │   ├── prompts.py       # Fire analysis prompts
│   │   └── image_utils.py   # HLS to RGB conversion
│   ├── evaluation/          # Metrics and evaluation
│   └── cli/                 # Command-line interface
│       └── main.py          # CLI commands
├── notebooks/               # Jupyter notebooks
│   └── 01_prithvi_value_test.ipynb  # Main experiment notebook
├── configs/                 # Configuration files
│   └── stacking.yaml        # Stacking ensemble config
├── docs/                    # Documentation
│   ├── data_download.md     # Data download guide
│   ├── CONTRIBUTING.md      # Contribution guidelines
│   └── CHANGELOG.md         # Version history
└── assets/                  # Images and media
    └── pipeline.png         # Architecture diagram
```

## Key Modules

### Data (`src/pyrosense/data/`)

- **MesogeosLoader**: Loads fire events from the Mesogeos Zarr datacube
- **HLSDownloader**: Downloads HLS satellite composites from NASA Earthdata
- **FireEvent**: Dataclass representing a fire/no-fire event with location and date

### Features (`src/pyrosense/features/`)

- **PrithviExtractor**: Extracts 1024-dim embeddings using NASA Prithvi-EO-2.0-300M
- **WeatherExtractor**: Fetches weather data from Open-Meteo API
- **AlphaEarthExtractor**: Queries Google Earth Engine for AlphaEarth embeddings
- **FeatureStore**: Caches extracted features in Parquet format

### Models (`src/pyrosense/models/`)

- **StackingEnsemble**: Two-level stacking architecture
  - Level 1: Source-specific base models (RandomForest)
  - Level 2: Meta-learner (LogisticRegression)
- **ClassifierConfig**: Factory for creating sklearn classifiers

### VLM (`src/pyrosense/vlm/`)

- **EarthDialAssistant**: Vision-language model for fire area analysis
- Supports single-turn analysis and multi-turn chat
- Generates structured fire risk reports

### CLI (`src/pyrosense/cli/`)

Commands:
- `pyrosense train` - Train stacking ensemble
- `pyrosense predict` - Make predictions
- `pyrosense download` - Download HLS imagery
- `pyrosense features` - Extract features
- `pyrosense analyze` - Analyze image with EarthDial
- `pyrosense chat` - Interactive chat with EarthDial
