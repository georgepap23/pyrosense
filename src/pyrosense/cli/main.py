"""
PyroSense Command-Line Interface
================================

CLI commands for wildfire prediction pipeline.

Commands:
    train     - Train a stacking ensemble model
    predict   - Make predictions for a location
    download  - Download HLS imagery
    features  - Extract features from events
"""

from __future__ import annotations

import warnings
# Suppress warnings for cleaner CLI output
warnings.filterwarnings("ignore", message=".*found in sys.modules.*")
warnings.filterwarnings("ignore", category=UserWarning)  # sklearn CV warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)  # numpy divide warnings

import sys
import json
import joblib
from pathlib import Path
from datetime import datetime

import click
import yaml
import numpy as np
import pandas as pd
from loguru import logger

# Configure loguru for CLI: only show warnings and errors
logger.remove()
logger.add(sys.stderr, level="WARNING")


@click.group()
@click.version_option(message="%(prog)s %(version)s")
def cli():
    """PyroSense: Wildfire prediction with foundation models."""
    pass


@cli.command()
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    default="configs/stacking.yaml",
    help="Path to configuration file",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default="outputs/",
    help="Output directory for model and results",
)
@click.option(
    "--skip-download",
    is_flag=True,
    help="Skip HLS imagery download (use cached)",
)
@click.option(
    "--skip-extraction",
    is_flag=True,
    help="Skip feature extraction (use cached)",
)
def train(config: str, output: str, skip_download: bool, skip_extraction: bool):
    """Train a stacking ensemble model."""
    from pyrosense.data import MesogeosLoader, HLSDownloader, load_events_csv, save_events_csv
    from pyrosense.features import PrithviExtractor, WeatherExtractor, FeatureStore
    from pyrosense.models import StackingEnsemble, StackingConfig
    from pyrosense.evaluation import evaluate_classifier

    from sklearn.model_selection import train_test_split

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    click.echo(f"Loading config: {config}")
    with open(config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get("data", {})
    feature_cfg = cfg.get("features", {})
    model_cfg = cfg.get("model", {})

    # Step 1: Load or extract fire events
    events_cache = Path(data_cfg.get("events_cache", "data/fire_events.csv"))
    mesogeos_path = data_cfg.get("mesogeos_path", "data/mesogeos/mesogeos.zarr")

    if events_cache.exists():
        all_events = load_events_csv(str(events_cache))
    else:
        click.echo("Extracting events from Mesogeos...")
        loader = MesogeosLoader(
            zarr_path=mesogeos_path,
            region=data_cfg.get("region", "greece"),
        )
        fire_events = loader.extract_fire_events(
            n_samples=data_cfg.get("n_samples", 100),
            start_year=data_cfg.get("start_year", 2015),
            end_year=data_cfg.get("end_year", 2021),
        )
        # Sample negative events from DIFFERENT locations (no spatial leakage)
        negative_events = loader.sample_negative_events_different_locations(
            n_samples=data_cfg.get("n_samples", 100),
            fire_events=fire_events,
            min_distance_deg=data_cfg.get("min_distance_deg", 0.1),
        )
        all_events = fire_events + negative_events
        save_events_csv(all_events, str(events_cache))

    fire_events = [e for e in all_events if e.burned_area > 0]
    negative_events = [e for e in all_events if e.burned_area == 0]
    click.echo(f"Events: {len(fire_events)} fire, {len(negative_events)} no-fire")

    # Step 2: Download HLS imagery (if not skipped)
    if not skip_download:
        click.echo("Downloading HLS imagery...")
        hls_dir = data_cfg.get("hls_dir", "data/hls/")
        downloader = HLSDownloader(
            output_dir=hls_dir,
            days_before=feature_cfg.get("prithvi", {}).get("days_before", 14),
        )
        downloader.download_for_events(all_events)

    # Step 3: Extract features (if not skipped)
    store = FeatureStore(data_cfg.get("feature_store", "data/features/"))

    if not skip_extraction or not store.exists("prithvi"):
        # Extract Prithvi features
        if feature_cfg.get("prithvi", {}).get("enabled", True):
            click.echo("Extracting Prithvi features...")
            prithvi = PrithviExtractor(
                hls_dir=data_cfg.get("hls_dir", "data/hls/"),
            )
            prithvi_df = prithvi.extract_batch(all_events)
            store.save("prithvi", prithvi_df)

        # Extract weather features
        if feature_cfg.get("weather", {}).get("enabled", True):
            click.echo("Extracting Weather features...")
            weather = WeatherExtractor(
                days_before=feature_cfg.get("weather", {}).get("days_before", 7),
            )
            weather_df = weather.extract_batch(all_events)
            store.save("weather", weather_df)

    # Step 4: Load and combine features
    sources = []
    if feature_cfg.get("prithvi", {}).get("enabled", True):
        sources.append("prithvi")
    if feature_cfg.get("weather", {}).get("enabled", True):
        sources.append("weather")
    if feature_cfg.get("alphaearth", {}).get("enabled", False):
        sources.append("alphaearth")

    X = store.get_combined(sources)
    y = np.array([1.0 if e.burned_area > 0 else 0.0 for e in all_events])

    # Align X and y
    event_ids = [e.event_id for e in all_events]
    X = X.loc[X.index.isin(event_ids)]
    y = np.array([
        1.0 if e.burned_area > 0 else 0.0
        for e in all_events if e.event_id in X.index
    ])

    click.echo(f"Features: {X.shape[1]} dims, {X.shape[0]} samples")

    # Step 5: Train/test split
    test_size = data_cfg.get("test_size", 0.2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=42,
    )

    # Step 6: Train stacking ensemble
    click.echo("Training stacking ensemble...")
    config_obj = StackingConfig.simple(sources)
    ensemble = StackingEnsemble(config_obj)
    ensemble.fit(X_train, y_train)

    # Step 7: Evaluate
    results = evaluate_classifier(
        ensemble, X_test, y_test,
        X_train=X_train, y_train=y_train,
    )

    click.echo("")
    click.echo("Results:")
    click.echo(f"  AUC:      {results.auc:.4f}")
    click.echo(f"  Accuracy: {results.accuracy:.4f}")
    click.echo(f"  F1:       {results.f1:.4f}")

    # Step 8: Save model and results
    model_path = output_dir / "stacking_ensemble.pkl"
    joblib.dump(ensemble, model_path)
    click.echo(f"\nModel saved: {model_path}")

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump({
            **results.to_dict(),
            "timestamp": datetime.now().isoformat(),
            "config": cfg,
        }, f, indent=2)


@cli.command()
@click.option(
    "--model", "-m",
    type=click.Path(exists=True),
    required=True,
    help="Path to trained model",
)
@click.option("--lat", type=float, required=True, help="Latitude")
@click.option("--lon", type=float, required=True, help="Longitude")
@click.option(
    "--date", "-d",
    type=str,
    required=True,
    help="Date (YYYY-MM-DD)",
)
def predict(model: str, lat: float, lon: float, date: str):
    """Make a prediction for a single location."""
    from pyrosense.data import FireEvent
    from pyrosense.features import PrithviExtractor, WeatherExtractor

    # Load model
    ensemble = joblib.load(model)

    # Create event
    event = FireEvent(
        event_id="query",
        latitude=lat,
        longitude=lon,
        date=pd.Timestamp(date),
        burned_area=0.0,
    )

    # Extract features
    logger.info(f"Extracting features for ({lat}, {lon}) on {date}")

    features = {}

    # Prithvi features
    prithvi = PrithviExtractor()
    prithvi_result = prithvi.extract(event)
    for name, val in zip(prithvi_result.feature_names, prithvi_result.features):
        features[f"prithvi_{name}"] = val

    # Weather features
    weather = WeatherExtractor()
    weather_result = weather.extract(event)
    for name, val in zip(weather_result.feature_names, weather_result.features):
        features[f"weather_{name}"] = val

    # Create DataFrame
    X = pd.DataFrame([features])

    # Predict
    probas = ensemble.predict_proba(X)
    fire_prob = probas[0, 1]

    click.echo(f"\nPrediction for ({lat}, {lon}) on {date}:")
    click.echo(f"  Fire probability: {fire_prob:.2%}")
    click.echo(f"  Risk level: {'HIGH' if fire_prob > 0.7 else 'MEDIUM' if fire_prob > 0.3 else 'LOW'}")


@cli.command()
@click.option(
    "--n-samples", "-n",
    type=int,
    default=100,
    help="Number of fire events to sample",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default="data/",
    help="Output directory",
)
@click.option(
    "--mesogeos-path",
    type=click.Path(exists=True),
    default="data/mesogeos/mesogeos.zarr",
    help="Path to Mesogeos datacube",
)
def download(n_samples: int, output: str, mesogeos_path: str):
    """Download HLS imagery for fire events."""
    from pyrosense.data import MesogeosLoader, HLSDownloader, save_events_csv

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract events
    logger.info(f"Extracting {n_samples} fire events from Mesogeos")
    loader = MesogeosLoader(zarr_path=mesogeos_path)
    fire_events = loader.extract_fire_events(n_samples=n_samples)
    negative_events = loader.sample_negative_events_different_locations(
        n_samples=n_samples, fire_events=fire_events, min_distance_deg=0.1
    )

    all_events = fire_events + negative_events
    save_events_csv(all_events, str(output_dir / "fire_events.csv"))

    # Download HLS
    logger.info(f"Downloading HLS imagery for {len(all_events)} events")
    downloader = HLSDownloader(output_dir=str(output_dir / "hls"))
    successes, failures = downloader.download_for_events(all_events)

    click.echo(f"\nDownload complete:")
    click.echo(f"  Success: {len(successes)}")
    click.echo(f"  Failed: {len(failures)}")


@cli.command()
@click.option(
    "--events", "-e",
    type=click.Path(exists=True),
    required=True,
    help="Path to events CSV",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default="data/features/",
    help="Output directory for features",
)
@click.option(
    "--source", "-s",
    type=click.Choice(["prithvi", "weather", "alphaearth", "all"]),
    default="all",
    help="Feature source to extract",
)
def features(events: str, output: str, source: str):
    """Extract features from events."""
    from pyrosense.data import load_events_csv
    from pyrosense.features import PrithviExtractor, WeatherExtractor, FeatureStore

    # Load events
    all_events = load_events_csv(events)
    logger.info(f"Loaded {len(all_events)} events")

    store = FeatureStore(output)
    sources_to_extract = ["prithvi", "weather"] if source == "all" else [source]

    for src in sources_to_extract:
        logger.info(f"Extracting {src} features")

        if src == "prithvi":
            extractor = PrithviExtractor()
        elif src == "weather":
            extractor = WeatherExtractor()
        elif src == "alphaearth":
            try:
                from pyrosense.features import AlphaEarthExtractor
                extractor = AlphaEarthExtractor()
            except ImportError:
                logger.error("AlphaEarth requires earthengine-api. Install with: pip install earthengine-api")
                continue
        else:
            continue

        df = extractor.extract_batch(all_events)
        store.save(src, df)

    click.echo(f"\nFeatures saved to {output}")


@cli.command()
@click.option(
    "--image", "-i",
    type=click.Path(exists=True),
    required=True,
    help="Path to HLS composite or RGB image",
)
@click.option("--lat", type=float, default=None, help="Latitude (optional)")
@click.option("--lon", type=float, default=None, help="Longitude (optional)")
@click.option(
    "--model-path", "-m",
    type=click.Path(exists=True),
    default=None,
    help="Path to trained stacking model (for fire probability)",
)
@click.option(
    "--question", "-q",
    type=str,
    default=None,
    help="Custom question (default: general fire risk analysis)",
)
@click.option(
    "--device",
    type=click.Choice(["auto", "cuda", "mps", "cpu"]),
    default="auto",
    help="Device for VLM inference",
)
@click.option(
    "--load-8bit",
    is_flag=True,
    help="Use 8-bit quantization (lower memory)",
)
def analyze(image: str, lat: float, lon: float, model_path: str, question: str, device: str, load_8bit: bool):
    """Analyze a satellite image with EarthDial VLM."""
    # Suppress verbose warnings for cleaner output
    import warnings
    import logging
    import os
    warnings.filterwarnings("ignore")
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        from pyrosense.vlm import EarthDialAssistant
    except ImportError:
        click.echo("EarthDial requires additional dependencies.")
        click.echo("Install with: pip install pyrosense[earthdial]")
        return

    image_path = Path(image)
    click.echo(f"Analyzing: {image_path.name}")

    # Initialize EarthDial
    assistant = EarthDialAssistant(
        device=device,
        load_in_8bit=load_8bit,
    )

    # Show device info
    info = assistant.device_info
    click.echo(f"Device: {info['device']}")

    # Default question
    if question is None:
        question = "Analyze this satellite image and assess fire risk factors. Describe the vegetation, terrain, and any features relevant to fire behavior."

    # Add context if lat/lon provided
    if lat is not None and lon is not None:
        question = f"Location: {lat:.4f}, {lon:.4f}\n\n{question}"

    click.echo("\nLoading EarthDial model...")

    try:
        # Pass lat/lon for proper cropping (crops to 6.7 km x 6.7 km area)
        response = assistant.analyze_image(
            image_path, question,
            center_lat=lat, center_lon=lon
        )
        click.echo("\n" + "=" * 60)
        click.echo("EarthDial Analysis:")
        click.echo("=" * 60)
        click.echo(response)
        click.echo("=" * 60)
    except Exception as e:
        click.echo(f"Analysis failed: {e}")
        raise


@cli.command()
@click.option(
    "--image", "-i",
    type=click.Path(exists=True),
    required=True,
    help="Path to satellite image",
)
@click.option("--lat", type=float, default=None, help="Latitude for cropping (optional)")
@click.option("--lon", type=float, default=None, help="Longitude for cropping (optional)")
@click.option(
    "--device",
    type=click.Choice(["auto", "cuda", "mps", "cpu"]),
    default="auto",
    help="Device for VLM inference",
)
@click.option(
    "--load-8bit",
    is_flag=True,
    help="Use 8-bit quantization (lower memory)",
)
def chat(image: str, lat: float, lon: float, device: str, load_8bit: bool):
    """Interactive chat about a satellite image with EarthDial."""
    # Suppress verbose warnings for cleaner output
    import warnings
    import logging
    import os
    warnings.filterwarnings("ignore")
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        from pyrosense.vlm import EarthDialAssistant
    except ImportError:
        click.echo("EarthDial requires additional dependencies.")
        click.echo("Install with: pip install pyrosense[earthdial]")
        return

    image_path = Path(image)
    click.echo(f"Image: {image_path.name}")
    click.echo("Loading EarthDial model...")

    # Initialize EarthDial
    assistant = EarthDialAssistant(
        device=device,
        load_in_8bit=load_8bit,
    )

    # Show device info
    info = assistant.device_info
    click.echo(f"Device: {info['device']}")
    if lat is not None and lon is not None:
        click.echo(f"Cropping to: {lat:.4f}, {lon:.4f} (6.7 km × 6.7 km area)")
    click.echo("\nEarthDial ready. Type 'quit' or 'exit' to end.\n")

    history = None

    while True:
        try:
            question = click.prompt("You", type=str)
        except click.Abort:
            break

        if question.lower() in ("quit", "exit", "q"):
            break

        if not question.strip():
            continue

        try:
            response, history = assistant.chat(
                image_path, question, history,
                center_lat=lat, center_lon=lon
            )
            click.echo(f"\nEarthDial: {response}\n")
        except Exception as e:
            click.echo(f"Error: {e}")
            continue

    click.echo("Goodbye!")


if __name__ == "__main__":
    cli()
