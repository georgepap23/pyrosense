#!/usr/bin/env python3
"""
Download HLS imagery for negative events from DIFFERENT locations.

This script:
1. Loads existing fire events from Mesogeos
2. Samples negative events from locations that never burned
3. Downloads HLS satellite imagery for negative events
4. Saves the complete event list to fire_events.csv
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pyrosense.data.mesogeos_loader import MesogeosLoader, save_events_csv, load_events_csv
from pyrosense.data.hls_downloader import HLSDownloader

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"
MESOGEOS_PATH = DATA_DIR / "mesogeos/mesogeos.zarr"
HLS_DIR = DATA_DIR / "hls"
EVENTS_CACHE = DATA_DIR / "fire_events.csv"

N_SAMPLES = 100
MIN_DISTANCE_DEG = 0.1  # ~10km minimum distance from any fire location


def main():
    print("=" * 60)
    print("PyroSense: Download No-Fire HLS from Different Locations")
    print("=" * 60)

    # Step 1: Extract fire events (or load from existing HLS)
    print("\n[1/4] Loading fire events...")
    loader = MesogeosLoader(str(MESOGEOS_PATH), region="greece")

    fire_events = loader.extract_fire_events(
        n_samples=N_SAMPLES,
        min_burned_area=0.0,
        start_year=2015,
        end_year=2021,
        random_seed=42,
    )
    print(f"  Fire events: {len(fire_events)}")

    # Step 2: Sample negative events from DIFFERENT locations
    print("\n[2/4] Sampling negative events from different locations...")
    print(f"  Min distance from fire: {MIN_DISTANCE_DEG}° (~{MIN_DISTANCE_DEG * 111:.0f} km)")

    negative_events = loader.sample_negative_events_different_locations(
        n_samples=N_SAMPLES,
        fire_events=fire_events,
        min_distance_deg=MIN_DISTANCE_DEG,
        random_seed=42,
    )
    print(f"  Negative events: {len(negative_events)}")

    # Combine and save
    all_events = fire_events + negative_events
    save_events_csv(all_events, str(EVENTS_CACHE))
    print(f"  Saved to: {EVENTS_CACHE}")

    # Step 3: Download HLS for negative events only (fire events already downloaded)
    print("\n[3/4] Downloading HLS imagery for negative events...")
    downloader = HLSDownloader(
        output_dir=str(HLS_DIR),
        days_before=30,
        min_days_before=7,
    )

    # Only download for negative events
    successes, failures = downloader.download_for_events(negative_events)

    print(f"\n[4/4] Download complete!")
    print(f"  Success: {len(successes)}/{len(negative_events)}")
    print(f"  Failed:  {len(failures)}/{len(negative_events)}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Total events: {len(all_events)}")
    print(f"    Fire:    {len(fire_events)}")
    print(f"    No-fire: {len(negative_events)}")
    print(f"  Events CSV: {EVENTS_CACHE}")
    print(f"  HLS dir:    {HLS_DIR}")
    print("\nNext step: Run the notebook to extract features and train!")


if __name__ == "__main__":
    main()
