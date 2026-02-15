#!/usr/bin/env python3
"""
Download Mesogeos Dataset
=========================

Downloads the real Mesogeos dataset from Google Drive.

The dataset is available at:
https://drive.google.com/drive/folders/1aRXQXVvw6hz0eYgtJDoixjPQO-_bRKz9
"""

import subprocess
import sys
from pathlib import Path
from loguru import logger

logger.remove()
logger.add(lambda msg: print(msg, end=""), format="{time:HH:mm:ss} | {message}", level="INFO")


def setup_rclone_gdrive():
    """Check if rclone is configured for Google Drive."""

    logger.info("Checking rclone configuration...")

    # Check if gdrive remote exists
    result = subprocess.run(["rclone", "listremotes"], capture_output=True, text=True)

    if "gdrive:" in result.stdout:
        logger.info("Google Drive remote 'gdrive' already configured")
        return True

    logger.info("""
rclone needs to be configured for Google Drive access.

Please run this command in your terminal:

    rclone config

Then follow these steps:
1. Select 'n' for new remote
2. Name it: gdrive
3. Select 'drive' for Google Drive
4. Leave client_id blank (press Enter)
5. Leave client_secret blank (press Enter)
6. Select '1' for full access
7. Leave root_folder_id blank
8. Leave service_account_file blank
9. Select 'n' for advanced config
10. Select 'y' for auto config
11. A browser will open - sign in to Google and authorize
12. Select 'n' for team drive
13. Select 'y' to confirm
14. Select 'q' to quit config

After setup, run this script again.
""")
    return False


def download_mesogeos_subset():
    """Download a subset of Mesogeos for testing."""

    output_dir = Path("data/mesogeos")
    output_dir.mkdir(parents=True, exist_ok=True)

    # The Mesogeos datacube folder ID (from the Google Drive link)
    # Full path: 1aRXQXVvw6hz0eYgtJDoixjPQO-_bRKz9
    gdrive_folder = "1aRXQXVvw6hz0eYgtJDoixjPQO-_bRKz9"

    logger.info("=" * 60)
    logger.info("Downloading Mesogeos Dataset")
    logger.info("=" * 60)
    logger.info(f"Source: Google Drive folder {gdrive_folder}")
    logger.info(f"Destination: {output_dir}")
    logger.info("")
    logger.info("This may take 10-30 minutes depending on your connection...")
    logger.info("")

    # First, list contents to see what's available
    logger.info("Listing available files...")
    list_cmd = [
        "rclone", "lsf",
        f"gdrive:--drive-shared-with-me/{gdrive_folder}",
        "--max-depth", "1"
    ]

    try:
        result = subprocess.run(list_cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            # Try alternative method - direct folder access
            list_cmd = ["rclone", "lsf", f"gdrive:{gdrive_folder}", "--max-depth", "1"]
            result = subprocess.run(list_cmd, capture_output=True, text=True, timeout=60)

        if result.stdout:
            logger.info(f"Found files:\n{result.stdout}")
        else:
            logger.warning("Could not list files. Trying direct download...")

    except Exception as e:
        logger.warning(f"Listing failed: {e}")

    # Download the zarr datacube
    # We'll download just the essential variables to save space
    logger.info("")
    logger.info("Starting download of mesogeos_cube.zarr...")
    logger.info("(This is a large dataset, downloading may take a while)")

    download_cmd = [
        "rclone", "copy",
        f"gdrive:{gdrive_folder}/mesogeos_cube.zarr",
        str(output_dir / "mesogeos.zarr"),
        "--progress",
        "--transfers", "4",
        "-v"
    ]

    try:
        # Run with real-time output
        process = subprocess.Popen(
            download_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        for line in process.stdout:
            print(line, end="")

        process.wait()

        if process.returncode == 0:
            logger.info("Download complete!")
            return output_dir / "mesogeos.zarr"
        else:
            logger.error(f"Download failed with code {process.returncode}")
            return None

    except Exception as e:
        logger.error(f"Download error: {e}")
        return None


def main():
    # Check rclone setup
    if not setup_rclone_gdrive():
        sys.exit(1)

    # Download
    result = download_mesogeos_subset()

    if result:
        logger.info("")
        logger.info("=" * 60)
        logger.info("SUCCESS!")
        logger.info("=" * 60)
        logger.info(f"Mesogeos data saved to: {result}")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Run the HLS download script")
        logger.info("  2. Update notebook paths")
        logger.info("  3. Run the experiment")
    else:
        logger.error("Download failed. Please try manual download from:")
        logger.error("https://drive.google.com/drive/folders/1aRXQXVvw6hz0eYgtJDoixjPQO-_bRKz9")


if __name__ == "__main__":
    main()
