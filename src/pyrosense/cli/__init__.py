"""
Command-Line Interface
======================

PyroSense CLI for training and prediction.

Usage:
    pyrosense train --config configs/stacking.yaml
    pyrosense predict --model outputs/model.pkl --lat 38.0 --lon 23.5 --date 2023-07-15
    pyrosense download --n-samples 100 --output data/
"""

from pyrosense.cli.main import cli

__all__ = ["cli"]
