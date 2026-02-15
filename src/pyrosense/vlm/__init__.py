"""
PyroSense Vision-Language Module
================================

Integration with EarthDial VLM for interactive fire area analysis.

Classes:
    EarthDialAssistant: Main VLM wrapper for fire analysis
"""

from __future__ import annotations

from pyrosense.vlm.earthdial import EarthDialAssistant
from pyrosense.vlm.image_utils import hls_to_rgb
from pyrosense.vlm.prompts import FIRE_ANALYSIS_PROMPT, REPORT_PROMPTS

__all__ = [
    "EarthDialAssistant",
    "hls_to_rgb",
    "FIRE_ANALYSIS_PROMPT",
    "REPORT_PROMPTS",
]
