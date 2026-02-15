"""
PyroSense VLM Prompts
=====================

Prompt templates for fire analysis with EarthDial VLM.
"""

from __future__ import annotations

# Main analysis prompt with fire context
FIRE_ANALYSIS_PROMPT = """Analyze this satellite image of a potential fire risk area.

Location: {lat:.4f}, {lon:.4f}
Date: {date}
Model Fire Probability: {probability:.1%}

Please provide:
1. Vegetation type and density visible
2. Terrain characteristics
3. Potential fire spread factors
4. Recommended firefighting strategies
5. Overall risk assessment
"""

# Prompts for generating structured reports
REPORT_PROMPTS = {
    "summary": (
        "Provide a 2-sentence summary of fire risk for this satellite image. "
        "Focus on vegetation density and terrain factors."
    ),
    "vegetation": (
        "Describe the vegetation visible in this satellite image. "
        "Identify vegetation types, density, and any patterns. "
        "Note any areas that appear dry or stressed."
    ),
    "terrain": (
        "Analyze the terrain and topography visible in this satellite image. "
        "Describe elevation changes, slopes, valleys, and ridges. "
        "Note how terrain might affect fire behavior and spread."
    ),
    "strategies": (
        "Based on this satellite image, suggest 3-5 firefighting strategies. "
        "Consider access routes, natural firebreaks, water sources, "
        "and defensible positions visible in the imagery."
    ),
    "risk": (
        "Provide a fire risk assessment for this area based on the satellite image. "
        "Consider fuel load, moisture, terrain, and any visible structures or infrastructure. "
        "Rate the risk as LOW, MEDIUM, HIGH, or EXTREME with justification."
    ),
}

# Quick assessment prompts
QUICK_PROMPTS = {
    "describe": "Describe what you see in this satellite image.",
    "fuel_load": "Assess the fuel load (vegetation) visible in this image for fire risk.",
    "access": "Identify potential access routes and roads visible in this image.",
    "structures": "Are there any buildings or structures visible in this image?",
    "water": "Are there any water bodies or potential water sources visible?",
    "firebreaks": "Identify any natural or artificial firebreaks in this image.",
}
