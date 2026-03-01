"""
PyroSense VLM Prompts
=====================

Prompt templates for fire analysis with EarthDial VLM.
"""

from __future__ import annotations

# Main analysis prompt with fire context
FIRE_ANALYSIS_PROMPT = """Act as an expert Wildland Fire Behavior Analyst (FBAN). Analyze this satellite imagery for fire risk and behavior prediction.

Context:
- Location: {lat:.4f}, {lon:.4f}
- Date: {date}
- Predictive Model Fire Probability: {probability:.1%}

Provide a highly structured, objective analysis covering:
1. Fuel Profile: Dominant vegetation types, estimated canopy cover (%), fuel continuity (patchy vs. continuous), and visible signs of drought stress or curing.
2. Topographic Hazards: Identify relative slope steepness, aspect (sun exposure), and dangerous terrain features (e.g., chimneys, saddles, steep drainages).
3. Spread Vectors: Likely direction of maximum fire spread based on topography and fuel alignment.
4. Tactical Considerations: Potential anchor points, defensible spaces, and proximity to Wildland-Urban Interface (WUI) or critical infrastructure.
5. Overall Risk Assessment: Rate as LOW, MEDIUM, HIGH, or EXTREME, followed by a 1-sentence justification.

Constraint: Format your response using clear headings and concise bullet points. State "Cannot be determined" if image resolution or cloud cover obscures specific details.
"""

# Prompts for generating structured reports
REPORT_PROMPTS = {
    "summary": (
        "Provide a strict 2-sentence bottom-line-up-front (BLUF) summary of fire risk for this satellite image. "
        "Sentence 1 must state the overall risk level and primary driving factor (fuel or terrain). "
        "Sentence 2 must highlight the most critical exposure or hazard."
    ),
    "vegetation": (
        "Act as a wildland fire fuel specialist. Analyze the vegetation in this image. "
        "Identify specific fuel types (grass, brush, timber), fuel continuity, and canopy closure. "
        "Explicitly point out areas showing discoloration indicative of drought stress, curing, or dead-and-down timber."
    ),
    "terrain": (
        "Analyze the topography in this imagery for fire behavior implications. "
        "Detail elevation gradients, prominent ridgelines, and valleys. "
        "Specifically identify terrain features that accelerate fire spread, such as narrow V-shaped canyons, steep slopes, or saddles."
    ),
    "strategies": (
        "Based on the visible landscape, suggest 3-5 operational firefighting strategies. "
        "Specifically identify: "
        "1. Ingress/egress routes for heavy equipment. "
        "2. Natural anchor points (lakes, major roads, rock outcroppings). "
        "3. Defensible terrain for establishing handlines or dozer lines. "
        "Output as a bulleted list."
    ),
    "risk": (
        "Conduct a comprehensive fire risk assessment. Evaluate the alignment of heavy fuel loads, "
        "steep terrain, and Wildland-Urban Interface (WUI) exposures. "
        "Conclude with a definitive rating (LOW, MEDIUM, HIGH, EXTREME) and list the top 3 driving factors for this rating."
    ),
	"evacuation_logistics": (
        "Act as an Emergency Management Coordinator. Evaluate the visible road network for evacuation viability. "
        "Identify primary egress routes, potential traffic choke points (e.g., narrow bridges, single-lane roads, dead ends), "
        "and note any areas where heavy continuous fuels directly threaten these escape routes."
    ),
    "anthropogenic_risk": (
        "Analyze the area for anthropogenic (human-caused) fire risks and ignition sources. "
        "Identify unmaintained lots, visible debris accumulation, industrial sites, rail lines, powerline corridors, "
        "or informal campsites/off-road trails. Assess how these elements increase ignition probability."
    ),
    "defensible_space": (
        "Assess Wildland-Urban Interface (WUI) vulnerability and defensible space compliance. "
        "Evaluate the clearance distance between visible structures and heavy vegetation. "
        "Note hazardous conditions like overhanging canopies, unmanaged brush adjacent to buildings, "
        "and the density of residential clustering."
    )
}

# Quick assessment prompts
QUICK_PROMPTS = {
    "describe": "Identify the key topographic and vegetative features in this image relevant to wildland fire spread.",
    "fuel_load": "Categorize the visible fuel load (light/flashy, medium brush, heavy timber) and assess its continuity.",
    "access": "Pinpoint primary and secondary access roads. Note any visible bottlenecks, unimproved dirt roads, or blocked routes.",
    "structures": "Scan for Wildland-Urban Interface (WUI). Identify any residential, commercial, or infrastructural assets at risk.",
    "water": "Locate drafting sources: identify ponds, lakes, rivers, or large agricultural tanks suitable for helicopter dip sites or engine drafting.",
    "firebreaks": "List distinct firebreaks. Differentiate between natural (rivers, rock slides) and artificial (highways, clear cuts).","choke_points": "Identify critical egress bottlenecks for evacuation, such as dead-end roads, narrow bridges, or single-access neighborhoods.",
    "ignition_sources": "Pinpoint potential human ignition sources: industrial facilities, powerline clearings, rail lines, or active agricultural/slash burning.",
    "unmaintained_lots": "Detect parcels with hazardous fuel accumulation near structures, such as unmanaged brush, overgrown grass, debris piles, or visibly abandoned properties.",
    "population_density": "Categorize the human footprint (e.g., high-density subdivision, scattered rural residential, industrial, or unpopulated wildland)."
}