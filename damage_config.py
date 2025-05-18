#!/usr/bin/env python
"""
Configuration file for car damage assessment parameters.
This file centralizes damage assessment parameters to ensure consistency across the application.
"""

# Damage type severity weights (higher means more expensive to repair)
DAMAGE_SEVERITY = {
    "scratch": 0.3,        # Usually just needs paint touch-up
    "dent": 0.8,           # Moderate repair cost for panel work
    "crack": 1.5,          # More significant repair needed
    "glass shatter": 2.0,  # Glass replacement cost
    "lamp broken": 1.5,    # Light assembly replacement
    "tire flat": 0.6       # Relatively inexpensive to replace tire
}

# Component importance weights (higher means more expensive to repair)
COMPONENT_IMPORTANCE = {
    "hood": 2.0,           # Large panel, expensive to replace
    "door": 1.5,           # Moderately expensive, contains electronics
    "bumper": 1.2,         # Usually plastic, moderately priced
    "fender": 1.0,         # Relatively simple panel replacement
    "headlight": 1.8,      # Modern headlights are expensive to replace
    "taillight": 1.2,      # Usually less expensive than headlights
    "windshield": 2.5,     # Large glass component, expensive
    "wheel": 1.0,          # Moderate cost to replace
    "engine": 8.0,         # Most expensive component to repair/replace
    "default": 1.0         # Default weight if component not identified
}

# Small damage adjustment settings
SMALL_DAMAGE_THRESHOLD = 5.0      # Percentage below which a damage is considered "small"
VERY_SMALL_DAMAGE_THRESHOLD = 0.5  # Raw percentage below which damage is considered "very small"
VERY_SMALL_DAMAGE_FACTOR = 0.2    # Factor to apply to very small damages
MIN_DAMAGE_REPORT = 0.5           # Minimum damage percentage to report (even for tiny damages)

# Maximum values - reduced to prevent tire flat from always being 70%+
MAX_INDIVIDUAL_DAMAGE = 50.0      # Cap for individual damage contribution
MAX_TOTAL_DAMAGE = 100.0          # Cap for total damage percentage 