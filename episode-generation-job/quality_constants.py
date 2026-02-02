#!/usr/bin/env python3
"""
Quality Threshold Constants for BlueprintPipeline.

This module provides a single source of truth for all quality thresholds
used across the episode generation pipeline.

LABS-BLOCKER-002 FIX: Unified quality thresholds to ensure consistency
across validation, certification, and episode generation.
"""

# =============================================================================
# Quality Score Thresholds (0.0 - 1.0 scale)
# =============================================================================

# Minimum quality score for episodes to be included in datasets
# This is used by:
# - SimulationValidator to filter episodes
# - EpisodeGenerator to decide which episodes to keep
# - Quality gates in the pipeline
MIN_QUALITY_SCORE = 0.85  # 85% - minimum for lab delivery

# Quality score for production training
# Episodes above this threshold are suitable for primary model training
# This is used by:
# - QualityCertificate to assess training suitability
PRODUCTION_TRAINING_THRESHOLD = 0.90  # 90% - production ready

# Quality score for fine-tuning
# Episodes between this and MIN_QUALITY_SCORE can be used for fine-tuning
# but not for primary training
# This is used by:
# - QualityCertificate to assess training suitability
FINE_TUNING_THRESHOLD = 0.80  # 80% - fine-tuning acceptable

# Quality score for development/testing
# Episodes below FINE_TUNING_THRESHOLD are only suitable for testing
DEVELOPMENT_THRESHOLD = 0.60  # 60% - development/testing only


# =============================================================================
# Component-Specific Quality Thresholds
# =============================================================================

# Physics validation quality
# Used by SimulationValidator for physics-based checks
PHYSICS_QUALITY_MIN = MIN_QUALITY_SCORE  # Must match overall minimum

# Trajectory quality
# Used for motion planning and trajectory validation
TRAJECTORY_QUALITY_MIN = MIN_QUALITY_SCORE  # Must match overall minimum

# Sensor data quality
# Used for validating RGB/depth/segmentation quality
SENSOR_DATA_QUALITY_MIN = MIN_QUALITY_SCORE  # Must match overall minimum


# =============================================================================
# Validation Constants
# =============================================================================

# Maximum retries for failed episode generation
MAX_RETRIES = 3

# Stability threshold for object placement (meters)
STABILITY_THRESHOLD = 0.001  # 1mm position change

# Object displacement threshold for pick/place success (meters)
OBJECT_DISPLACEMENT_THRESHOLD = 0.01  # 1cm minimum displacement

# Placement tolerance for final object pose (meters)
PLACEMENT_POSITION_TOLERANCE = 0.05  # 5cm placement tolerance

# Collision penetration threshold (meters)
COLLISION_PENETRATION_THRESHOLD = 0.005  # 5mm


# =============================================================================
# Helper Functions
# =============================================================================

def get_training_suitability_level(quality_score: float) -> str:
    """
    Get the training suitability level for a given quality score.

    Args:
        quality_score: Overall quality score (0.0 - 1.0)

    Returns:
        One of: "production_training", "fine_tuning", "development_only", "unsuitable"
    """
    if quality_score >= PRODUCTION_TRAINING_THRESHOLD:
        return "production_training"
    elif quality_score >= FINE_TUNING_THRESHOLD:
        return "fine_tuning"
    elif quality_score >= DEVELOPMENT_THRESHOLD:
        return "development_only"
    else:
        return "unsuitable"


def meets_minimum_quality(quality_score: float) -> bool:
    """
    Check if quality score meets minimum threshold for lab delivery.

    Args:
        quality_score: Overall quality score (0.0 - 1.0)

    Returns:
        True if quality score >= MIN_QUALITY_SCORE
    """
    return quality_score >= MIN_QUALITY_SCORE


def get_quality_thresholds() -> dict:
    """
    Get all quality thresholds as a dictionary.

    Useful for logging, configuration, and documentation.

    Returns:
        Dictionary of threshold names to values
    """
    return {
        "min_quality_score": MIN_QUALITY_SCORE,
        "production_training_threshold": PRODUCTION_TRAINING_THRESHOLD,
        "fine_tuning_threshold": FINE_TUNING_THRESHOLD,
        "development_threshold": DEVELOPMENT_THRESHOLD,
        "physics_quality_min": PHYSICS_QUALITY_MIN,
        "trajectory_quality_min": TRAJECTORY_QUALITY_MIN,
        "sensor_data_quality_min": SENSOR_DATA_QUALITY_MIN,
        "max_retries": MAX_RETRIES,
        "stability_threshold": STABILITY_THRESHOLD,
        "object_displacement_threshold": OBJECT_DISPLACEMENT_THRESHOLD,
        "placement_position_tolerance": PLACEMENT_POSITION_TOLERANCE,
        "collision_penetration_threshold": COLLISION_PENETRATION_THRESHOLD,
    }
