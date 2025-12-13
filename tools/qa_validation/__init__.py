"""End-to-End QA Validation Gate for BlueprintPipeline.

This module provides comprehensive validation of generated scenes to ensure
they meet the "Definition of Done" requirements:

1. USD Validation:
   - scene.usda loads without errors
   - No missing textures or references
   - Proper coordinate system and scale

2. Physics Validation:
   - PhysX simulation stable for N steps
   - Objects don't explode or fall through floor
   - Articulated joints controllable

3. Replicator Validation:
   - Placement regions are valid
   - Policy scripts execute without errors
   - Frames can be generated

4. Isaac Lab Validation:
   - Task imports successfully
   - Reset/step work correctly
   - Observations/actions have correct shapes

Usage:
    from tools.qa_validation import SceneValidator, ValidationReport

    validator = SceneValidator(scene_dir)
    report = validator.run_full_validation()

    if report.passed:
        print("Scene validated successfully!")
    else:
        for issue in report.issues:
            print(f"FAIL: {issue}")
"""

from .validator import (
    SceneValidator,
    ValidationReport,
    ValidationResult,
    ValidationLevel,
    run_qa_validation,
)

__all__ = [
    "SceneValidator",
    "ValidationReport",
    "ValidationResult",
    "ValidationLevel",
    "run_qa_validation",
]
