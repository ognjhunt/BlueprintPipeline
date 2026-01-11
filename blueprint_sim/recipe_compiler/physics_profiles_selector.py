"""Physics Profile Selector - Implements physics profile selection rules.

GAP-PHYSICS-011 FIX: This module applies physics profiles based on task characteristics
to optimize simulation fidelity and performance for specific use cases.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PhysicsProfileSelector:
    """Selects and applies physics profiles based on task characteristics."""

    def __init__(self, profiles_path: Optional[Path] = None) -> None:
        """Initialize profile selector.

        Args:
            profiles_path: Optional path to physics_profiles.json. Defaults to
                          policy_configs/physics_profiles.json relative to repo root.
        """
        if profiles_path is None:
            repo_root = Path(__file__).resolve().parents[2]
            profiles_path = repo_root / "policy_configs" / "physics_profiles.json"

        self.profiles_path = profiles_path
        self.profiles: Dict[str, Any] = {}
        self.rules: List[Dict[str, Any]] = []
        self.meta_randomization: Dict[str, Any] = {}

        self._load_profiles()

    def _load_profiles(self) -> None:
        """Load physics profiles from JSON file."""
        if not self.profiles_path.exists():
            logger.warning(
                f"Physics profiles file not found at {self.profiles_path}, "
                "using hardcoded defaults"
            )
            self.profiles = self._get_default_profiles()
            self.rules = self._get_default_rules()
            return

        try:
            with open(self.profiles_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.profiles = data.get("profiles", {})
            profile_selection = data.get("profile_selection_rules", {})
            self.rules = profile_selection.get("rules", [])
            self.meta_randomization = data.get("solver_meta_randomization", {})

            logger.info(f"Loaded {len(self.profiles)} physics profiles from {self.profiles_path}")
        except Exception as e:
            logger.error(f"Failed to load physics profiles: {e}")
            self.profiles = self._get_default_profiles()
            self.rules = self._get_default_rules()

    def select_profile(self, task_name: str) -> str:
        """Select a profile based on task name.

        Args:
            task_name: Name or description of the task

        Returns:
            Profile name (e.g., "manipulation_standard", "navigation")
        """
        task_lower = task_name.lower() if task_name else ""

        # Evaluate rules in order
        for rule in self.rules:
            condition = rule.get("condition", "").lower()

            # Skip description and default
            if condition.startswith("_") or condition == "default":
                continue

            # Evaluate condition
            if self._matches_condition(condition, task_lower):
                profile_name = rule.get("profile")
                if profile_name and profile_name in self.profiles:
                    logger.info(f"Selected profile '{profile_name}' for task '{task_name}' (rule: '{condition}')")
                    return profile_name

        # Default profile
        for rule in self.rules:
            if rule.get("condition", "").lower() == "default":
                default_profile = rule.get("profile", "manipulation_standard")
                if default_profile in self.profiles:
                    logger.info(f"Using default profile '{default_profile}' for task '{task_name}'")
                    return default_profile

        # Fallback
        logger.warning(f"No suitable profile found for task '{task_name}', using 'manipulation_standard'")
        return "manipulation_standard"

    def get_profile_config(self, profile_name: str) -> Dict[str, Any]:
        """Get the complete configuration for a profile.

        Args:
            profile_name: Name of the profile

        Returns:
            Profile configuration dict with simulation, contact, and randomization settings
        """
        if profile_name not in self.profiles:
            logger.warning(f"Profile '{profile_name}' not found, using 'manipulation_standard'")
            profile_name = "manipulation_standard"

        profile = self.profiles.get(profile_name, {})
        return {
            "name": profile_name,
            "description": profile.get("description", ""),
            "simulation": profile.get("simulation", {}),
            "contact": profile.get("contact", {}),
            "randomization": profile.get("randomization", {}),
        }

    def apply_profile_to_physics(
        self, physics_config: Dict[str, Any], profile_name: str
    ) -> Dict[str, Any]:
        """Apply profile settings to a physics configuration.

        Args:
            physics_config: Current physics configuration dict
            profile_name: Name of profile to apply

        Returns:
            Updated physics configuration with profile settings applied
        """
        if profile_name not in self.profiles:
            logger.warning(f"Profile '{profile_name}' not found, returning unchanged config")
            return physics_config

        profile = self.profiles.get(profile_name, {})
        contact_settings = profile.get("contact", {})

        # Apply contact settings that override physics properties
        merged = dict(physics_config)

        # Map profile contact settings to physics properties
        if "contact_offset" in contact_settings:
            merged["contact_offset_m"] = contact_settings["contact_offset"]

        if "rest_offset" in contact_settings:
            merged["rest_offset_m"] = contact_settings["rest_offset"]

        # Store profile reference for Isaac Lab task generation
        merged["physics_profile"] = profile_name
        merged["physics_profile_config"] = self.get_profile_config(profile_name)

        return merged

    def get_scenario_suggestions(self, task_name: str) -> Dict[str, Any]:
        """Get recommended settings for a scenario/task.

        Args:
            task_name: Name of the task

        Returns:
            Dict with profile recommendations and settings
        """
        profile_name = self.select_profile(task_name)
        profile_config = self.get_profile_config(profile_name)

        return {
            "task_name": task_name,
            "recommended_profile": profile_name,
            "profile_config": profile_config,
            "meta_randomization_enabled": self.meta_randomization.get("enabled", False),
            "use_cases": profile_config.get("simulation", {}).get("use_cases", []),
        }

    def _matches_condition(self, condition: str, task_lower: str) -> bool:
        """Check if a condition matches a task name.

        Args:
            condition: Condition string (e.g., "task contains 'insertion' or 'assembly'")
            task_lower: Lowercase task name

        Returns:
            True if condition matches
        """
        # Parse condition: "task contains 'X' or 'Y' or contains 'Z'"
        # This is a simplified parser - enhance as needed

        if "contains" not in condition:
            return False

        # Extract keywords using regex
        # Pattern: contains 'keyword' or contains 'keyword'
        pattern = r"contains\s+['\"]([^'\"]+)['\"]"
        keywords = re.findall(pattern, condition)

        if not keywords:
            return False

        # Check if any keyword appears in the task name
        for keyword in keywords:
            if keyword.lower() in task_lower:
                return True

        return False

    @staticmethod
    def _get_default_profiles() -> Dict[str, Any]:
        """Return hardcoded default profiles when file is unavailable."""
        return {
            "manipulation_standard": {
                "description": "Balanced settings for general manipulation tasks",
                "use_cases": ["pick_place", "articulated_access"],
                "simulation": {
                    "dt": 0.008,
                    "substeps": 2,
                    "solver_iterations": 16,
                    "solver_type": "TGS",
                },
                "contact": {
                    "contact_offset": 0.005,
                    "rest_offset": 0.001,
                },
            },
            "navigation": {
                "description": "Fast settings for navigation tasks",
                "use_cases": ["navigation", "locomotion"],
                "simulation": {
                    "dt": 0.016,
                    "substeps": 1,
                    "solver_iterations": 8,
                    "solver_type": "PGS",
                },
                "contact": {
                    "contact_offset": 0.01,
                    "rest_offset": 0.002,
                },
            },
        }

    @staticmethod
    def _get_default_rules() -> List[Dict[str, Any]]:
        """Return hardcoded default selection rules when file is unavailable."""
        return [
            {
                "condition": "task contains 'insertion' or 'assembly'",
                "profile": "manipulation_contact_rich",
            },
            {
                "condition": "task contains 'navigation' or 'locomotion'",
                "profile": "navigation",
            },
            {"condition": "default", "profile": "manipulation_standard"},
        ]


def create_profile_selector(profiles_path: Optional[Path] = None) -> PhysicsProfileSelector:
    """Factory function to create a profile selector.

    Args:
        profiles_path: Optional path to physics_profiles.json

    Returns:
        Initialized PhysicsProfileSelector
    """
    return PhysicsProfileSelector(profiles_path)


__all__ = ["PhysicsProfileSelector", "create_profile_selector"]
