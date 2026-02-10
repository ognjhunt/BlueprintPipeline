#!/usr/bin/env python3
"""
Master Replicator Script for Scene: ChIJBc5E5wTjrIkRKrrWR_meHbc
Environment Type: generic

This script provides a unified interface to run any of the available
policy-specific Replicator configurations for this scene.

Usage in Isaac Sim Script Editor:
    from replicator_master import ReplicatorManager

    manager = ReplicatorManager()
    manager.list_policies()
    manager.run_policy("dish_loading", num_frames=500)
"""

import importlib.util
import sys
from pathlib import Path
from typing import Optional, List

# Available policies for this scene
AVAILABLE_POLICIES = {
    "dexterous_pick_place": "Precision Desk Manipulation",
    "general_manipulation": "Room Tidying and Sorting"
}

SCENE_ID = "ChIJBc5E5wTjrIkRKrrWR_meHbc"
ENVIRONMENT_TYPE = "generic"


class ReplicatorManager:
    """Manager for running Replicator policies on this scene."""

    def __init__(self, scripts_dir: Optional[str] = None):
        """Initialize the manager."""
        if scripts_dir is None:
            # Assume scripts are in the same directory
            self.scripts_dir = Path(__file__).parent / "policies"
        else:
            self.scripts_dir = Path(scripts_dir)

    def list_policies(self) -> List[str]:
        """List all available policies."""
        print(f"\nAvailable policies for {SCENE_ID} ({ENVIRONMENT_TYPE}):")
        print("-" * 50)
        for policy_id, policy_name in AVAILABLE_POLICIES.items():
            print(f"  {policy_id}: {policy_name}")
        print("-" * 50)
        return list(AVAILABLE_POLICIES.keys())

    def load_policy_module(self, policy_id: str):
        """Dynamically load a policy script module."""
        script_path = self.scripts_dir / f"{policy_id}.py"

        if not script_path.exists():
            raise FileNotFoundError(f"Policy script not found: {script_path}")

        spec = importlib.util.spec_from_file_location(policy_id, script_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[policy_id] = module
        spec.loader.exec_module(module)

        return module

    def run_policy(self, policy_id: str, num_frames: int = 100, **kwargs):
        """Run a specific policy."""
        if policy_id not in AVAILABLE_POLICIES:
            print(f"Error: Unknown policy '{policy_id}'")
            self.list_policies()
            return

        print(f"\n[REPLICATOR] Loading policy: {AVAILABLE_POLICIES[policy_id]}")

        module = self.load_policy_module(policy_id)

        if hasattr(module, 'run_replicator'):
            module.run_replicator(num_frames=num_frames)
        else:
            print(f"Error: Policy module does not have run_replicator function")

    def run_all_policies(self, num_frames_each: int = 100):
        """Run all available policies sequentially."""
        for policy_id in AVAILABLE_POLICIES.keys():
            print(f"\n============================================================")
            print(f"Running policy: {policy_id}")
            print(f"============================================================")
            self.run_policy(policy_id, num_frames=num_frames_each)


# Quick access functions
def list_policies():
    """List all available policies."""
    manager = ReplicatorManager()
    return manager.list_policies()


def run_policy(policy_id: str, num_frames: int = 100):
    """Run a specific policy."""
    manager = ReplicatorManager()
    manager.run_policy(policy_id, num_frames=num_frames)


def run_all(num_frames_each: int = 100):
    """Run all policies."""
    manager = ReplicatorManager()
    manager.run_all_policies(num_frames_each)


if __name__ == "__main__":
    print("[REPLICATOR] Master script loaded")
    list_policies()
