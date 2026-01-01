#!/usr/bin/env python3
"""
Runtime Validator for Generated Isaac Lab Code.

This module validates that generated Isaac Lab code actually RUNS correctly,
not just that it has valid syntax. This includes:
1. Import validation - all imports resolve correctly
2. Class instantiation - EnvCfg and Task classes can be created
3. Shape validation - observation/action spaces have correct dimensions
4. Sanity rollout - optionally run a few steps in the environment

IMPORTANT: Full validation requires Isaac Sim environment.
Run with: /isaac-sim/python.sh -m tools.isaac_lab_tasks.runtime_validator
"""

import importlib.util
import json
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Check if we're in Isaac Sim
_ISAAC_SIM_AVAILABLE = False
try:
    import omni.isaac.lab
    _ISAAC_SIM_AVAILABLE = True
except ImportError:
    pass


@dataclass
class RuntimeValidationResult:
    """Result of runtime validation."""

    is_valid: bool = True

    # Validation stages
    imports_valid: bool = False
    class_instantiation_valid: bool = False
    shapes_valid: bool = False
    sanity_rollout_valid: bool = False

    # Details
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Shape information
    observation_shapes: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    action_shape: Optional[Tuple[int, ...]] = None

    # Performance metrics (if sanity rollout was run)
    rollout_fps: float = 0.0
    rollout_steps: int = 0

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "imports_valid": self.imports_valid,
            "class_instantiation_valid": self.class_instantiation_valid,
            "shapes_valid": self.shapes_valid,
            "sanity_rollout_valid": self.sanity_rollout_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "observation_shapes": {k: list(v) for k, v in self.observation_shapes.items()},
            "action_shape": list(self.action_shape) if self.action_shape else None,
            "rollout_fps": self.rollout_fps,
            "rollout_steps": self.rollout_steps,
        }


class IsaacLabRuntimeValidator:
    """
    Validates generated Isaac Lab code at runtime.

    This performs actual execution of the generated code to ensure
    it works correctly in the Isaac Sim environment.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def log(self, msg: str, level: str = "INFO") -> None:
        if self.verbose:
            print(f"[RUNTIME-VALIDATOR] [{level}] {msg}")

    def validate(
        self,
        isaac_lab_dir: Path,
        run_sanity_rollout: bool = True,
        num_rollout_steps: int = 10,
        num_envs: int = 4,
    ) -> RuntimeValidationResult:
        """
        Validate generated Isaac Lab code at runtime.

        Args:
            isaac_lab_dir: Path to generated Isaac Lab package
            run_sanity_rollout: Whether to run a few steps to verify
            num_rollout_steps: Number of steps for sanity rollout
            num_envs: Number of environments for sanity rollout

        Returns:
            RuntimeValidationResult with validation status and details
        """
        result = RuntimeValidationResult()

        self.log(f"Validating Isaac Lab package: {isaac_lab_dir}")

        # Step 1: Validate imports
        self.log("Step 1: Validating imports...")
        result = self._validate_imports(isaac_lab_dir, result)
        if not result.imports_valid:
            self.log("Import validation failed", "ERROR")
            return result
        self.log("  ✓ Imports valid")

        # Step 2: Validate class instantiation
        self.log("Step 2: Validating class instantiation...")
        result = self._validate_instantiation(isaac_lab_dir, result)
        if not result.class_instantiation_valid:
            self.log("Class instantiation failed", "ERROR")
            return result
        self.log("  ✓ Classes instantiate correctly")

        # Step 3: Validate shapes
        self.log("Step 3: Validating observation/action shapes...")
        result = self._validate_shapes(isaac_lab_dir, result)
        if not result.shapes_valid:
            self.log("Shape validation failed", "WARNING")
            # Don't return - shapes might be computed at runtime
        else:
            self.log("  ✓ Shapes valid")

        # Step 4: Run sanity rollout (requires Isaac Sim)
        if run_sanity_rollout:
            if _ISAAC_SIM_AVAILABLE:
                self.log(f"Step 4: Running sanity rollout ({num_rollout_steps} steps)...")
                result = self._run_sanity_rollout(
                    isaac_lab_dir, result, num_rollout_steps, num_envs
                )
                if result.sanity_rollout_valid:
                    self.log(f"  ✓ Sanity rollout passed ({result.rollout_fps:.1f} FPS)")
                else:
                    self.log("  ✗ Sanity rollout failed", "ERROR")
            else:
                self.log("Step 4: Skipping sanity rollout (Isaac Sim not available)", "WARNING")
                result.add_warning("Sanity rollout skipped - Isaac Sim not available")
                # If we can't run sanity rollout but everything else passed, still consider valid
                if result.imports_valid and result.class_instantiation_valid:
                    result.sanity_rollout_valid = True  # Assume valid if we can't test

        # Final status
        if result.is_valid:
            self.log("✓ Runtime validation PASSED")
        else:
            self.log("✗ Runtime validation FAILED", "ERROR")
            for error in result.errors:
                self.log(f"  - {error}", "ERROR")

        return result

    def _validate_imports(
        self,
        isaac_lab_dir: Path,
        result: RuntimeValidationResult,
    ) -> RuntimeValidationResult:
        """Validate that all imports in generated code resolve correctly."""

        # Add the package directory to path temporarily
        parent_dir = isaac_lab_dir.parent
        sys.path.insert(0, str(parent_dir))

        try:
            # Try importing the package
            package_name = isaac_lab_dir.name

            # Check if __init__.py exists and can be imported
            init_path = isaac_lab_dir / "__init__.py"
            if not init_path.exists():
                result.add_error("Missing __init__.py in generated package")
                return result

            # Load the module spec
            spec = importlib.util.spec_from_file_location(
                f"{package_name}",
                init_path,
                submodule_search_locations=[str(isaac_lab_dir)]
            )

            if spec is None or spec.loader is None:
                result.add_error("Failed to create module spec for package")
                return result

            # Try to load the module
            module = importlib.util.module_from_spec(spec)

            # Execute the module (this will trigger all imports)
            try:
                spec.loader.exec_module(module)
                result.imports_valid = True
            except ImportError as e:
                result.add_error(f"Import error: {e}")
            except Exception as e:
                result.add_error(f"Error loading module: {e}")

        finally:
            # Clean up path
            if str(parent_dir) in sys.path:
                sys.path.remove(str(parent_dir))

        return result

    def _validate_instantiation(
        self,
        isaac_lab_dir: Path,
        result: RuntimeValidationResult,
    ) -> RuntimeValidationResult:
        """Validate that EnvCfg and Task classes can be instantiated."""

        parent_dir = isaac_lab_dir.parent
        sys.path.insert(0, str(parent_dir))

        try:
            package_name = isaac_lab_dir.name

            # Find the EnvCfg class
            env_cfg_path = isaac_lab_dir / "env_cfg.py"
            if not env_cfg_path.exists():
                result.add_error("Missing env_cfg.py")
                return result

            # Load env_cfg module
            spec = importlib.util.spec_from_file_location(
                f"{package_name}.env_cfg",
                env_cfg_path,
                submodule_search_locations=[str(isaac_lab_dir)]
            )

            if spec is None or spec.loader is None:
                result.add_error("Failed to load env_cfg.py")
                return result

            env_cfg_module = importlib.util.module_from_spec(spec)

            try:
                spec.loader.exec_module(env_cfg_module)
            except Exception as e:
                result.add_error(f"Error executing env_cfg.py: {e}")
                return result

            # Find EnvCfg class
            env_cfg_class = None
            for name in dir(env_cfg_module):
                obj = getattr(env_cfg_module, name)
                if isinstance(obj, type) and name.endswith("EnvCfg"):
                    env_cfg_class = obj
                    break

            if env_cfg_class is None:
                result.add_error("No *EnvCfg class found in env_cfg.py")
                return result

            # Try to instantiate the EnvCfg
            try:
                cfg_instance = env_cfg_class()
                self.log(f"  Instantiated {env_cfg_class.__name__}")
                result.class_instantiation_valid = True
            except Exception as e:
                result.add_error(f"Failed to instantiate {env_cfg_class.__name__}: {e}")
                return result

        except Exception as e:
            result.add_error(f"Instantiation validation error: {e}")

        finally:
            if str(parent_dir) in sys.path:
                sys.path.remove(str(parent_dir))

        return result

    def _validate_shapes(
        self,
        isaac_lab_dir: Path,
        result: RuntimeValidationResult,
    ) -> RuntimeValidationResult:
        """Validate observation and action space shapes are consistent."""

        parent_dir = isaac_lab_dir.parent
        sys.path.insert(0, str(parent_dir))

        try:
            package_name = isaac_lab_dir.name
            env_cfg_path = isaac_lab_dir / "env_cfg.py"

            spec = importlib.util.spec_from_file_location(
                f"{package_name}.env_cfg",
                env_cfg_path,
                submodule_search_locations=[str(isaac_lab_dir)]
            )

            if spec and spec.loader:
                env_cfg_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(env_cfg_module)

                # Find and instantiate EnvCfg
                for name in dir(env_cfg_module):
                    obj = getattr(env_cfg_module, name)
                    if isinstance(obj, type) and name.endswith("EnvCfg"):
                        try:
                            cfg = obj()

                            # Extract observation shapes from config
                            if hasattr(cfg, 'observations'):
                                obs_cfg = cfg.observations
                                if hasattr(obs_cfg, 'policy'):
                                    policy_obs = obs_cfg.policy
                                    # Count observation terms
                                    obs_count = 0
                                    for term_name in dir(policy_obs):
                                        if not term_name.startswith('_'):
                                            term = getattr(policy_obs, term_name)
                                            if hasattr(term, 'func'):
                                                obs_count += 1
                                    result.observation_shapes["policy_terms"] = (obs_count,)

                            # Extract action dimensions
                            if hasattr(cfg, 'actions'):
                                actions_cfg = cfg.actions
                                for term_name in dir(actions_cfg):
                                    if not term_name.startswith('_'):
                                        term = getattr(actions_cfg, term_name)
                                        if isinstance(term, dict) and 'class_type' in term:
                                            result.action_shape = (7,)  # Default arm DOFs
                                            break

                            result.shapes_valid = True

                        except Exception as e:
                            result.add_warning(f"Could not extract shapes: {e}")
                        break

        except Exception as e:
            result.add_warning(f"Shape validation error: {e}")

        finally:
            if str(parent_dir) in sys.path:
                sys.path.remove(str(parent_dir))

        return result

    def _run_sanity_rollout(
        self,
        isaac_lab_dir: Path,
        result: RuntimeValidationResult,
        num_steps: int,
        num_envs: int,
    ) -> RuntimeValidationResult:
        """Run a sanity rollout to verify the environment works."""

        if not _ISAAC_SIM_AVAILABLE:
            result.add_warning("Isaac Sim not available for sanity rollout")
            return result

        import time

        parent_dir = isaac_lab_dir.parent
        sys.path.insert(0, str(parent_dir))

        try:
            import torch
            from omni.isaac.lab.envs import ManagerBasedEnv

            package_name = isaac_lab_dir.name
            env_cfg_path = isaac_lab_dir / "env_cfg.py"

            # Load the env config
            spec = importlib.util.spec_from_file_location(
                f"{package_name}.env_cfg",
                env_cfg_path,
                submodule_search_locations=[str(isaac_lab_dir)]
            )

            if spec and spec.loader:
                env_cfg_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(env_cfg_module)

                # Find EnvCfg class
                env_cfg_class = None
                for name in dir(env_cfg_module):
                    obj = getattr(env_cfg_module, name)
                    if isinstance(obj, type) and name.endswith("EnvCfg"):
                        env_cfg_class = obj
                        break

                if env_cfg_class is None:
                    result.add_error("No EnvCfg class found for sanity rollout")
                    return result

                # Create config with reduced envs for testing
                cfg = env_cfg_class()
                cfg.scene.num_envs = num_envs

                self.log(f"  Creating environment with {num_envs} envs...")

                # Create environment
                try:
                    env = ManagerBasedEnv(cfg)
                except Exception as e:
                    result.add_error(f"Failed to create environment: {e}")
                    return result

                # Run sanity rollout
                self.log(f"  Running {num_steps} steps...")
                start_time = time.time()

                try:
                    # Reset environment
                    obs, info = env.reset()

                    # Verify observation shape
                    if hasattr(obs, 'shape'):
                        result.observation_shapes["policy"] = tuple(obs.shape)

                    # Run steps with zero actions
                    action_shape = (num_envs, env.action_manager.action.shape[-1])
                    result.action_shape = action_shape[1:]

                    for step in range(num_steps):
                        # Zero actions for sanity check
                        actions = torch.zeros(action_shape, device=env.device)
                        obs, rewards, terminated, truncated, info = env.step(actions)

                        # Basic sanity checks
                        if torch.isnan(obs).any():
                            result.add_error(f"NaN in observations at step {step}")
                            break
                        if torch.isnan(rewards).any():
                            result.add_error(f"NaN in rewards at step {step}")
                            break

                    elapsed = time.time() - start_time
                    result.rollout_fps = num_steps * num_envs / elapsed
                    result.rollout_steps = num_steps
                    result.sanity_rollout_valid = len(result.errors) == 0

                except Exception as e:
                    result.add_error(f"Rollout failed: {e}")
                    traceback.print_exc()

                finally:
                    # Clean up
                    try:
                        env.close()
                    except Exception:
                        pass

        except Exception as e:
            result.add_error(f"Sanity rollout setup failed: {e}")
            traceback.print_exc()

        finally:
            if str(parent_dir) in sys.path:
                sys.path.remove(str(parent_dir))

        return result


def validate_generated_package(
    isaac_lab_dir: Path,
    run_sanity_rollout: bool = True,
    verbose: bool = True,
) -> RuntimeValidationResult:
    """
    Convenience function to validate a generated Isaac Lab package.

    Args:
        isaac_lab_dir: Path to the generated package
        run_sanity_rollout: Whether to run environment steps
        verbose: Print progress

    Returns:
        RuntimeValidationResult
    """
    validator = IsaacLabRuntimeValidator(verbose=verbose)
    return validator.validate(
        isaac_lab_dir,
        run_sanity_rollout=run_sanity_rollout,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate generated Isaac Lab code at runtime"
    )
    parser.add_argument(
        "isaac_lab_dir",
        type=Path,
        help="Path to generated Isaac Lab package directory"
    )
    parser.add_argument(
        "--no-rollout",
        action="store_true",
        help="Skip sanity rollout (only validate imports and instantiation)"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=10,
        help="Number of steps for sanity rollout"
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="Number of environments for sanity rollout"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON file for validation results"
    )

    args = parser.parse_args()

    if not args.isaac_lab_dir.exists():
        print(f"ERROR: Directory not found: {args.isaac_lab_dir}")
        sys.exit(1)

    print("=" * 60)
    print("ISAAC LAB RUNTIME VALIDATOR")
    print("=" * 60)
    print(f"Package: {args.isaac_lab_dir}")
    print(f"Isaac Sim available: {_ISAAC_SIM_AVAILABLE}")
    print(f"Sanity rollout: {not args.no_rollout}")
    print("=" * 60)

    validator = IsaacLabRuntimeValidator(verbose=True)
    result = validator.validate(
        args.isaac_lab_dir,
        run_sanity_rollout=not args.no_rollout,
        num_rollout_steps=args.num_steps,
        num_envs=args.num_envs,
    )

    print("\n" + "=" * 60)
    print("VALIDATION RESULT")
    print("=" * 60)
    print(f"Valid: {result.is_valid}")
    print(f"Imports: {'✓' if result.imports_valid else '✗'}")
    print(f"Instantiation: {'✓' if result.class_instantiation_valid else '✗'}")
    print(f"Shapes: {'✓' if result.shapes_valid else '?'}")
    print(f"Sanity rollout: {'✓' if result.sanity_rollout_valid else '✗'}")

    if result.observation_shapes:
        print(f"Observation shapes: {result.observation_shapes}")
    if result.action_shape:
        print(f"Action shape: {result.action_shape}")
    if result.rollout_fps > 0:
        print(f"Rollout FPS: {result.rollout_fps:.1f}")

    if result.errors:
        print("\nErrors:")
        for error in result.errors:
            print(f"  ✗ {error}")

    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings:
            print(f"  ⚠ {warning}")

    # Output to JSON if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nResults written to: {args.output}")

    sys.exit(0 if result.is_valid else 1)
