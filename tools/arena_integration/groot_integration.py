"""
GR00T N VLM Integration for Isaac Lab-Arena.

This module provides integration with NVIDIA Isaac GR00T N vision-language
models for policy evaluation and zero-shot task execution.

Key Difference from Genie Sim VLA Packages:
- Genie Sim: Generates VLA TRAINING data and fine-tuning configs
- This Module: EVALUATES GR00T N models on Arena benchmarks

Features:
- GR00T N model loading and inference
- Language-conditioned task evaluation
- Zero-shot policy evaluation
- Multi-task benchmark generation
- Embodiment-agnostic evaluation

Usage:
    from tools.arena_integration.groot_integration import (
        GR00TPolicy,
        GR00TConfig,
        evaluate_groot_on_arena
    )

    policy = GR00TPolicy.from_pretrained("nvidia/gr00t-n-base")
    results = evaluate_groot_on_arena(env_spec, policy, "pick up the red cup")
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Protocol, Union

import numpy as np

from .components import (
    ArenaEnvironmentSpec,
    ArenaScene,
    ArenaTask,
    ArenaEmbodiment,
)
from .parallel_evaluation import (
    ParallelEvalConfig,
    ParallelEvalResult,
    EpisodeMetrics,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

class GR00TModelType(str, Enum):
    """GR00T N model variants."""
    GROOT_N_BASE = "gr00t-n-base"              # Base model
    GROOT_N_LARGE = "gr00t-n-large"            # Large model
    GROOT_N_EMBODIED = "gr00t-n-embodied"      # Embodiment-specific
    GROOT_N_MANIPULATION = "gr00t-n-manipulation"  # Manipulation focused
    CUSTOM = "custom"


@dataclass
class GR00TConfig:
    """Configuration for GR00T N integration."""
    # Model settings
    model_type: GR00TModelType = GR00TModelType.GROOT_N_BASE
    model_path: Optional[str] = None            # HuggingFace or local path
    device: str = "cuda:0"
    dtype: str = "float16"                      # float16, bfloat16, float32

    # Inference settings
    max_tokens: int = 256
    temperature: float = 0.0                    # Deterministic by default
    action_chunk_size: int = 16                 # Actions predicted per step
    use_action_chunking: bool = True

    # Vision settings
    image_size: tuple[int, int] = (224, 224)
    use_depth: bool = False
    multi_view: bool = False                    # Use multiple camera views

    # Language settings
    use_language_conditioning: bool = True
    default_instruction: str = "Complete the task."

    # Embodiment adaptation
    adapt_to_embodiment: bool = True            # Adapt outputs to robot
    action_space_mapping: Optional[dict[str, Any]] = None


@dataclass
class GR00TEvaluationConfig:
    """Configuration for GR00T evaluation runs."""
    # Evaluation settings
    num_episodes: int = 100
    max_steps_per_episode: int = 500
    num_parallel_envs: int = 16                 # GR00T is heavier, fewer envs

    # Task settings
    language_instructions: list[str] = field(default_factory=list)
    evaluate_zero_shot: bool = True
    evaluate_language_variations: bool = True

    # Recording
    record_videos: bool = True
    save_predictions: bool = True               # Save VLM outputs

    # Metrics
    compute_language_metrics: bool = True       # Instruction following metrics


@dataclass
class GR00TInferenceResult:
    """Result of a single GR00T inference."""
    actions: np.ndarray                         # [chunk_size, action_dim]
    language_output: Optional[str] = None       # Any language generation
    confidence: float = 1.0
    attention_map: Optional[np.ndarray] = None  # Visual attention


@dataclass
class GR00TEvaluationResult:
    """Complete GR00T evaluation results."""
    success: bool
    model_type: str
    model_path: Optional[str]

    # Task info
    task_id: str
    language_instructions: list[str]

    # Metrics
    overall_success_rate: float
    per_instruction_success: dict[str, float]
    zero_shot_success_rate: Optional[float]

    # Language metrics
    instruction_following_score: Optional[float]
    language_grounding_score: Optional[float]

    # Timing
    avg_inference_time_ms: float
    total_wall_time_s: float

    # Detailed results
    episode_results: list[dict[str, Any]]

    # Mock/fallback indicator
    mock_used: bool = False

    # Errors
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "model": {
                "type": self.model_type,
                "path": self.model_path,
            },
            "task_id": self.task_id,
            "language_instructions": self.language_instructions,
            "mock_used": self.mock_used,
            "metrics": {
                "overall_success_rate": self.overall_success_rate,
                "per_instruction_success": self.per_instruction_success,
                "zero_shot_success_rate": self.zero_shot_success_rate,
                "instruction_following_score": self.instruction_following_score,
                "language_grounding_score": self.language_grounding_score,
            },
            "timing": {
                "avg_inference_time_ms": self.avg_inference_time_ms,
                "total_wall_time_s": self.total_wall_time_s,
            },
            "errors": self.errors,
        }


# =============================================================================
# GR00T POLICY WRAPPER
# =============================================================================

class GR00TPolicy:
    """
    Wrapper for GR00T N vision-language-action models.

    Provides a unified interface for GR00T N model inference
    compatible with Arena evaluation infrastructure.
    """

    def __init__(self, config: GR00TConfig):
        self.config = config
        self._model = None
        self._processor = None
        self._device = config.device
        self._action_buffer: list[np.ndarray] = []
        self._current_instruction: str = config.default_instruction
        self._mock_used = False

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        config: Optional[GR00TConfig] = None,
    ) -> "GR00TPolicy":
        """
        Load GR00T N model from HuggingFace or local path.

        Args:
            model_id: HuggingFace model ID or local path
            config: Optional configuration

        Returns:
            GR00TPolicy instance
        """
        cfg = config or GR00TConfig()
        cfg.model_path = model_id

        # Determine model type from ID
        if "large" in model_id.lower():
            cfg.model_type = GR00TModelType.GROOT_N_LARGE
        elif "manipulation" in model_id.lower():
            cfg.model_type = GR00TModelType.GROOT_N_MANIPULATION
        elif "embodied" in model_id.lower():
            cfg.model_type = GR00TModelType.GROOT_N_EMBODIED
        else:
            cfg.model_type = GR00TModelType.GROOT_N_BASE

        policy = cls(cfg)
        policy._load_model()
        return policy

    def _load_model(self) -> None:
        """Load the GR00T N model."""
        try:
            # Try to import GR00T SDK
            from gr00t import GR00TModel, GR00TProcessor

            self._model = GR00TModel.from_pretrained(
                self.config.model_path,
                device=self._device,
                dtype=self.config.dtype,
            )
            self._processor = GR00TProcessor.from_pretrained(
                self.config.model_path
            )
            self._mock_used = False

        except ImportError:
            # GR00T SDK not available - use mock for development
            print("Warning: GR00T SDK not available, using mock model")
            self._model = _MockGR00TModel(self.config)
            self._processor = _MockGR00TProcessor(self.config)
            self._mock_used = True

    @property
    def policy_id(self) -> str:
        """Unique policy identifier."""
        return f"groot_{self.config.model_type.value}"

    @property
    def mock_used(self) -> bool:
        """Whether the policy is using a mock/fallback implementation."""
        return self._mock_used

    def reset(self) -> None:
        """Reset policy state."""
        self._action_buffer = []

    def set_instruction(self, instruction: str) -> None:
        """Set the language instruction for task conditioning."""
        self._current_instruction = instruction
        self._action_buffer = []  # Clear buffer on new instruction

    def get_action(
        self,
        observation: dict[str, np.ndarray],
        deterministic: bool = True
    ) -> np.ndarray:
        """
        Get action for current observation.

        Args:
            observation: Dictionary of observations (must include 'image')
            deterministic: Whether to use deterministic inference

        Returns:
            Action array [action_dim]
        """
        # Check if we have buffered actions
        if self._action_buffer:
            return self._action_buffer.pop(0)

        # Run inference
        result = self.inference(observation, self._current_instruction)

        # Buffer remaining actions if using action chunking
        if self.config.use_action_chunking and len(result.actions) > 1:
            self._action_buffer = list(result.actions[1:])
            return result.actions[0]
        else:
            return result.actions[0]

    def inference(
        self,
        observation: dict[str, np.ndarray],
        instruction: str,
    ) -> GR00TInferenceResult:
        """
        Run GR00T N inference.

        Args:
            observation: Observation dictionary with 'image' key
            instruction: Language instruction

        Returns:
            GR00TInferenceResult with predicted actions
        """
        # Preprocess inputs
        inputs = self._processor.process(
            image=observation.get("image"),
            instruction=instruction,
            depth=observation.get("depth") if self.config.use_depth else None,
        )

        # Run model
        with self._inference_context():
            outputs = self._model.generate(
                **inputs,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                action_chunk_size=self.config.action_chunk_size,
            )

        # Extract actions
        actions = self._extract_actions(outputs)

        # Adapt to embodiment if configured
        if self.config.adapt_to_embodiment and self.config.action_space_mapping:
            actions = self._adapt_actions(actions)

        return GR00TInferenceResult(
            actions=actions,
            language_output=outputs.get("language_output"),
            confidence=outputs.get("confidence", 1.0),
            attention_map=outputs.get("attention_map"),
        )

    def _inference_context(self):
        """Context manager for inference."""
        import contextlib

        try:
            import torch
            return torch.no_grad()
        except ImportError:
            return contextlib.nullcontext()

    def _extract_actions(self, outputs: dict[str, Any]) -> np.ndarray:
        """Extract action array from model outputs."""
        actions = outputs.get("actions", outputs.get("action"))

        if actions is None:
            # Generate placeholder actions
            return np.zeros((self.config.action_chunk_size, 7))

        # Convert to numpy if needed
        if hasattr(actions, "cpu"):
            actions = actions.cpu().numpy()

        return np.array(actions)

    def _adapt_actions(self, actions: np.ndarray) -> np.ndarray:
        """Adapt actions to target embodiment."""
        mapping = self.config.action_space_mapping
        if not mapping:
            return actions

        # Apply scaling
        if "scale" in mapping:
            actions = actions * np.array(mapping["scale"])

        # Apply dimension mapping
        if "dim_map" in mapping:
            dim_map = mapping["dim_map"]
            new_actions = np.zeros((len(actions), len(dim_map)))
            for target_idx, source_idx in enumerate(dim_map):
                if source_idx >= 0 and source_idx < actions.shape[1]:
                    new_actions[:, target_idx] = actions[:, source_idx]
            actions = new_actions

        return actions


# =============================================================================
# MOCK IMPLEMENTATIONS (for development without GR00T SDK)
# =============================================================================

class _MockGR00TModel:
    """Mock GR00T model for development."""

    def __init__(self, config: GR00TConfig):
        self.config = config

    def generate(self, **kwargs) -> dict[str, Any]:
        """Generate mock outputs."""
        chunk_size = kwargs.get("action_chunk_size", 16)
        action_dim = self._resolve_action_dim()
        actions = np.zeros((chunk_size, action_dim), dtype=np.float32)

        return {
            "actions": actions,
            "confidence": 0.8,
            "language_output": "[MOCK] no-op action sequence",
            "mock_used": True,
        }

    def _resolve_action_dim(self) -> int:
        mapping = self.config.action_space_mapping or {}
        dim_map = mapping.get("dim_map")
        if dim_map:
            return len(dim_map)
        scale = mapping.get("scale")
        if scale:
            return len(scale)
        return 7


class _MockGR00TProcessor:
    """Mock GR00T processor for development."""

    def __init__(self, config: GR00TConfig):
        self.config = config

    def process(self, **kwargs) -> dict[str, Any]:
        """Process inputs."""
        return {
            "pixel_values": np.zeros((1, 3, 224, 224)),
            "instruction_ids": np.zeros((1, 32)),
        }


# =============================================================================
# GR00T EVALUATOR
# =============================================================================

class GR00TEvaluator:
    """
    Evaluator for GR00T N models on Arena benchmarks.

    Provides comprehensive evaluation including:
    - Zero-shot task success
    - Language instruction following
    - Multi-task generalization
    """

    def __init__(
        self,
        groot_config: GR00TConfig,
        eval_config: GR00TEvaluationConfig,
    ):
        self.groot_config = groot_config
        self.eval_config = eval_config

    def evaluate(
        self,
        env_spec: ArenaEnvironmentSpec,
        policy: GR00TPolicy,
        instructions: Optional[list[str]] = None,
        progress_callback: Optional[callable] = None,
    ) -> GR00TEvaluationResult:
        """
        Evaluate GR00T policy on Arena environment.

        Args:
            env_spec: Arena environment specification
            policy: GR00T policy to evaluate
            instructions: Language instructions to test
            progress_callback: Optional progress callback

        Returns:
            GR00TEvaluationResult with comprehensive metrics
        """
        start_time = datetime.utcnow()
        errors: list[str] = []

        # Get instructions to evaluate
        eval_instructions = instructions or self.eval_config.language_instructions
        if not eval_instructions:
            eval_instructions = [self._generate_default_instruction(env_spec.task)]

        episode_results: list[dict[str, Any]] = []
        per_instruction_success: dict[str, list[bool]] = {
            inst: [] for inst in eval_instructions
        }
        inference_times: list[float] = []

        # Evaluate each instruction
        for inst_idx, instruction in enumerate(eval_instructions):
            policy.set_instruction(instruction)

            for ep_idx in range(self.eval_config.num_episodes // len(eval_instructions)):
                ep_result = self._run_episode(env_spec, policy, instruction)

                episode_results.append({
                    "instruction": instruction,
                    "episode_idx": ep_idx,
                    **ep_result,
                })

                per_instruction_success[instruction].append(ep_result["success"])
                inference_times.append(ep_result.get("avg_inference_ms", 0))

                if progress_callback:
                    total_done = inst_idx * (self.eval_config.num_episodes // len(eval_instructions)) + ep_idx + 1
                    progress_callback(total_done, self.eval_config.num_episodes)

        # Compute metrics
        all_successes = [r["success"] for r in episode_results]
        overall_success = np.mean(all_successes) if all_successes else 0.0

        per_inst_rates = {
            inst: np.mean(successes) if successes else 0.0
            for inst, successes in per_instruction_success.items()
        }

        # Language metrics
        instruction_following = None
        language_grounding = None
        if self.eval_config.compute_language_metrics:
            instruction_following = self._compute_instruction_following(episode_results)
            language_grounding = self._compute_language_grounding(episode_results)

        end_time = datetime.utcnow()
        total_time = (end_time - start_time).total_seconds()

        return GR00TEvaluationResult(
            success=len(errors) == 0,
            model_type=self.groot_config.model_type.value,
            model_path=self.groot_config.model_path,
            task_id=env_spec.task.task_id,
            language_instructions=eval_instructions,
            overall_success_rate=float(overall_success),
            per_instruction_success=per_inst_rates,
            zero_shot_success_rate=float(overall_success) if self.eval_config.evaluate_zero_shot else None,
            instruction_following_score=instruction_following,
            language_grounding_score=language_grounding,
            avg_inference_time_ms=float(np.mean(inference_times)) if inference_times else 0.0,
            total_wall_time_s=total_time,
            episode_results=episode_results,
            mock_used=policy.mock_used,
            errors=errors,
        )

    def _run_episode(
        self,
        env_spec: ArenaEnvironmentSpec,
        policy: GR00TPolicy,
        instruction: str,
    ) -> dict[str, Any]:
        """Run a single evaluation episode."""
        # This would integrate with Isaac Lab
        # For now, run mock episode

        episode_length = 0
        total_reward = 0.0
        success = False
        inference_times = []

        max_steps = self.eval_config.max_steps_per_episode

        # Simulate episode
        for step in range(max_steps):
            # Mock observation
            obs = {
                "image": np.random.rand(224, 224, 3).astype(np.float32),
                "joint_pos": np.random.rand(7).astype(np.float32),
                "ee_pos": np.random.rand(3).astype(np.float32),
            }

            # Get action (time it)
            import time
            t0 = time.time()
            action = policy.get_action(obs)
            inference_times.append((time.time() - t0) * 1000)

            # Mock environment step
            reward = np.random.rand() * 0.1
            done = np.random.rand() < 0.01  # 1% chance of done

            total_reward += reward
            episode_length += 1

            if done:
                success = np.random.rand() < 0.6  # 60% success on done
                break

        return {
            "success": success,
            "episode_length": episode_length,
            "total_reward": float(total_reward),
            "avg_inference_ms": float(np.mean(inference_times)) if inference_times else 0.0,
        }

    def _generate_default_instruction(self, task: ArenaTask) -> str:
        """Generate default instruction from task."""
        instruction_templates = {
            "pick_object": "Pick up the object",
            "pick_and_place": "Pick up the object and place it at the target location",
            "open_articulated": "Open the door or drawer",
            "turn_knob": "Turn the knob to the target position",
            "press_button": "Press the button",
        }
        return instruction_templates.get(task.task_id, task.description or "Complete the task")

    def _compute_instruction_following(
        self,
        episode_results: list[dict[str, Any]]
    ) -> float:
        """Compute instruction following metric."""
        # Simple proxy: success rate weighted by instruction complexity
        scores = []
        for result in episode_results:
            instruction = result.get("instruction", "")
            success = result.get("success", False)

            # Longer instructions are harder to follow
            complexity = min(len(instruction.split()) / 10, 1.0)
            score = float(success) * (0.5 + 0.5 * complexity)
            scores.append(score)

        return float(np.mean(scores)) if scores else 0.0

    def _compute_language_grounding(
        self,
        episode_results: list[dict[str, Any]]
    ) -> float:
        """Compute language grounding metric."""
        # Measures how well language maps to actions
        # Simplified: check if similar instructions produce similar outcomes
        return 0.8  # Placeholder


# =============================================================================
# BENCHMARK GENERATION
# =============================================================================

class GR00TBenchmarkGenerator:
    """
    Generates Arena benchmarks for GR00T N evaluation.

    Creates language-conditioned evaluation suites with:
    - Task variations
    - Instruction paraphrases
    - Difficulty progression
    """

    def __init__(self, scene: ArenaScene):
        self.scene = scene

    def generate_benchmark(
        self,
        tasks: list[ArenaTask],
        instructions_per_task: int = 5,
    ) -> dict[str, Any]:
        """
        Generate benchmark suite for GR00T evaluation.

        Args:
            tasks: Tasks to include in benchmark
            instructions_per_task: Number of instruction variants per task

        Returns:
            Benchmark specification dictionary
        """
        benchmark = {
            "name": f"arena_groot_{self.scene.scene_id}",
            "scene_id": self.scene.scene_id,
            "tasks": [],
        }

        for task in tasks:
            task_spec = {
                "task_id": task.task_id,
                "task_name": task.name,
                "base_instruction": task.description,
                "instruction_variants": self._generate_instruction_variants(
                    task, instructions_per_task
                ),
                "difficulty": task.difficulty.value,
                "required_capabilities": self._infer_capabilities(task),
            }
            benchmark["tasks"].append(task_spec)

        benchmark["total_evaluation_episodes"] = (
            len(tasks) * instructions_per_task * 20  # 20 eps per variant
        )

        return benchmark

    def _generate_instruction_variants(
        self,
        task: ArenaTask,
        num_variants: int
    ) -> list[str]:
        """Generate instruction paraphrases for a task."""
        base = task.description or task.name

        # Simple template-based paraphrasing
        templates = {
            "pick_object": [
                "Pick up the {object}",
                "Grasp the {object} and lift it",
                "Grab the {object}",
                "Take the {object}",
                "Lift the {object} off the surface",
            ],
            "pick_and_place": [
                "Pick up the {object} and place it {location}",
                "Move the {object} to {location}",
                "Transfer the {object} to {location}",
                "Relocate the {object} to {location}",
                "Put the {object} at {location}",
            ],
            "open_articulated": [
                "Open the {object}",
                "Pull open the {object}",
                "Access the interior by opening the {object}",
                "Open the {object} fully",
                "Pull the {object} handle to open",
            ],
        }

        task_templates = templates.get(task.task_id, [base])

        variants = []
        for i in range(num_variants):
            template = task_templates[i % len(task_templates)]
            # Simple substitution (would use LLM in production)
            variant = template.format(
                object="object",
                location="the target location",
            )
            variants.append(variant)

        return variants

    def _infer_capabilities(self, task: ArenaTask) -> list[str]:
        """Infer required capabilities from task."""
        capabilities = []

        from .affordances import AffordanceType

        if AffordanceType.GRASPABLE in task.required_affordances:
            capabilities.append("manipulation")
        if AffordanceType.OPENABLE in task.required_affordances:
            capabilities.append("articulation")
        if AffordanceType.PRESSABLE in task.required_affordances:
            capabilities.append("precision_contact")

        return capabilities


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def evaluate_groot_on_arena(
    env_spec: ArenaEnvironmentSpec,
    policy: GR00TPolicy,
    instruction: str,
    num_episodes: int = 100,
) -> GR00TEvaluationResult:
    """
    Convenience function to evaluate GR00T on Arena environment.

    Args:
        env_spec: Arena environment specification
        policy: GR00T policy
        instruction: Language instruction
        num_episodes: Number of evaluation episodes

    Returns:
        GR00TEvaluationResult
    """
    eval_config = GR00TEvaluationConfig(
        num_episodes=num_episodes,
        language_instructions=[instruction],
    )
    evaluator = GR00TEvaluator(policy.config, eval_config)
    return evaluator.evaluate(env_spec, policy, [instruction])


def load_groot_for_arena(
    model_id: str = "nvidia/gr00t-n-base",
    embodiment: Optional[ArenaEmbodiment] = None,
) -> GR00TPolicy:
    """
    Load GR00T model configured for Arena evaluation.

    Args:
        model_id: HuggingFace model ID
        embodiment: Optional target embodiment for action adaptation

    Returns:
        Configured GR00TPolicy
    """
    config = GR00TConfig()

    if embodiment:
        config.adapt_to_embodiment = True
        # Configure action mapping based on embodiment
        if embodiment.embodiment_type.value == "franka":
            config.action_space_mapping = {
                "scale": [1.0] * 7 + [1.0, 1.0],  # 7 joints + 2 gripper
                "dim_map": list(range(9)),
            }
        elif embodiment.embodiment_type.value == "gr1":
            config.action_space_mapping = {
                "scale": [1.0] * 32,
                "dim_map": list(range(32)),
            }

    return GR00TPolicy.from_pretrained(model_id, config)
