"""
Composite Task Chaining System for Isaac Lab-Arena.

This module enables dynamic composition of primitive tasks into
complex, long-horizon workflows for policy evaluation.

Key Features:
- Chain affordance-based tasks into workflows
- Define task dependencies and handoff states
- Generate composite evaluation benchmarks
- Support hierarchical skill decomposition

Distinction from Genie Sim:
- Genie Sim: Generates TRAINING data for individual tasks
- This Module: Evaluates POLICIES on composite task chains

Example Workflow - Dish Loading:
    1. approach_dishwasher (navigation)
    2. open_dishwasher_door (articulation)
    3. pick_dish_from_counter (manipulation)
    4. place_dish_in_rack (manipulation)
    5. close_dishwasher_door (articulation)

Usage:
    from tools.arena_integration.composite_tasks import (
        CompositeTask,
        TaskChain,
        CompositeTaskBuilder,
        evaluate_composite_task
    )

    chain = TaskChain([
        TaskNode("open_door", task=ArenaTask.open_articulated()),
        TaskNode("pick_object", task=ArenaTask.pick_object()),
        TaskNode("place_object", task=ArenaTask.pick_and_place()),
    ])

    composite = CompositeTaskBuilder().build(chain, scene)
    results = evaluate_composite_task(env, policy, composite)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
import os
from pathlib import Path
import time
from typing import Any, Callable, Optional, Protocol, Union

import numpy as np
import requests

from .components import (
    ArenaScene,
    ArenaTask,
    ArenaEmbodiment,
    ArenaObject,
    ArenaEnvironmentSpec,
    TaskDifficulty,
)
from .affordances import AffordanceType


# =============================================================================
# TASK NODE AND TRANSITIONS
# =============================================================================

class TransitionType(str, Enum):
    """Types of transitions between task nodes."""
    SEQUENTIAL = "sequential"           # Complete current before next
    PARALLEL = "parallel"               # Can execute simultaneously
    CONDITIONAL = "conditional"         # Based on condition
    LOOP = "loop"                       # Repeat until condition
    OPTIONAL = "optional"               # Can be skipped


@dataclass
class TransitionCondition:
    """Condition for task transitions."""
    condition_type: str                 # "success", "timeout", "state"
    condition_fn: Optional[Callable[[dict], bool]] = None
    state_key: Optional[str] = None
    state_value: Optional[Any] = None
    timeout_steps: Optional[int] = None

    def evaluate(self, context: dict[str, Any]) -> bool:
        """Evaluate transition condition."""
        if self.condition_type == "success":
            return context.get("task_success", False)
        elif self.condition_type == "timeout":
            return context.get("steps", 0) >= (self.timeout_steps or 500)
        elif self.condition_type == "state":
            return context.get(self.state_key) == self.state_value
        elif self.condition_fn:
            return self.condition_fn(context)
        return True


@dataclass
class TaskTransition:
    """Transition between task nodes."""
    target_node: str                    # Target node ID
    transition_type: TransitionType
    condition: Optional[TransitionCondition] = None
    priority: int = 0                   # Higher = evaluated first

    def can_transition(self, context: dict[str, Any]) -> bool:
        """Check if transition is possible."""
        if self.condition is None:
            return True
        return self.condition.evaluate(context)


@dataclass
class HandoffState:
    """State passed between task nodes."""
    object_in_hand: Optional[str] = None
    gripper_state: str = "open"         # open, closed
    ee_position: Optional[tuple[float, float, float]] = None
    target_object: Optional[str] = None
    custom_state: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskNode:
    """
    A node in the task chain representing a subtask.

    Each node encapsulates:
    - The primitive task to execute
    - Success/failure criteria
    - Transitions to other nodes
    - Handoff state requirements
    """
    node_id: str
    task: ArenaTask
    transitions: list[TaskTransition] = field(default_factory=list)

    # Execution settings
    max_steps: Optional[int] = None     # Override task max_steps
    retry_count: int = 0                # Retries on failure

    # Handoff requirements
    required_state: Optional[HandoffState] = None   # State needed to start
    produced_state: Optional[HandoffState] = None   # State after completion

    # Success criteria override
    success_fn: Optional[Callable[[dict], bool]] = None

    def add_transition(
        self,
        target: str,
        transition_type: TransitionType = TransitionType.SEQUENTIAL,
        condition: Optional[TransitionCondition] = None,
    ) -> "TaskNode":
        """Add transition to another node."""
        self.transitions.append(TaskTransition(
            target_node=target,
            transition_type=transition_type,
            condition=condition,
        ))
        return self

    def on_success(self, target: str) -> "TaskNode":
        """Add transition on success."""
        return self.add_transition(
            target,
            TransitionType.CONDITIONAL,
            TransitionCondition("success"),
        )

    def on_failure(self, target: str) -> "TaskNode":
        """Add transition on failure."""
        return self.add_transition(
            target,
            TransitionType.CONDITIONAL,
            TransitionCondition("success", condition_fn=lambda c: not c.get("task_success", True)),
        )


# =============================================================================
# TASK CHAIN
# =============================================================================

@dataclass
class TaskChain:
    """
    A chain of tasks forming a composite workflow.

    The chain is a directed graph where nodes are tasks and edges
    are transitions with conditions.
    """
    nodes: list[TaskNode]
    start_node: Optional[str] = None
    end_nodes: list[str] = field(default_factory=list)

    # Chain metadata
    name: str = "composite_task"
    description: str = ""
    total_max_steps: int = 2000
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.nodes and self.start_node is None:
            self.start_node = self.nodes[0].node_id

        # Build node lookup
        self._node_map = {node.node_id: node for node in self.nodes}

        # Auto-detect end nodes if not specified
        if not self.end_nodes:
            nodes_with_outgoing = {t.target_node for n in self.nodes for t in n.transitions}
            self.end_nodes = [n.node_id for n in self.nodes if n.node_id not in nodes_with_outgoing]
            # If no end nodes found, last node is end
            if not self.end_nodes and self.nodes:
                self.end_nodes = [self.nodes[-1].node_id]

    def get_node(self, node_id: str) -> Optional[TaskNode]:
        """Get node by ID."""
        return self._node_map.get(node_id)

    def get_next_nodes(
        self,
        current_node: str,
        context: dict[str, Any]
    ) -> list[str]:
        """Get next nodes based on current context."""
        node = self.get_node(current_node)
        if not node:
            return []

        # Sort transitions by priority
        sorted_transitions = sorted(
            node.transitions,
            key=lambda t: t.priority,
            reverse=True
        )

        next_nodes = []
        for transition in sorted_transitions:
            if transition.can_transition(context):
                if transition.transition_type == TransitionType.PARALLEL:
                    next_nodes.append(transition.target_node)
                else:
                    # For sequential/conditional, take first valid
                    return [transition.target_node]

        return next_nodes

    def validate(self) -> list[str]:
        """Validate chain structure."""
        errors = []

        if not self.nodes:
            errors.append("Chain has no nodes")
            return errors

        if self.start_node not in self._node_map:
            errors.append(f"Start node '{self.start_node}' not found")

        # Check all transition targets exist
        for node in self.nodes:
            for transition in node.transitions:
                if transition.target_node not in self._node_map:
                    errors.append(
                        f"Node '{node.node_id}' has transition to unknown node "
                        f"'{transition.target_node}'"
                    )

        # Check for unreachable nodes
        reachable = {self.start_node}
        queue = [self.start_node]
        while queue:
            current = queue.pop(0)
            node = self.get_node(current)
            if node:
                for t in node.transitions:
                    if t.target_node not in reachable:
                        reachable.add(t.target_node)
                        queue.append(t.target_node)

        unreachable = set(self._node_map.keys()) - reachable
        for node_id in unreachable:
            errors.append(f"Node '{node_id}' is unreachable from start")

        return errors

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "total_max_steps": self.total_max_steps,
            "start_node": self.start_node,
            "end_nodes": self.end_nodes,
            "metadata": self.metadata,
            "nodes": [
                {
                    "node_id": n.node_id,
                    "task_id": n.task.task_id,
                    "task_name": n.task.name,
                    "max_steps": n.max_steps or n.task.config.max_steps,
                    "transitions": [
                        {
                            "target": t.target_node,
                            "type": t.transition_type.value,
                        }
                        for t in n.transitions
                    ],
                }
                for n in self.nodes
            ],
        }


# =============================================================================
# COMPOSITE TASK
# =============================================================================

@dataclass
class CompositeTask:
    """
    A complete composite task ready for evaluation.

    Combines a task chain with scene context and evaluation settings.
    """
    chain: TaskChain
    scene: ArenaScene
    embodiment: ArenaEmbodiment

    # Evaluation settings
    episode_max_steps: int = 2000
    success_requires_all: bool = True   # All nodes must succeed
    partial_success_threshold: float = 0.5  # For partial credit

    # Reward settings
    subtask_reward_weight: float = 0.3  # Weight of subtask rewards
    completion_bonus: float = 100.0     # Bonus for full completion

    # State tracking
    enable_state_tracking: bool = True

    @property
    def task_id(self) -> str:
        return f"composite_{self.chain.name}"

    @property
    def num_subtasks(self) -> int:
        return len(self.chain.nodes)

    def get_observation_keys(self) -> list[str]:
        """Get union of all subtask observation keys."""
        keys = set()
        for node in self.chain.nodes:
            keys.update(node.task.observation_keys)
        return list(keys)

    def to_arena_task(self) -> ArenaTask:
        """Convert to ArenaTask for compatibility."""
        # Aggregate required affordances
        all_affordances = set()
        for node in self.chain.nodes:
            all_affordances.update(node.task.required_affordances)

        return ArenaTask(
            task_id=self.task_id,
            name=self.chain.name,
            description=self.chain.description,
            required_affordances=list(all_affordances),
            config=ArenaTask.ArenaTaskConfig(
                max_steps=self.episode_max_steps,
                reward_scale=1.0,
            ) if hasattr(ArenaTask, 'ArenaTaskConfig') else type('Config', (), {
                'max_steps': self.episode_max_steps,
                'reward_scale': 1.0,
                'success_threshold': 0.9,
                'early_termination': True,
                'domain_randomization': False,
            })(),
            difficulty=TaskDifficulty.HARD,
            params={"chain": self.chain.to_dict()},
            observation_keys=self.get_observation_keys(),
        )


# =============================================================================
# GOAL DECOMPOSITION
# =============================================================================


@dataclass
class GoalDecompositionConfig:
    """Configuration for goal decomposition with an LLM."""
    enabled: bool = False
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    timeout_s: float = 20.0
    max_retries: int = 2
    retry_backoff_s: float = 1.0


@dataclass
class GoalDecompositionResult:
    """Result of a goal decomposition request."""
    subtasks: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)


class GoalDecomposer(Protocol):
    """Pluggable interface for goal decomposition."""

    def decompose(self, goal: str, scene: ArenaScene) -> GoalDecompositionResult:
        ...


class KeywordGoalDecomposer:
    """Keyword-based goal decomposition fallback."""

    def decompose(self, goal: str, scene: ArenaScene) -> GoalDecompositionResult:
        subtasks = []
        goal_lower = goal.lower()

        if "load" in goal_lower and "dish" in goal_lower:
            subtasks = [
                {"name": "open_dishwasher", "affordance": "Openable"},
                {"name": "pick_dish", "affordance": "Graspable"},
                {"name": "place_dish", "affordance": "Containable"},
                {"name": "close_dishwasher", "affordance": "Openable"},
            ]
        elif "clear" in goal_lower and "table" in goal_lower:
            subtasks = [
                {"name": "pick_item", "affordance": "Graspable"},
                {"name": "place_item", "affordance": "Placeable"},
            ]
        elif "retrieve" in goal_lower or "get" in goal_lower:
            subtasks = [
                {"name": "open_container", "affordance": "Openable"},
                {"name": "pick_object", "affordance": "Graspable"},
            ]
        else:
            # Default: single manipulation
            subtasks = [
                {"name": "pick_object", "affordance": "Graspable"},
                {"name": "place_object", "affordance": "Placeable"},
            ]

        return GoalDecompositionResult(subtasks=subtasks)


class LLMGoalDecomposer:
    """LLM-based goal decomposition client with schema validation."""

    def __init__(self, config: GoalDecompositionConfig):
        self.config = config

    def decompose(self, goal: str, scene: ArenaScene) -> GoalDecompositionResult:
        if not self.config.enabled:
            raise RuntimeError("LLM goal decomposition is disabled by config.")
        if self.config.provider != "openai":
            raise RuntimeError(f"Unsupported LLM provider '{self.config.provider}'.")

        prompt = self._build_prompt(goal, scene)
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert task planner for robotics. "
                        "Return only JSON that matches the provided schema."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": 0.2,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "goal_decomposition",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "subtasks": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "affordance": {
                                            "type": ["string", "null"],
                                            "description": (
                                                "One of the known affordance types "
                                                "or null if unknown."
                                            ),
                                        },
                                    },
                                    "required": ["name", "affordance"],
                                    "additionalProperties": False,
                                },
                                "minItems": 1,
                            },
                        },
                        "required": ["subtasks"],
                        "additionalProperties": False,
                    },
                },
            },
        }

        response_data = self._post_with_retry(payload)
        content = self._extract_content(response_data)
        parsed = json.loads(content)
        subtasks = parsed.get("subtasks", [])
        self._validate_subtasks(subtasks)
        return GoalDecompositionResult(subtasks=subtasks)

    def _build_prompt(self, goal: str, scene: ArenaScene) -> str:
        affordance_list = ", ".join(a.value for a in AffordanceType)
        scene_objects = [
            f"- {obj.name}: {', '.join(a.value for a in obj.affordances) or 'unknown'}"
            for obj in scene.objects
        ]
        scene_summary = "\n".join(scene_objects) if scene_objects else "No objects listed."
        return (
            "Decompose the following high-level goal into an ordered list of robot subtasks.\n"
            "Each subtask must have a concise snake_case name and an affordance label.\n"
            "Goal:\n"
            f"{goal}\n\n"
            "Scene objects and affordances:\n"
            f"{scene_summary}\n\n"
            "Valid affordances:\n"
            f"{affordance_list}\n\n"
            "Return JSON with a top-level 'subtasks' array. Each item must include:\n"
            "- name: snake_case task name\n"
            "- affordance: one of the valid affordances or null\n"
        )

    def _post_with_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set for LLM decomposition.")

        url = os.environ.get("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        last_error: Optional[Exception] = None
        max_attempts = self.config.max_retries + 1
        for attempt in range(1, max_attempts + 1):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.config.timeout_s,
                )
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt < max_attempts:
                    time.sleep(self.config.retry_backoff_s * attempt)
                else:
                    break
        raise RuntimeError(f"LLM request failed after {max_attempts} attempts: {last_error}")

    def _extract_content(self, response_data: dict[str, Any]) -> str:
        try:
            return response_data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("LLM response missing expected content.") from exc

    def _validate_subtasks(self, subtasks: list[dict[str, Any]]) -> None:
        if not isinstance(subtasks, list) or not subtasks:
            raise ValueError("LLM response must include a non-empty subtasks list.")

        valid_affordances = {aff.value for aff in AffordanceType}
        for subtask in subtasks:
            if not isinstance(subtask, dict):
                raise ValueError("Each subtask must be a JSON object.")
            name = subtask.get("name")
            affordance = subtask.get("affordance")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Each subtask must have a non-empty name.")
            if affordance is not None and affordance not in valid_affordances:
                raise ValueError(f"Invalid affordance '{affordance}' in LLM response.")


# =============================================================================
# COMPOSITE TASK BUILDER
# =============================================================================

class CompositeTaskBuilder:
    """
    Builder for creating composite tasks from scene analysis.

    Can automatically generate task chains based on:
    - Scene affordances
    - Target goal states
    - Workflow templates
    """

    # Predefined workflow templates
    WORKFLOW_TEMPLATES = {
        "dish_loading": [
            ("approach", None, "Navigate to dishwasher"),
            ("open_door", AffordanceType.OPENABLE, "Open dishwasher door"),
            ("pick_dish", AffordanceType.GRASPABLE, "Pick up dish"),
            ("place_dish", AffordanceType.CONTAINABLE, "Place dish in rack"),
            ("close_door", AffordanceType.OPENABLE, "Close dishwasher door"),
        ],
        "table_clearing": [
            ("pick_item", AffordanceType.GRASPABLE, "Pick item from table"),
            ("place_item", AffordanceType.PLACEABLE, "Place in container"),
        ],
        "cabinet_retrieval": [
            ("open_cabinet", AffordanceType.OPENABLE, "Open cabinet door"),
            ("pick_object", AffordanceType.GRASPABLE, "Pick object from cabinet"),
            ("close_cabinet", AffordanceType.OPENABLE, "Close cabinet door"),
        ],
        "drawer_organization": [
            ("open_drawer", AffordanceType.OPENABLE, "Open drawer"),
            ("pick_item", AffordanceType.GRASPABLE, "Pick item"),
            ("place_item", AffordanceType.CONTAINABLE, "Place in drawer"),
            ("close_drawer", AffordanceType.OPENABLE, "Close drawer"),
        ],
    }

    def __init__(
        self,
        decomposition_config: Optional["GoalDecompositionConfig"] = None,
        goal_decomposer: Optional["GoalDecomposer"] = None,
    ):
        self.decomposition_config = decomposition_config or GoalDecompositionConfig()
        self.goal_decomposer = goal_decomposer
        self._keyword_decomposer = KeywordGoalDecomposer()

    def build_from_template(
        self,
        template_name: str,
        scene: ArenaScene,
        embodiment: ArenaEmbodiment,
    ) -> CompositeTask:
        """
        Build composite task from predefined template.

        Args:
            template_name: Name of workflow template
            scene: Arena scene
            embodiment: Robot embodiment

        Returns:
            CompositeTask
        """
        if template_name not in self.WORKFLOW_TEMPLATES:
            raise ValueError(f"Unknown template: {template_name}")

        template = self.WORKFLOW_TEMPLATES[template_name]
        nodes = []

        for i, (step_name, affordance, description) in enumerate(template):
            # Find matching object in scene
            target_obj = None
            if affordance:
                matching_objs = scene.get_objects_by_affordance(affordance)
                if matching_objs:
                    target_obj = matching_objs[0]

            # Create appropriate task
            task = self._create_task_for_step(step_name, affordance, target_obj)

            # Create node
            node = TaskNode(
                node_id=f"{step_name}_{i}",
                task=task,
            )

            # Add sequential transition to next node
            if i < len(template) - 1:
                next_step = template[i + 1][0]
                node.on_success(f"{next_step}_{i + 1}")

            nodes.append(node)

        chain = TaskChain(
            nodes=nodes,
            name=template_name,
            description=f"Composite workflow: {template_name}",
        )

        return CompositeTask(
            chain=chain,
            scene=scene,
            embodiment=embodiment,
        )

    def build_from_goal(
        self,
        goal_description: str,
        scene: ArenaScene,
        embodiment: ArenaEmbodiment,
    ) -> CompositeTask:
        """
        Build composite task from natural language goal.

        Uses LLM to decompose goal into subtasks.

        Args:
            goal_description: Natural language goal
            scene: Arena scene
            embodiment: Robot embodiment

        Returns:
            CompositeTask
        """
        decomposition = self._decompose_goal(goal_description, scene)
        subtasks = decomposition.subtasks

        nodes = []
        for i, subtask in enumerate(subtasks):
            task = self._create_task_for_step(
                subtask["name"],
                AffordanceType(subtask["affordance"]) if subtask.get("affordance") else None,
                None,  # Object binding done at runtime
            )

            node = TaskNode(
                node_id=f"step_{i}",
                task=task,
            )

            if i < len(subtasks) - 1:
                node.on_success(f"step_{i + 1}")

            nodes.append(node)

        chain = TaskChain(
            nodes=nodes,
            name=f"goal_{hash(goal_description) % 10000}",
            description=goal_description,
            metadata=decomposition.metadata,
        )

        return CompositeTask(
            chain=chain,
            scene=scene,
            embodiment=embodiment,
        )

    def build_custom(
        self,
        chain: TaskChain,
        scene: ArenaScene,
        embodiment: ArenaEmbodiment,
        **kwargs,
    ) -> CompositeTask:
        """
        Build composite task from custom chain.

        Args:
            chain: Pre-built task chain
            scene: Arena scene
            embodiment: Robot embodiment
            **kwargs: Additional CompositeTask parameters

        Returns:
            CompositeTask
        """
        return CompositeTask(
            chain=chain,
            scene=scene,
            embodiment=embodiment,
            **kwargs,
        )

    def _create_task_for_step(
        self,
        step_name: str,
        affordance: Optional[AffordanceType],
        target_obj: Optional[ArenaObject],
    ) -> ArenaTask:
        """Create ArenaTask for a workflow step."""
        if "pick" in step_name.lower():
            return ArenaTask.pick_object(target_object=target_obj)
        elif "place" in step_name.lower():
            return ArenaTask.pick_and_place(target_object=target_obj)
        elif "open" in step_name.lower():
            return ArenaTask.open_articulated(target_object=target_obj)
        elif "close" in step_name.lower():
            return ArenaTask.open_articulated(target_object=target_obj, target_openness=0.1)
        elif "turn" in step_name.lower() or "knob" in step_name.lower():
            return ArenaTask.turn_knob(target_object=target_obj)
        elif "press" in step_name.lower() or "button" in step_name.lower():
            return ArenaTask.press_button(target_object=target_obj)
        else:
            # Generic manipulation task
            return ArenaTask.pick_and_place(target_object=target_obj)

    def _decompose_goal(
        self,
        goal: str,
        scene: ArenaScene,
    ) -> "GoalDecompositionResult":
        """Decompose goal into subtasks, using LLM with keyword fallback."""
        if self.goal_decomposer:
            try:
                result = self.goal_decomposer.decompose(goal, scene)
                result.metadata.setdefault("goal_decomposition", {})
                result.metadata["goal_decomposition"].setdefault(
                    "provenance",
                    "custom_decomposer",
                )
                return result
            except Exception as exc:
                fallback = self._keyword_decomposer.decompose(goal, scene)
                fallback.metadata["goal_decomposition"] = {
                    "provenance": "keyword_fallback",
                    "llm_enabled": False,
                    "fallback_reason": f"custom_decomposer_failed: {exc}",
                }
                return fallback

        if not self.decomposition_config.enabled:
            fallback = self._keyword_decomposer.decompose(goal, scene)
            fallback.metadata["goal_decomposition"] = {
                "provenance": "keyword_fallback",
                "llm_enabled": False,
            }
            return fallback

        llm_decomposer = LLMGoalDecomposer(self.decomposition_config)
        try:
            result = llm_decomposer.decompose(goal, scene)
            result.metadata["goal_decomposition"] = {
                "provenance": "llm",
                "llm_enabled": True,
                "llm_provider": self.decomposition_config.provider,
                "llm_model": self.decomposition_config.model,
            }
            return result
        except Exception as exc:
            fallback = self._keyword_decomposer.decompose(goal, scene)
            fallback.metadata["goal_decomposition"] = {
                "provenance": "keyword_fallback",
                "llm_enabled": True,
                "llm_provider": self.decomposition_config.provider,
                "llm_model": self.decomposition_config.model,
                "fallback_reason": str(exc),
            }
            return fallback


# =============================================================================
# COMPOSITE TASK EXECUTOR
# =============================================================================

@dataclass
class CompositeExecutionState:
    """State during composite task execution."""
    current_node: str
    completed_nodes: list[str] = field(default_factory=list)
    failed_nodes: list[str] = field(default_factory=list)
    total_steps: int = 0
    node_steps: int = 0
    handoff_state: HandoffState = field(default_factory=HandoffState)
    subtask_rewards: list[float] = field(default_factory=list)


@dataclass
class CompositeExecutionResult:
    """Result of composite task execution."""
    success: bool
    nodes_completed: int
    nodes_total: int
    completion_ratio: float
    total_steps: int
    total_reward: float
    subtask_results: list[dict[str, Any]]
    execution_trace: list[str]
    errors: list[str] = field(default_factory=list)


class CompositeTaskExecutor:
    """
    Executes composite tasks and tracks progress.

    Manages state transitions, handoffs, and success evaluation
    for composite task chains.
    """

    def __init__(self, composite_task: CompositeTask):
        self.task = composite_task
        self.state: Optional[CompositeExecutionState] = None

    def reset(self) -> dict[str, Any]:
        """Reset executor for new episode."""
        self.state = CompositeExecutionState(
            current_node=self.task.chain.start_node,
        )
        return self._get_info()

    def step(
        self,
        task_done: bool,
        task_success: bool,
        task_reward: float,
    ) -> tuple[bool, float, dict[str, Any]]:
        """
        Step the composite task execution.

        Args:
            task_done: Whether current subtask is done
            task_success: Whether current subtask succeeded
            task_reward: Reward from current subtask

        Returns:
            (episode_done, composite_reward, info)
        """
        if self.state is None:
            raise RuntimeError("Must call reset() before step()")

        self.state.total_steps += 1
        self.state.node_steps += 1

        # Accumulate reward
        composite_reward = task_reward * self.task.subtask_reward_weight

        # Check if current subtask is done
        if task_done:
            node_id = self.state.current_node

            if task_success:
                self.state.completed_nodes.append(node_id)
                self.state.subtask_rewards.append(task_reward)
            else:
                self.state.failed_nodes.append(node_id)

            # Determine next node
            context = {
                "task_success": task_success,
                "steps": self.state.node_steps,
            }

            next_nodes = self.task.chain.get_next_nodes(node_id, context)

            if next_nodes:
                # Transition to next node
                self.state.current_node = next_nodes[0]
                self.state.node_steps = 0
            else:
                # No more transitions - episode complete
                episode_success = self._evaluate_success()

                if episode_success:
                    composite_reward += self.task.completion_bonus

                return True, composite_reward, self._get_info(episode_success)

        # Check total step limit
        if self.state.total_steps >= self.task.episode_max_steps:
            return True, composite_reward, self._get_info(False)

        return False, composite_reward, self._get_info()

    def _evaluate_success(self) -> bool:
        """Evaluate overall composite task success."""
        if self.task.success_requires_all:
            # All nodes must be completed successfully
            return (
                len(self.state.completed_nodes) == len(self.task.chain.nodes)
                and len(self.state.failed_nodes) == 0
            )
        else:
            # Partial success based on threshold
            completion_ratio = len(self.state.completed_nodes) / len(self.task.chain.nodes)
            return completion_ratio >= self.task.partial_success_threshold

    def _get_info(self, success: Optional[bool] = None) -> dict[str, Any]:
        """Get current execution info."""
        return {
            "current_node": self.state.current_node,
            "completed_nodes": list(self.state.completed_nodes),
            "failed_nodes": list(self.state.failed_nodes),
            "total_steps": self.state.total_steps,
            "completion_ratio": len(self.state.completed_nodes) / len(self.task.chain.nodes),
            "success": success,
        }

    def get_execution_result(self) -> CompositeExecutionResult:
        """Get final execution result."""
        if self.state is None:
            raise RuntimeError("No execution state")

        return CompositeExecutionResult(
            success=self._evaluate_success(),
            nodes_completed=len(self.state.completed_nodes),
            nodes_total=len(self.task.chain.nodes),
            completion_ratio=len(self.state.completed_nodes) / len(self.task.chain.nodes),
            total_steps=self.state.total_steps,
            total_reward=sum(self.state.subtask_rewards),
            subtask_results=[
                {"node_id": n, "success": n in self.state.completed_nodes}
                for n in [node.node_id for node in self.task.chain.nodes]
            ],
            execution_trace=self.state.completed_nodes + self.state.failed_nodes,
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def build_composite_task(
    template_name: str,
    scene: ArenaScene,
    embodiment: ArenaEmbodiment,
) -> CompositeTask:
    """
    Convenience function to build composite task from template.

    Args:
        template_name: Workflow template name
        scene: Arena scene
        embodiment: Robot embodiment

    Returns:
        CompositeTask
    """
    builder = CompositeTaskBuilder()
    return builder.build_from_template(template_name, scene, embodiment)


def evaluate_composite_task(
    env,  # Isaac Lab environment
    policy,  # Policy to evaluate
    composite: CompositeTask,
    num_episodes: int = 100,
) -> dict[str, Any]:
    """
    Evaluate policy on composite task.

    Args:
        env: Isaac Lab environment
        policy: Policy to evaluate
        composite: Composite task specification
        num_episodes: Number of evaluation episodes

    Returns:
        Evaluation results dictionary
    """
    executor = CompositeTaskExecutor(composite)
    results = []

    for ep in range(num_episodes):
        obs = env.reset()
        executor.reset()
        policy.reset()

        episode_done = False
        while not episode_done:
            action = policy.get_action(obs)
            obs, reward, task_done, info = env.step(action)

            task_success = info.get("success", False)
            episode_done, composite_reward, exec_info = executor.step(
                task_done, task_success, reward
            )

        result = executor.get_execution_result()
        results.append({
            "success": result.success,
            "completion_ratio": result.completion_ratio,
            "total_steps": result.total_steps,
        })

    # Aggregate results
    successes = [r["success"] for r in results]
    completion_ratios = [r["completion_ratio"] for r in results]

    return {
        "composite_task_id": composite.task_id,
        "num_episodes": num_episodes,
        "num_subtasks": composite.num_subtasks,
        "full_success_rate": np.mean(successes),
        "mean_completion_ratio": np.mean(completion_ratios),
        "std_completion_ratio": np.std(completion_ratios),
        "episode_results": results,
    }
