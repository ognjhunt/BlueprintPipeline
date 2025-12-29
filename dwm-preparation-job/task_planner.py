#!/usr/bin/env python3
"""
Task Planner for DWM - Generates episode action sequences.

Uses Gemini 3.0 Pro with Grounded Search to plan manipulation tasks:
1. Generate meaningful action sequences for each task
2. Break tasks into DWM-compatible clips (49 frames each)
3. Define hand trajectories and camera paths for each action
4. Create semantic text prompts for DWM conditioning

This module bridges scene analysis to actual episode generation.
"""

import json
import math
import os
import sys
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add parent to path for imports
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene_analyzer import (
    EnvironmentType,
    ObjectAffordance,
    ObjectSemantics,
    SceneAnalysisResult,
    TaskTemplate,
)
from models import (
    CameraPose,
    CameraTrajectory,
    HandActionType,
    HandPose,
    HandTrajectory,
    TrajectoryType,
)

try:
    from tools.llm_client import create_llm_client, LLMResponse
    HAVE_LLM_CLIENT = True
except ImportError:
    HAVE_LLM_CLIENT = False
    create_llm_client = None


# =============================================================================
# Constants
# =============================================================================

# DWM video parameters
DWM_FRAMES_PER_CLIP = 49
DWM_FPS = 24.0
DWM_CLIP_DURATION = DWM_FRAMES_PER_CLIP / DWM_FPS  # ~2.04 seconds

# Mapping from affordances to hand actions
AFFORDANCE_TO_HAND_ACTION = {
    ObjectAffordance.GRASP: HandActionType.GRASP,
    ObjectAffordance.LIFT: HandActionType.LIFT,
    ObjectAffordance.PLACE: HandActionType.PLACE,
    ObjectAffordance.PUSH: HandActionType.PUSH,
    ObjectAffordance.PULL: HandActionType.PULL,
    ObjectAffordance.ROTATE: HandActionType.ROTATE,
    ObjectAffordance.OPEN: HandActionType.PULL,
    ObjectAffordance.CLOSE: HandActionType.PUSH,
    ObjectAffordance.EXTEND: HandActionType.PULL,
    ObjectAffordance.RETRACT: HandActionType.PUSH,
    ObjectAffordance.INSERT: HandActionType.PLACE,
    ObjectAffordance.REMOVE: HandActionType.LIFT,
    ObjectAffordance.SLIDE: HandActionType.SLIDE,
}


# =============================================================================
# Data Models
# =============================================================================


class EpisodePhase(str, Enum):
    """Phases within a manipulation episode."""
    APPROACH = "approach"
    REACH = "reach"
    GRASP = "grasp"
    LIFT = "lift"
    TRANSPORT = "transport"
    POSITION = "position"
    PLACE = "place"
    RELEASE = "release"
    RETRACT = "retract"
    ARTICULATE = "articulate"  # For opening/closing
    IDLE = "idle"


@dataclass
class ActionStep:
    """A single action step within an episode."""
    step_id: str
    phase: EpisodePhase
    action_type: HandActionType
    target_object_id: Optional[str] = None
    target_position: Optional[np.ndarray] = None
    object_state_start: Optional[Dict[str, Any]] = None
    object_state_goal: Optional[Dict[str, Any]] = None

    # Timing
    start_frame: int = 0
    end_frame: int = 0
    duration_seconds: float = 0.0

    # Text description for DWM prompt
    description: str = ""

    # Trajectory hints
    camera_trajectory_type: TrajectoryType = TrajectoryType.REACH_MANIPULATE
    hand_motion_type: str = "linear"  # "linear", "arc", "spiral"

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame


@dataclass
class EpisodeClip:
    """A single DWM clip (49 frames) within an episode."""
    clip_id: str
    clip_index: int  # Index within episode

    # Frames
    start_frame: int
    end_frame: int  # Should be start_frame + 49

    # Actions in this clip
    action_steps: List[ActionStep] = field(default_factory=list)
    step_goals: List[Dict[str, Any]] = field(default_factory=list)

    # Primary action (for DWM prompt)
    primary_action: Optional[HandActionType] = None
    primary_target: Optional[str] = None

    # Text prompt for DWM
    text_prompt: str = ""

    # Trajectories (to be filled by trajectory generator)
    camera_trajectory: Optional[CameraTrajectory] = None
    hand_trajectory: Optional[HandTrajectory] = None
    object_states_start: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    object_states_end: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame


@dataclass
class ManipulationEpisode:
    """A complete manipulation episode composed of multiple clips."""
    episode_id: str
    task_id: str
    task_name: str
    description: str

    # Source task template
    source_template: Optional[TaskTemplate] = None

    # Environment context
    environment_type: EnvironmentType = EnvironmentType.GENERIC
    scene_id: str = ""

    # Actions
    action_steps: List[ActionStep] = field(default_factory=list)

    # Clips (each is 49 frames)
    clips: List[EpisodeClip] = field(default_factory=list)
    object_states_start: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    object_states_end: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Objects involved
    source_objects: List[str] = field(default_factory=list)
    target_objects: List[str] = field(default_factory=list)
    manipulated_object: Optional[str] = None

    # Timing
    total_frames: int = 0
    total_duration_seconds: float = 0.0

    # Metadata
    difficulty: str = "medium"
    priority: int = 1

    @property
    def clip_count(self) -> int:
        return len(self.clips)

    def get_all_text_prompts(self) -> List[str]:
        """Get text prompts for all clips."""
        return [clip.text_prompt for clip in self.clips]


@dataclass
class TaskPlannerOutput:
    """Output from the task planner."""
    scene_id: str
    environment_type: EnvironmentType
    episodes: List[ManipulationEpisode] = field(default_factory=list)

    # Statistics
    total_clips: int = 0
    total_frames: int = 0
    total_duration_seconds: float = 0.0

    # Metadata
    llm_sources: List[Dict[str, str]] = field(default_factory=list)
    planning_confidence: float = 0.0


# =============================================================================
# Task Planner
# =============================================================================


class TaskPlanner:
    """
    Plans manipulation episodes from scene analysis.

    Uses Gemini 3.0 Pro with Grounded Search to generate:
    1. Detailed action sequences for each task
    2. Frame-level timing for each action
    3. Text prompts for DWM conditioning

    Usage:
        planner = TaskPlanner()
        output = planner.plan_episodes(analysis_result)

        for episode in output.episodes:
            for clip in episode.clips:
                print(f"Clip {clip.clip_index}: {clip.text_prompt}")
    """

    def __init__(
        self,
        frames_per_clip: int = DWM_FRAMES_PER_CLIP,
        fps: float = DWM_FPS,
        verbose: bool = True,
    ):
        """Initialize the task planner."""
        self.frames_per_clip = frames_per_clip
        self.fps = fps
        self.clip_duration = frames_per_clip / fps
        self.verbose = verbose
        self._client = None

    def log(self, msg: str, level: str = "INFO") -> None:
        """Log a message."""
        if self.verbose:
            print(f"[TASK-PLANNER] [{level}] {msg}")

    def _get_client(self):
        """Get or create LLM client."""
        if self._client is None:
            if not HAVE_LLM_CLIENT:
                raise ImportError(
                    "LLM client not available. Install google-genai or openai package."
                )
            self._client = create_llm_client()
        return self._client

    def plan_episodes(
        self,
        analysis: SceneAnalysisResult,
        max_episodes: int = 10,
        prioritize_by: str = "dwm_relevance",  # "dwm_relevance", "difficulty", "variety"
    ) -> TaskPlannerOutput:
        """
        Plan manipulation episodes from scene analysis.

        Args:
            analysis: SceneAnalysisResult from scene analyzer
            max_episodes: Maximum number of episodes to generate
            prioritize_by: How to prioritize task selection

        Returns:
            TaskPlannerOutput with planned episodes
        """
        self.log(f"Planning episodes for scene: {analysis.scene_id}")
        self.log(f"Available tasks: {len(analysis.task_templates)}")

        # Select tasks to plan
        selected_tasks = self._select_tasks(
            analysis.task_templates,
            max_episodes,
            prioritize_by,
        )

        self.log(f"Selected {len(selected_tasks)} tasks for episode planning")

        # Generate episodes for each task
        episodes = []
        for task in selected_tasks:
            try:
                episode = self._plan_single_episode(
                    task=task,
                    analysis=analysis,
                )
                if episode:
                    episodes.append(episode)
            except Exception as e:
                self.log(f"Failed to plan episode for {task.task_id}: {e}", "ERROR")

        # Calculate totals
        total_clips = sum(ep.clip_count for ep in episodes)
        total_frames = sum(ep.total_frames for ep in episodes)
        total_duration = sum(ep.total_duration_seconds for ep in episodes)

        self.log(f"Planned {len(episodes)} episodes, {total_clips} clips, "
                 f"{total_duration:.1f}s total")

        return TaskPlannerOutput(
            scene_id=analysis.scene_id,
            environment_type=analysis.environment_type,
            episodes=episodes,
            total_clips=total_clips,
            total_frames=total_frames,
            total_duration_seconds=total_duration,
        )

    def _select_tasks(
        self,
        tasks: List[TaskTemplate],
        max_tasks: int,
        prioritize_by: str,
    ) -> List[TaskTemplate]:
        """Select which tasks to plan based on priority."""

        if not tasks:
            return []

        # Sort by priority
        if prioritize_by == "dwm_relevance":
            # Prioritize by dwm_clip_count (more clips = more training data)
            sorted_tasks = sorted(tasks, key=lambda t: (t.priority, t.dwm_clip_count), reverse=True)
        elif prioritize_by == "difficulty":
            # Easy tasks first
            difficulty_order = {"easy": 0, "medium": 1, "hard": 2}
            sorted_tasks = sorted(tasks, key=lambda t: difficulty_order.get(t.difficulty, 1))
        else:
            # Variety - try to get different task types
            sorted_tasks = tasks

        return sorted_tasks[:max_tasks]

    def _plan_single_episode(
        self,
        task: TaskTemplate,
        analysis: SceneAnalysisResult,
    ) -> Optional[ManipulationEpisode]:
        """Plan a single manipulation episode."""

        self.log(f"Planning episode: {task.task_name}")

        # Use LLM to get detailed action sequence if available
        detailed_actions = self._get_detailed_actions(task, analysis)

        if not detailed_actions:
            # Fall back to task template actions
            detailed_actions = task.action_sequence

        # Convert to ActionStep objects
        action_steps = self._parse_action_sequence(detailed_actions, analysis)

        if not action_steps:
            self.log(f"No valid actions for task {task.task_id}", "WARNING")
            return None

        # Assign frame timing to actions
        self._assign_frame_timing(action_steps)

        # Split into clips
        clips = self._split_into_clips(
            action_steps, task.task_id, analysis.object_states
        )

        # Generate text prompts for each clip
        for clip in clips:
            clip.text_prompt = self._generate_clip_prompt(clip, task, analysis)

        # Calculate totals
        total_frames = sum(clip.frame_count for clip in clips)
        total_duration = total_frames / self.fps

        episode = ManipulationEpisode(
            episode_id=f"ep_{task.task_id}_{uuid.uuid4().hex[:8]}",
            task_id=task.task_id,
            task_name=task.task_name,
            description=task.description,
            source_template=task,
            environment_type=analysis.environment_type,
            scene_id=analysis.scene_id,
            action_steps=action_steps,
            clips=clips,
            object_states_start=clips[0].object_states_start if clips else {},
            object_states_end=clips[-1].object_states_end if clips else {},
            source_objects=task.source_objects,
            target_objects=task.target_objects,
            total_frames=total_frames,
            total_duration_seconds=total_duration,
            difficulty=task.difficulty,
            priority=task.priority,
        )

        return episode

    def _get_detailed_actions(
        self,
        task: TaskTemplate,
        analysis: SceneAnalysisResult,
    ) -> List[Dict[str, Any]]:
        """Use LLM to get detailed action sequence for a task."""

        if not HAVE_LLM_CLIENT:
            return task.action_sequence

        # Build prompt
        prompt = self._build_action_planning_prompt(task, analysis)

        try:
            client = self._get_client()
            response = client.generate(
                prompt=prompt,
                json_output=True,
                use_web_search=True,
                temperature=0.4,
                max_tokens=8000,
            )

            data = response.parse_json()
            return data.get("detailed_actions", task.action_sequence)

        except Exception as e:
            self.log(f"LLM action planning failed: {e}", "WARNING")
            return task.action_sequence

    def _build_action_planning_prompt(
        self,
        task: TaskTemplate,
        analysis: SceneAnalysisResult,
    ) -> str:
        """Build prompt for detailed action planning."""

        # Get relevant objects
        relevant_objects = []
        for obj in analysis.object_semantics:
            if obj.object_id in task.source_objects or obj.object_id in task.target_objects:
                relevant_objects.append({
                    "id": obj.object_id,
                    "category": obj.category,
                    "affordances": [a.value for a in obj.affordances],
                    "is_articulated": obj.is_articulated,
                })

        prompt = f"""You are an expert in robotics manipulation planning for egocentric video generation.

Generate a detailed action sequence for this manipulation task. Each action will become a segment of an egocentric manipulation video (DWM - Dexterous World Model).

## Task Information

Task: {task.task_name}
Description: {task.description}
Environment: {analysis.environment_type.value}
Difficulty: {task.difficulty}

Original action sequence:
{json.dumps(task.action_sequence, indent=2)}

Relevant objects:
{json.dumps(relevant_objects, indent=2)}

## Your Task

Expand the action sequence into detailed, frame-level actions suitable for video generation.

Each action should be a specific hand manipulation that takes 1-3 seconds (24-72 frames at 24fps).

Action types to use:
- approach: Move toward an object (REACH trajectory)
- reach: Extend hand toward object (REACH action)
- grasp: Close hand around object (GRASP action)
- lift: Lift object upward (LIFT action)
- transport: Move object through space (combined motion)
- position: Align object with target (PLACE preparation)
- place: Lower object onto surface/into container (PLACE action)
- release: Open hand to release object (release action)
- retract: Pull hand back after action
- open: Open an articulated object (PULL for doors/drawers)
- close: Close an articulated object (PUSH for doors/drawers)
- push: Push an object (PUSH action)
- pull: Pull an object (PULL action)
- rotate: Rotate an object (ROTATE action)

## Output Format

Return ONLY valid JSON:

{{
  "detailed_actions": [
    {{
      "action": "approach",
      "target": "counter",
      "description": "Walk toward the counter where the dirty dish is placed",
      "duration_seconds": 2.0,
      "hand_action": "reach",
      "camera_motion": "approach"
    }},
    {{
      "action": "reach",
      "target": "dish",
      "description": "Extend right hand toward the dirty plate on the counter",
      "duration_seconds": 1.5,
      "hand_action": "reach",
      "camera_motion": "stable"
    }},
    ...
  ],
  "estimated_total_duration": 8.5,
  "notes": "Optional planning notes"
}}

## Guidelines

1. **Natural Motion**: Actions should flow naturally as a person would perform them
2. **Appropriate Timing**: Simple actions (grasp) ~1s, complex (transport) ~2-3s
3. **Include Transitions**: Add approach/retract phases between major actions
4. **Articulation Handling**: For doors/drawers, include handle grasp + motion
5. **Camera Stability**: Note when camera should be stable vs moving

Return ONLY the JSON.
"""
        return prompt

    def _parse_action_sequence(
        self,
        actions: List[Dict[str, Any]],
        analysis: SceneAnalysisResult,
    ) -> List[ActionStep]:
        """Parse action sequence into ActionStep objects."""

        steps = []
        current_states = {k: dict(v) for k, v in analysis.object_states.items()}
        for i, action in enumerate(actions):
            action_name = action.get("action", "reach").lower()

            # Map action name to phase and hand action
            phase = self._action_to_phase(action_name)
            hand_action = self._action_to_hand_action(action_name)

            # Get target position if available
            target_id = action.get("target")
            target_position = self._get_object_position(target_id, analysis)
            start_state, goal_state = self._get_object_states_for_action(
                action_name, target_id, current_states, target_position
            )
            if goal_state and target_id:
                current_states[target_id] = goal_state

            step = ActionStep(
                step_id=f"step_{i:03d}",
                phase=phase,
                action_type=hand_action,
                target_object_id=target_id,
                target_position=target_position,
                object_state_start=start_state,
                object_state_goal=goal_state,
                duration_seconds=action.get("duration_seconds", 1.5),
                description=action.get("description", f"{action_name} {target_id}"),
                camera_trajectory_type=self._get_trajectory_type(action_name),
                hand_motion_type=action.get("hand_motion", "linear"),
            )
            steps.append(step)

        return steps

    def _action_to_phase(self, action_name: str) -> EpisodePhase:
        """Map action name to episode phase."""
        phase_map = {
            "approach": EpisodePhase.APPROACH,
            "reach": EpisodePhase.REACH,
            "grasp": EpisodePhase.GRASP,
            "grip": EpisodePhase.GRASP,
            "lift": EpisodePhase.LIFT,
            "raise": EpisodePhase.LIFT,
            "transport": EpisodePhase.TRANSPORT,
            "move": EpisodePhase.TRANSPORT,
            "carry": EpisodePhase.TRANSPORT,
            "position": EpisodePhase.POSITION,
            "align": EpisodePhase.POSITION,
            "place": EpisodePhase.PLACE,
            "put": EpisodePhase.PLACE,
            "insert": EpisodePhase.PLACE,
            "release": EpisodePhase.RELEASE,
            "drop": EpisodePhase.RELEASE,
            "retract": EpisodePhase.RETRACT,
            "withdraw": EpisodePhase.RETRACT,
            "open": EpisodePhase.ARTICULATE,
            "close": EpisodePhase.ARTICULATE,
            "pull": EpisodePhase.ARTICULATE,
            "push": EpisodePhase.ARTICULATE,
            "rotate": EpisodePhase.ARTICULATE,
            "idle": EpisodePhase.IDLE,
            "wait": EpisodePhase.IDLE,
        }
        return phase_map.get(action_name, EpisodePhase.REACH)

    def _action_to_hand_action(self, action_name: str) -> HandActionType:
        """Map action name to hand action type."""
        action_map = {
            "approach": HandActionType.REACH,
            "reach": HandActionType.REACH,
            "grasp": HandActionType.GRASP,
            "grip": HandActionType.GRASP,
            "lift": HandActionType.LIFT,
            "raise": HandActionType.LIFT,
            "transport": HandActionType.LIFT,
            "move": HandActionType.LIFT,
            "carry": HandActionType.LIFT,
            "position": HandActionType.PLACE,
            "align": HandActionType.PLACE,
            "place": HandActionType.PLACE,
            "put": HandActionType.PLACE,
            "insert": HandActionType.PLACE,
            "release": HandActionType.PLACE,
            "drop": HandActionType.PLACE,
            "retract": HandActionType.REACH,
            "withdraw": HandActionType.REACH,
            "open": HandActionType.PULL,
            "pull": HandActionType.PULL,
            "close": HandActionType.PUSH,
            "push": HandActionType.PUSH,
            "rotate": HandActionType.ROTATE,
            "slide": HandActionType.SLIDE,
        }
        return action_map.get(action_name, HandActionType.REACH)

    def _get_trajectory_type(self, action_name: str) -> TrajectoryType:
        """Get camera trajectory type for an action."""
        if action_name in ["approach", "walk", "move"]:
            return TrajectoryType.APPROACH
        elif action_name in ["orbit", "inspect"]:
            return TrajectoryType.ORBIT
        else:
            return TrajectoryType.REACH_MANIPULATE

    def _get_object_position(
        self,
        object_id: Optional[str],
        analysis: SceneAnalysisResult,
    ) -> Optional[np.ndarray]:
        """Get object position from analysis."""
        if not object_id:
            return None

        state = analysis.object_states.get(object_id)
        if state and "position" in state:
            try:
                return np.array(state["position"], dtype=float)
            except Exception:
                return None

        # Try to find in object semantics
        for obj in analysis.object_semantics:
            if obj.object_id == object_id and obj.typical_height_m:
                return np.array([0, obj.typical_height_m, 0.0], dtype=float)

        return None

    def _get_object_states_for_action(
        self,
        action_name: str,
        target_id: Optional[str],
        current_states: Dict[str, Dict[str, Any]],
        target_position: Optional[np.ndarray],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Derive start/goal state for an action, threading pose/articulation."""
        if not target_id:
            return None, None

        start_state = dict(current_states.get(target_id, {}))
        goal_state = dict(start_state)

        if target_position is not None:
            goal_state["position"] = target_position.tolist()

        articulation_targets = {
            "open": "opened",
            "close": "closed",
            "pull": "extended",
            "push": "compressed",
            "extend": "extended",
            "retract": "retracted",
            "rotate": "rotated",
        }
        if action_name in articulation_targets:
            goal_state["articulation"] = articulation_targets[action_name]

        if action_name in ["grasp", "grip"]:
            goal_state["held"] = True
        if action_name in ["lift", "transport", "move", "carry"]:
            goal_state["held"] = True
        if action_name in ["place", "insert", "release", "drop"]:
            goal_state["held"] = False

        if goal_state == start_state:
            return start_state or None, None

        return start_state or None, goal_state

    def _assign_frame_timing(self, steps: List[ActionStep]) -> None:
        """Assign frame timing to action steps."""
        current_frame = 0

        for step in steps:
            step.start_frame = current_frame
            frame_count = int(step.duration_seconds * self.fps)
            step.end_frame = current_frame + frame_count
            current_frame = step.end_frame

    def _split_into_clips(
        self,
        steps: List[ActionStep],
        task_id: str,
        initial_states: Dict[str, Dict[str, Any]],
    ) -> List[EpisodeClip]:
        """Split action steps into DWM clips (49 frames each)."""

        if not steps:
            return []

        total_frames = steps[-1].end_frame
        num_clips = math.ceil(total_frames / self.frames_per_clip)

        clips = []
        state_tracker = {k: dict(v) for k, v in initial_states.items()}
        applied_steps: set[str] = set()
        for clip_idx in range(num_clips):
            start_frame = clip_idx * self.frames_per_clip
            end_frame = min(start_frame + self.frames_per_clip, total_frames)

            # Find actions that overlap with this clip
            clip_steps = []
            step_goals = []
            for step in steps:
                if step.end_frame > start_frame and step.start_frame < end_frame:
                    clip_steps.append(step)
                    if step.object_state_goal:
                        step_goals.append(
                            {
                                "step_id": step.step_id,
                                "target_object_id": step.target_object_id,
                                "goal_state": step.object_state_goal,
                                "start_state": step.object_state_start,
                            }
                        )

            # Determine primary action for the clip
            primary_action = None
            primary_target = None
            if clip_steps:
                # Use the action that occupies the most frames in this clip
                best_step = max(clip_steps, key=lambda s: min(s.end_frame, end_frame) - max(s.start_frame, start_frame))
                primary_action = best_step.action_type
                primary_target = best_step.target_object_id

            clip_start_state = {k: dict(v) for k, v in state_tracker.items()}

            for step in clip_steps:
                if (
                    step.object_state_goal
                    and step.end_frame <= end_frame
                    and step.step_id not in applied_steps
                    and step.target_object_id
                ):
                    state_tracker[step.target_object_id] = dict(step.object_state_goal)
                    applied_steps.add(step.step_id)

            clip_end_state = {k: dict(v) for k, v in state_tracker.items()}

            clip = EpisodeClip(
                clip_id=f"{task_id}_clip_{clip_idx:03d}",
                clip_index=clip_idx,
                start_frame=start_frame,
                end_frame=end_frame,
                action_steps=clip_steps,
                step_goals=step_goals,
                primary_action=primary_action,
                primary_target=primary_target,
                object_states_start=clip_start_state,
                object_states_end=clip_end_state,
            )
            clips.append(clip)

        return clips

    def _generate_clip_prompt(
        self,
        clip: EpisodeClip,
        task: TaskTemplate,
        analysis: SceneAnalysisResult,
    ) -> str:
        """Generate a text prompt for DWM conditioning."""

        # Build description from clip actions
        action_descriptions = []
        for step in clip.action_steps:
            action_descriptions.append(step.description)

        if action_descriptions:
            combined_desc = ". ".join(action_descriptions)
        else:
            combined_desc = f"Performing {task.task_name}"

        # Get primary action info
        if clip.primary_action and clip.primary_target:
            primary_desc = f"{clip.primary_action.value} the {clip.primary_target}"
        else:
            primary_desc = task.task_name

        # Build final prompt
        prompt = f"A person's hands {combined_desc.lower()}."

        # Add environment context
        env_context = {
            EnvironmentType.KITCHEN: "in a kitchen",
            EnvironmentType.WAREHOUSE: "in a warehouse",
            EnvironmentType.GROCERY: "in a grocery store",
            EnvironmentType.BEDROOM: "in a bedroom",
            EnvironmentType.OFFICE: "in an office",
            EnvironmentType.LAUNDRY: "in a laundry room",
            EnvironmentType.LAB: "in a laboratory",
        }

        env_str = env_context.get(analysis.environment_type, "")
        if env_str:
            prompt = f"{prompt} The scene is {env_str}."

        return prompt


# =============================================================================
# Convenience Functions
# =============================================================================


def plan_episodes(
    analysis: SceneAnalysisResult,
    max_episodes: int = 10,
    verbose: bool = True,
) -> TaskPlannerOutput:
    """Convenience function to plan episodes from analysis."""
    planner = TaskPlanner(verbose=verbose)
    return planner.plan_episodes(analysis, max_episodes=max_episodes)


def estimate_clip_count(analysis: SceneAnalysisResult) -> int:
    """Estimate total number of DWM clips needed for a scene."""
    return sum(task.dwm_clip_count for task in analysis.task_templates)


# =============================================================================
# CLI Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse
    from scene_analyzer import analyze_scene

    parser = argparse.ArgumentParser(description="Plan DWM episodes from scene analysis")
    parser.add_argument("manifest_path", type=Path, help="Path to scene_manifest.json")
    parser.add_argument("--max-episodes", type=int, default=5, help="Max episodes to plan")
    parser.add_argument("--output", "-o", type=Path, help="Output path for plan JSON")
    parser.add_argument("--quiet", "-q", action="store_true", help="Reduce output")

    args = parser.parse_args()

    # First analyze the scene
    print("Analyzing scene...")
    analysis = analyze_scene(args.manifest_path, verbose=not args.quiet)

    # Then plan episodes
    print("\nPlanning episodes...")
    output = plan_episodes(analysis, max_episodes=args.max_episodes, verbose=not args.quiet)

    # Convert to JSON-serializable dict
    output_data = {
        "scene_id": output.scene_id,
        "environment_type": output.environment_type.value,
        "total_episodes": len(output.episodes),
        "total_clips": output.total_clips,
        "total_frames": output.total_frames,
        "total_duration_seconds": output.total_duration_seconds,
        "episodes": [
            {
                "episode_id": ep.episode_id,
                "task_name": ep.task_name,
                "description": ep.description,
                "clip_count": ep.clip_count,
                "total_frames": ep.total_frames,
                "clips": [
                    {
                        "clip_id": clip.clip_id,
                        "clip_index": clip.clip_index,
                        "frame_range": [clip.start_frame, clip.end_frame],
                        "primary_action": clip.primary_action.value if clip.primary_action else None,
                        "text_prompt": clip.text_prompt,
                    }
                    for clip in ep.clips
                ],
            }
            for ep in output.episodes
        ],
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output_data, indent=2))
        print(f"\nPlan saved to: {args.output}")
    else:
        print("\n" + json.dumps(output_data, indent=2))
