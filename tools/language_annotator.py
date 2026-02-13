#!/usr/bin/env python3
"""
Language Annotator for Episode Generation.

Generates natural language annotations for robot episodes, including:
- Task instructions (what the robot should do)
- Subtask/phase annotations (approach, grasp, lift, etc.)
- Object references (mapping natural language to object IDs)
- Paraphrase variations (diverse ways to describe the same task)

These annotations are critical for training:
- Foundation models (RT-2, OpenVLA, PaLM-E)
- Language-conditioned policies
- Multi-task learning with language grounding
- Natural language interfaces for robots

Usage:
    annotator = LanguageAnnotator()
    annotations = annotator.annotate_episode(episode_data)
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SubtaskAnnotation:
    """Annotation for a subtask/phase within an episode."""

    phase: str  # "approach", "grasp", "lift", "transport", "place", etc.
    instruction: str  # Natural language description of this phase
    start_frame: int
    end_frame: int
    objects_involved: List[str] = field(default_factory=list)


@dataclass
class LanguageAnnotations:
    """
    Complete language annotations for an episode.

    Schema compatible with RT-2, OpenVLA, and other VLA models.
    """

    # Primary task instruction (imperative form)
    task_instruction: str

    # Subtask/phase-level instructions
    subtask_instructions: List[SubtaskAnnotation] = field(default_factory=list)

    # Object reference mapping (natural language -> object_id)
    object_references: Dict[str, str] = field(default_factory=dict)

    # Paraphrase variations of the task instruction
    paraphrase_variations: List[str] = field(default_factory=list)

    # Additional context
    task_category: str = ""  # "pick_and_place", "manipulation", "locomotion"
    difficulty: str = ""  # "easy", "medium", "hard"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_instruction": self.task_instruction,
            "subtask_instructions": [
                {
                    "phase": s.phase,
                    "instruction": s.instruction,
                    "start_frame": s.start_frame,
                    "end_frame": s.end_frame,
                    "objects_involved": s.objects_involved,
                }
                for s in self.subtask_instructions
            ],
            "object_references": self.object_references,
            "paraphrase_variations": self.paraphrase_variations,
            "task_category": self.task_category,
            "difficulty": self.difficulty,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LanguageAnnotations":
        """Create from dictionary."""
        subtasks = [
            SubtaskAnnotation(
                phase=s["phase"],
                instruction=s["instruction"],
                start_frame=s["start_frame"],
                end_frame=s["end_frame"],
                objects_involved=s.get("objects_involved", []),
            )
            for s in data.get("subtask_instructions", [])
        ]

        return cls(
            task_instruction=data.get("task_instruction", ""),
            subtask_instructions=subtasks,
            object_references=data.get("object_references", {}),
            paraphrase_variations=data.get("paraphrase_variations", []),
            task_category=data.get("task_category", ""),
            difficulty=data.get("difficulty", ""),
        )


class LanguageAnnotator:
    """
    Generates language annotations for robot episodes.

    Uses templates and heuristics for local generation,
    with optional VLM (Gemini/GPT-4V) integration for richer annotations.
    """

    # Skill phase templates for generating subtask instructions
    PHASE_TEMPLATES = {
        "home": [
            "Start from the home position",
            "Begin at the default resting pose",
            "Initialize from the neutral position",
        ],
        "approach": [
            "Move the gripper towards the {object}",
            "Approach the {object}",
            "Navigate to the {object}",
            "Position the end-effector near the {object}",
        ],
        "pre_grasp": [
            "Prepare to grasp the {object}",
            "Position fingers around the {object}",
            "Align gripper with the {object}",
        ],
        "grasp": [
            "Close gripper to grasp the {object}",
            "Grip the {object}",
            "Secure hold on the {object}",
            "Grasp the {object} firmly",
        ],
        "lift": [
            "Lift the {object} off the surface",
            "Raise the {object}",
            "Pick up the {object}",
            "Elevate the {object}",
        ],
        "transport": [
            "Move the {object} towards the {target}",
            "Transport the {object} to the {target}",
            "Carry the {object} to the {target}",
            "Transfer the {object} to the {target}",
        ],
        "pre_place": [
            "Position above the {target}",
            "Align the {object} with the {target}",
            "Prepare to place on the {target}",
        ],
        "place": [
            "Place the {object} on the {target}",
            "Set down the {object} on the {target}",
            "Release the {object} onto the {target}",
            "Put the {object} on the {target}",
        ],
        "release": [
            "Open the gripper to release the {object}",
            "Let go of the {object}",
            "Release grasp on the {object}",
        ],
        "retract": [
            "Retract the gripper",
            "Move back from the placement location",
            "Return towards home position",
        ],
    }

    # Task instruction templates
    TASK_TEMPLATES = {
        "pick_and_place": [
            "Pick up the {object} and place it on the {target}",
            "Grasp the {object} and move it to the {target}",
            "Take the {object} from {source} and put it on the {target}",
            "Move the {object} to the {target}",
            "Transfer the {object} to the {target}",
        ],
        "pick": [
            "Pick up the {object}",
            "Grasp the {object}",
            "Take the {object}",
            "Grab the {object}",
        ],
        "place": [
            "Place the {object} on the {target}",
            "Put the {object} on the {target}",
            "Set the {object} down on the {target}",
        ],
        "pour": [
            "Pour from the {object} into the {target}",
            "Empty the {object} into the {target}",
            "Transfer contents from {object} to {target}",
        ],
        "stack": [
            "Stack the {object} on top of the {target}",
            "Place the {object} on the {target}",
            "Put the {object} on the {target}",
        ],
        "push": [
            "Push the {object} towards the {target}",
            "Slide the {object} to the {target}",
            "Move the {object} by pushing",
        ],
        "open": [
            "Open the {object}",
            "Pull open the {object}",
            "Slide the {object} open",
        ],
        "close": [
            "Close the {object}",
            "Push the {object} closed",
            "Shut the {object}",
        ],
    }

    # Object description templates
    OBJECT_ADJECTIVES = {
        "color": ["red", "blue", "green", "yellow", "orange", "purple", "white", "black", "silver", "gold"],
        "size": ["small", "large", "medium-sized", "tiny", "big"],
        "material": ["wooden", "plastic", "metal", "glass", "ceramic"],
    }

    def __init__(
        self,
        use_vlm: bool = False,
        vlm_provider: str = "gemini",
        api_key: Optional[str] = None,
    ):
        """
        Initialize the language annotator.

        Args:
            use_vlm: Whether to use VLM for enhanced annotations
            vlm_provider: VLM provider ("gemini", "openai", "anthropic")
            api_key: API key for VLM provider (or set via environment variable)
        """
        self.use_vlm = use_vlm
        self.vlm_provider = vlm_provider
        self.api_key = api_key or os.getenv(f"{vlm_provider.upper()}_API_KEY")

        if use_vlm and not self.api_key:
            logger.warning(
                f"VLM mode enabled but no API key found for {vlm_provider}. "
                "Falling back to template-based generation."
            )
            self.use_vlm = False

    def annotate_episode(
        self,
        task_name: str,
        task_description: Optional[str] = None,
        scene_objects: Optional[List[Dict[str, Any]]] = None,
        skill_segments: Optional[List[Dict[str, Any]]] = None,
        num_frames: int = 0,
        episode_images: Optional[List[Any]] = None,
    ) -> LanguageAnnotations:
        """
        Generate language annotations for an episode.

        Args:
            task_name: Name of the task (e.g., "pick_and_place_cup")
            task_description: Optional description of the task
            scene_objects: List of objects in the scene with metadata
            skill_segments: List of skill segments with frame ranges
            num_frames: Total number of frames in the episode
            episode_images: Optional images for VLM annotation (first, middle, last frames)

        Returns:
            LanguageAnnotations with task instruction, subtasks, and paraphrases
        """
        # Parse task name to infer task type and objects
        task_type, primary_object, target_object = self._parse_task_name(task_name)

        # Build object references from scene objects
        object_refs = self._build_object_references(scene_objects or [])

        # Enrich object names with descriptions
        primary_desc = self._get_object_description(primary_object, scene_objects)
        target_desc = self._get_object_description(target_object, scene_objects) if target_object else ""

        # Generate task instruction
        task_instruction = self._generate_task_instruction(
            task_type, primary_desc, target_desc, task_description
        )

        # Generate subtask annotations
        subtask_annotations = self._generate_subtask_annotations(
            skill_segments or [],
            primary_desc,
            target_desc,
            num_frames,
        )

        # Generate paraphrase variations
        paraphrases = self._generate_paraphrases(
            task_type, primary_desc, target_desc
        )

        # Determine difficulty based on task complexity
        difficulty = self._estimate_difficulty(task_type, scene_objects, skill_segments)

        return LanguageAnnotations(
            task_instruction=task_instruction,
            subtask_instructions=subtask_annotations,
            object_references=object_refs,
            paraphrase_variations=paraphrases,
            task_category=task_type,
            difficulty=difficulty,
        )

    def _parse_task_name(
        self,
        task_name: str,
    ) -> Tuple[str, str, Optional[str]]:
        """
        Parse task name to extract task type and objects.

        Args:
            task_name: Task name like "pick_and_place_red_cup_on_plate"

        Returns:
            Tuple of (task_type, primary_object, target_object)
        """
        # Normalize task name
        name = task_name.lower().replace("-", "_")

        # Common patterns
        if "pick_and_place" in name or "pick_place" in name:
            task_type = "pick_and_place"
            # Try to extract objects
            parts = name.replace("pick_and_place_", "").replace("pick_place_", "")
            if "_on_" in parts:
                obj_parts = parts.split("_on_")
                primary = obj_parts[0].replace("_", " ")
                target = obj_parts[1].replace("_", " ") if len(obj_parts) > 1 else "target"
            elif "_to_" in parts:
                obj_parts = parts.split("_to_")
                primary = obj_parts[0].replace("_", " ")
                target = obj_parts[1].replace("_", " ") if len(obj_parts) > 1 else "target"
            else:
                primary = parts.replace("_", " ")
                target = "target"
            return task_type, primary, target

        elif "pick" in name or "grasp" in name:
            task_type = "pick"
            parts = name.replace("pick_", "").replace("grasp_", "")
            primary = parts.replace("_", " ")
            return task_type, primary, None

        elif "place" in name or "put" in name:
            task_type = "place"
            parts = name.replace("place_", "").replace("put_", "")
            if "_on_" in parts:
                obj_parts = parts.split("_on_")
                primary = obj_parts[0].replace("_", " ")
                target = obj_parts[1].replace("_", " ") if len(obj_parts) > 1 else "target"
            else:
                primary = "object"
                target = parts.replace("_", " ")
            return task_type, primary, target

        elif "pour" in name:
            task_type = "pour"
            parts = name.replace("pour_", "")
            if "_into_" in parts:
                obj_parts = parts.split("_into_")
                primary = obj_parts[0].replace("_", " ")
                target = obj_parts[1].replace("_", " ") if len(obj_parts) > 1 else "container"
            else:
                primary = parts.replace("_", " ") if parts else "container"
                target = "target"
            return task_type, primary, target

        elif "stack" in name:
            task_type = "stack"
            parts = name.replace("stack_", "")
            if "_on_" in parts:
                obj_parts = parts.split("_on_")
                primary = obj_parts[0].replace("_", " ")
                target = obj_parts[1].replace("_", " ") if len(obj_parts) > 1 else "base"
            else:
                primary = parts.replace("_", " ") if parts else "block"
                target = "base"
            return task_type, primary, target

        elif "push" in name or "slide" in name:
            task_type = "push"
            parts = name.replace("push_", "").replace("slide_", "")
            primary = parts.replace("_", " ") if parts else "object"
            return task_type, primary, "target"

        elif "open" in name:
            task_type = "open"
            parts = name.replace("open_", "")
            primary = parts.replace("_", " ") if parts else "drawer"
            return task_type, primary, None

        elif "close" in name:
            task_type = "close"
            parts = name.replace("close_", "")
            primary = parts.replace("_", " ") if parts else "drawer"
            return task_type, primary, None

        # Default fallback
        return "manipulation", "object", "target"

    def _build_object_references(
        self,
        scene_objects: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """Build mapping from natural language descriptions to object IDs."""
        refs: Dict[str, str] = {}

        for obj in scene_objects:
            obj_id = obj.get("id", obj.get("name", ""))
            category = obj.get("category", "object")
            color = obj.get("color", "")

            # Simple name
            refs[category] = obj_id

            # With color
            if color:
                refs[f"{color} {category}"] = obj_id

            # The + category
            refs[f"the {category}"] = obj_id
            if color:
                refs[f"the {color} {category}"] = obj_id

        return refs

    def _get_object_description(
        self,
        object_name: str,
        scene_objects: Optional[List[Dict[str, Any]]],
    ) -> str:
        """Get a natural language description for an object."""
        if not scene_objects:
            return object_name

        # Try to find matching object in scene
        for obj in scene_objects:
            obj_id = obj.get("id", obj.get("name", "")).lower()
            category = obj.get("category", "").lower()

            if object_name.lower() in obj_id or object_name.lower() in category:
                color = obj.get("color", "")
                if color:
                    return f"the {color} {category or object_name}"
                return f"the {category or object_name}"

        return f"the {object_name}"

    def _generate_task_instruction(
        self,
        task_type: str,
        primary_object: str,
        target_object: Optional[str],
        task_description: Optional[str],
    ) -> str:
        """Generate the main task instruction."""
        # If a description is provided, use it
        if task_description:
            return task_description

        # Get templates for this task type
        templates = self.TASK_TEMPLATES.get(task_type, self.TASK_TEMPLATES["pick_and_place"])

        # Select first template (primary)
        template = templates[0]

        # Fill in objects
        instruction = template.format(
            object=primary_object,
            target=target_object or "the target location",
            source="its current position",
        )

        return instruction

    def _generate_subtask_annotations(
        self,
        skill_segments: List[Dict[str, Any]],
        primary_object: str,
        target_object: Optional[str],
        num_frames: int,
    ) -> List[SubtaskAnnotation]:
        """Generate annotations for each subtask/skill segment."""
        annotations = []

        if not skill_segments:
            # Generate default segments based on typical manipulation phases
            if num_frames > 0:
                phase_duration = num_frames // 5
                default_phases = ["approach", "grasp", "lift", "transport", "place"]

                for i, phase in enumerate(default_phases):
                    start = i * phase_duration
                    end = min((i + 1) * phase_duration, num_frames)

                    templates = self.PHASE_TEMPLATES.get(phase, [f"Perform {phase}"])
                    instruction = templates[0].format(
                        object=primary_object,
                        target=target_object or "target",
                    )

                    annotations.append(SubtaskAnnotation(
                        phase=phase,
                        instruction=instruction,
                        start_frame=start,
                        end_frame=end,
                        objects_involved=[primary_object] if "{object}" in templates[0] else [],
                    ))
        else:
            # Use provided skill segments
            for segment in skill_segments:
                phase = segment.get("phase", segment.get("skill", "unknown"))
                start = segment.get("start_frame", segment.get("start", 0))
                end = segment.get("end_frame", segment.get("end", 0))

                templates = self.PHASE_TEMPLATES.get(phase, [f"Perform {phase}"])
                instruction = templates[0].format(
                    object=primary_object,
                    target=target_object or "target",
                )

                annotations.append(SubtaskAnnotation(
                    phase=phase,
                    instruction=instruction,
                    start_frame=start,
                    end_frame=end,
                ))

        return annotations

    def _generate_paraphrases(
        self,
        task_type: str,
        primary_object: str,
        target_object: Optional[str],
    ) -> List[str]:
        """Generate paraphrase variations of the task instruction."""
        paraphrases = []

        templates = self.TASK_TEMPLATES.get(task_type, self.TASK_TEMPLATES["pick_and_place"])

        for template in templates[1:]:  # Skip first (already used as primary)
            paraphrase = template.format(
                object=primary_object,
                target=target_object or "the target location",
                source="its current position",
            )
            paraphrases.append(paraphrase)

        # Add some additional variations
        if task_type == "pick_and_place":
            paraphrases.extend([
                f"Grab {primary_object} and move it to {target_object or 'the target'}",
                f"Get {primary_object} and set it on {target_object or 'the target'}",
            ])

        return paraphrases[:5]  # Limit to 5 paraphrases

    def _estimate_difficulty(
        self,
        task_type: str,
        scene_objects: Optional[List[Dict[str, Any]]],
        skill_segments: Optional[List[Dict[str, Any]]],
    ) -> str:
        """Estimate task difficulty based on complexity."""
        score = 0

        # Task type complexity
        if task_type in ["pick", "place"]:
            score += 1
        elif task_type in ["pick_and_place", "stack"]:
            score += 2
        elif task_type in ["pour", "push"]:
            score += 3
        else:
            score += 2

        # Number of objects
        if scene_objects:
            if len(scene_objects) > 5:
                score += 2
            elif len(scene_objects) > 2:
                score += 1

        # Number of skill segments
        if skill_segments:
            if len(skill_segments) > 7:
                score += 2
            elif len(skill_segments) > 4:
                score += 1

        if score <= 2:
            return "easy"
        elif score <= 4:
            return "medium"
        else:
            return "hard"


def annotate_episode_batch(
    episodes: List[Dict[str, Any]],
    annotator: Optional[LanguageAnnotator] = None,
) -> List[LanguageAnnotations]:
    """
    Annotate a batch of episodes.

    Args:
        episodes: List of episode metadata dictionaries
        annotator: Optional pre-configured annotator

    Returns:
        List of LanguageAnnotations for each episode
    """
    if annotator is None:
        annotator = LanguageAnnotator()

    annotations = []
    for episode in episodes:
        task_name = episode.get("task_name", episode.get("task", "manipulation"))
        task_description = episode.get("task_description")
        scene_objects = episode.get("scene_objects", episode.get("objects", []))
        skill_segments = episode.get("skill_segments", episode.get("segments", []))
        num_frames = episode.get("num_frames", episode.get("length", 0))

        annotation = annotator.annotate_episode(
            task_name=task_name,
            task_description=task_description,
            scene_objects=scene_objects,
            skill_segments=skill_segments,
            num_frames=num_frames,
        )
        annotations.append(annotation)

    return annotations


def integrate_with_lerobot_export(
    lerobot_dir: str | Path,
    *,
    overwrite: bool = False,
    annotator: Optional[LanguageAnnotator] = None,
) -> Path:
    """
    Attach per-episode natural language annotations to a LeRobot export directory.

    This writes a line-delimited JSON file at:
      <lerobot_dir>/meta/language_annotations.jsonl

    The function is intentionally lightweight and template-driven (no LLM calls
    by default) so it can run in CI/local workflows without credentials.
    """
    lerobot_dir = Path(lerobot_dir)
    meta_dir = lerobot_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    out_path = meta_dir / "language_annotations.jsonl"

    if out_path.is_file() and not overwrite:
        return out_path

    if annotator is None:
        annotator = LanguageAnnotator(use_vlm=False)

    # Prefer LeRobot v3 meta index when present; fall back to legacy root episodes.jsonl.
    episode_index_candidates = [
        meta_dir / "episodes.jsonl",
        lerobot_dir / "episodes.jsonl",
    ]
    episode_records: List[Dict[str, Any]] = []
    for candidate in episode_index_candidates:
        if not candidate.is_file():
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if isinstance(rec, dict):
                episode_records.append(rec)
        break

    if not episode_records:
        logger.warning(
            "No episode index found under %s (checked: %s); skipping language annotation export.",
            lerobot_dir,
            ", ".join(str(p) for p in episode_index_candidates),
        )
        return out_path

    with out_path.open("w", encoding="utf-8") as f:
        for rec in episode_records:
            task_name = (
                rec.get("task_instruction")
                or rec.get("task_name")
                or rec.get("task")
                or rec.get("task_id")
                or "manipulation"
            )
            task_description = rec.get("task_description") or rec.get("instruction") or rec.get("language")
            scene_objects = rec.get("scene_objects") or rec.get("objects") or []
            skill_segments = rec.get("skill_segments") or rec.get("segments") or []
            num_frames = rec.get("num_frames") or rec.get("length") or 0
            try:
                num_frames_int = int(num_frames)
            except Exception:
                num_frames_int = 0

            annotation = annotator.annotate_episode(
                task_name=str(task_name),
                task_description=str(task_description) if task_description is not None else None,
                scene_objects=scene_objects if isinstance(scene_objects, list) else [],
                skill_segments=skill_segments if isinstance(skill_segments, list) else [],
                num_frames=num_frames_int,
            )
            payload = {
                "episode_id": rec.get("episode_id") or rec.get("id") or rec.get("episode") or "",
                **annotation.to_dict(),
            }
            f.write(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")

    return out_path
