#!/usr/bin/env python3
"""
Language Annotation Generator for VLA Training.

Generates natural language instructions for robot manipulation episodes.
Required for training Vision-Language-Action models (OpenVLA, Pi0, etc.)

Features:
- Template-based generation for common tasks
- LLM-powered variation generation (Gemini)
- Multi-style annotations (imperative, descriptive, casual)
- Automatic integration with LeRobot export

Upsell Value: +$1,500 per scene
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from tools.llm_client import create_llm_client, LLMResponse
    HAVE_LLM = True
except ImportError:
    HAVE_LLM = False
    create_llm_client = None


class AnnotationStyle(str, Enum):
    """Styles of language annotations."""
    IMPERATIVE = "imperative"      # "Pick up the cup"
    DESCRIPTIVE = "descriptive"    # "The robot picks up the cup"
    CASUAL = "casual"              # "Grab that cup over there"
    DETAILED = "detailed"          # "Carefully pick up the red cup from the counter"
    MINIMAL = "minimal"            # "Pick cup"


@dataclass
class LanguageAnnotation:
    """A single language annotation for an episode."""
    text: str
    style: AnnotationStyle
    task_type: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskAnnotations:
    """All annotations for a task."""
    task_id: str
    task_type: str
    base_description: str
    annotations: List[LanguageAnnotation] = field(default_factory=list)

    def get_random(self) -> str:
        """Get a random annotation."""
        if not self.annotations:
            return self.base_description
        return random.choice(self.annotations).text

    def get_by_style(self, style: AnnotationStyle) -> List[str]:
        """Get annotations by style."""
        return [a.text for a in self.annotations if a.style == style]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "base_description": self.base_description,
            "annotations": [
                {
                    "text": a.text,
                    "style": a.style.value,
                    "confidence": a.confidence,
                }
                for a in self.annotations
            ],
        }


class LanguageTemplates:
    """Template-based language generation for manipulation tasks."""

    TEMPLATES = {
        "pick_place": {
            AnnotationStyle.IMPERATIVE: [
                "Pick up the {object} and place it on the {location}",
                "Grab the {object} and put it on the {location}",
                "Take the {object} and move it to the {location}",
                "Get the {object} and set it on the {location}",
                "Lift the {object} and position it on the {location}",
            ],
            AnnotationStyle.DESCRIPTIVE: [
                "The robot picks up the {object} and places it on the {location}",
                "Moving the {object} from its current position to the {location}",
                "Transferring the {object} to the {location}",
            ],
            AnnotationStyle.CASUAL: [
                "Move that {object} over to the {location}",
                "Put the {object} on the {location}",
                "Can you move the {object} to the {location}?",
            ],
            AnnotationStyle.DETAILED: [
                "Carefully grasp the {object}, lift it up, and gently place it on the {location}",
                "Using the gripper, pick up the {object} and transfer it to the {location}",
            ],
            AnnotationStyle.MINIMAL: [
                "Pick {object}, place {location}",
                "{object} to {location}",
            ],
        },
        "pick": {
            AnnotationStyle.IMPERATIVE: [
                "Pick up the {object}",
                "Grab the {object}",
                "Take the {object}",
                "Lift the {object}",
                "Get the {object}",
            ],
            AnnotationStyle.DESCRIPTIVE: [
                "The robot picks up the {object}",
                "Grasping the {object}",
            ],
            AnnotationStyle.CASUAL: [
                "Grab that {object}",
                "Get the {object} for me",
            ],
            AnnotationStyle.DETAILED: [
                "Carefully approach and grasp the {object}",
            ],
            AnnotationStyle.MINIMAL: [
                "Pick {object}",
                "Grasp {object}",
            ],
        },
        "place": {
            AnnotationStyle.IMPERATIVE: [
                "Place the object on the {location}",
                "Put it on the {location}",
                "Set it down on the {location}",
            ],
            AnnotationStyle.DESCRIPTIVE: [
                "Placing the object on the {location}",
            ],
            AnnotationStyle.CASUAL: [
                "Put it there on the {location}",
            ],
            AnnotationStyle.DETAILED: [
                "Carefully lower and release the object onto the {location}",
            ],
            AnnotationStyle.MINIMAL: [
                "Place on {location}",
            ],
        },
        "open_drawer": {
            AnnotationStyle.IMPERATIVE: [
                "Open the {object}",
                "Pull open the {object}",
                "Slide the {object} open",
                "Pull the {object} toward you",
            ],
            AnnotationStyle.DESCRIPTIVE: [
                "The robot opens the {object}",
                "Opening the {object} by pulling the handle",
            ],
            AnnotationStyle.CASUAL: [
                "Open that {object}",
                "Pull out the {object}",
            ],
            AnnotationStyle.DETAILED: [
                "Grasp the {object} handle and pull it open smoothly",
            ],
            AnnotationStyle.MINIMAL: [
                "Open {object}",
            ],
        },
        "close_drawer": {
            AnnotationStyle.IMPERATIVE: [
                "Close the {object}",
                "Push the {object} closed",
                "Shut the {object}",
                "Push the {object} in",
            ],
            AnnotationStyle.DESCRIPTIVE: [
                "The robot closes the {object}",
                "Closing the {object} by pushing",
            ],
            AnnotationStyle.CASUAL: [
                "Close that {object}",
                "Push it closed",
            ],
            AnnotationStyle.DETAILED: [
                "Push the {object} closed until it clicks into place",
            ],
            AnnotationStyle.MINIMAL: [
                "Close {object}",
            ],
        },
        "open_door": {
            AnnotationStyle.IMPERATIVE: [
                "Open the {object}",
                "Pull open the {object}",
                "Swing the {object} open",
            ],
            AnnotationStyle.DESCRIPTIVE: [
                "The robot opens the {object}",
            ],
            AnnotationStyle.CASUAL: [
                "Open that {object}",
            ],
            AnnotationStyle.DETAILED: [
                "Grasp the {object} handle and pull it open",
            ],
            AnnotationStyle.MINIMAL: [
                "Open {object}",
            ],
        },
        "pour": {
            AnnotationStyle.IMPERATIVE: [
                "Pour from the {object} into the {target}",
                "Empty the {object} into the {target}",
                "Transfer the contents of the {object} to the {target}",
            ],
            AnnotationStyle.DESCRIPTIVE: [
                "The robot pours from the {object} into the {target}",
                "Pouring liquid from the {object} to the {target}",
            ],
            AnnotationStyle.CASUAL: [
                "Pour that into the {target}",
                "Empty it into the {target}",
            ],
            AnnotationStyle.DETAILED: [
                "Carefully tilt the {object} and pour its contents into the {target}",
            ],
            AnnotationStyle.MINIMAL: [
                "Pour {object} to {target}",
            ],
        },
        "push": {
            AnnotationStyle.IMPERATIVE: [
                "Push the {object}",
                "Slide the {object}",
                "Move the {object} forward",
            ],
            AnnotationStyle.DESCRIPTIVE: [
                "The robot pushes the {object}",
            ],
            AnnotationStyle.CASUAL: [
                "Push that {object}",
            ],
            AnnotationStyle.DETAILED: [
                "Carefully push the {object} across the surface",
            ],
            AnnotationStyle.MINIMAL: [
                "Push {object}",
            ],
        },
        "stack": {
            AnnotationStyle.IMPERATIVE: [
                "Stack the {object} on top of the {target}",
                "Place the {object} on the {target}",
                "Put the {object} on top of the {target}",
            ],
            AnnotationStyle.DESCRIPTIVE: [
                "The robot stacks the {object} on the {target}",
            ],
            AnnotationStyle.CASUAL: [
                "Stack that on the {target}",
            ],
            AnnotationStyle.DETAILED: [
                "Carefully place the {object} on top of the {target}, aligning them",
            ],
            AnnotationStyle.MINIMAL: [
                "Stack on {target}",
            ],
        },
        "insert": {
            AnnotationStyle.IMPERATIVE: [
                "Insert the {object} into the {target}",
                "Put the {object} in the {target}",
                "Fit the {object} into the {target}",
            ],
            AnnotationStyle.DESCRIPTIVE: [
                "The robot inserts the {object} into the {target}",
                "Inserting the {object} into the {target}",
            ],
            AnnotationStyle.CASUAL: [
                "Put it in the {target}",
                "Insert that into the {target}",
            ],
            AnnotationStyle.DETAILED: [
                "Carefully align and insert the {object} into the {target}",
            ],
            AnnotationStyle.MINIMAL: [
                "Insert into {target}",
            ],
        },
    }

    # Object name variations
    OBJECT_SYNONYMS = {
        "cup": ["cup", "mug", "coffee cup", "drinking cup"],
        "mug": ["mug", "cup", "coffee mug"],
        "bottle": ["bottle", "container", "water bottle"],
        "bowl": ["bowl", "dish", "container"],
        "plate": ["plate", "dish"],
        "box": ["box", "container", "package"],
        "drawer": ["drawer", "cabinet drawer"],
        "door": ["door", "cabinet door"],
        "cabinet": ["cabinet", "cupboard"],
        "can": ["can", "tin", "container"],
        "jar": ["jar", "container"],
    }

    # Location name variations
    LOCATION_SYNONYMS = {
        "counter": ["counter", "countertop", "kitchen counter", "surface"],
        "table": ["table", "tabletop", "surface"],
        "shelf": ["shelf", "rack"],
        "sink": ["sink", "basin"],
        "stove": ["stove", "stovetop", "range"],
        "tray": ["tray", "serving tray"],
    }

    @classmethod
    def generate(
        cls,
        task_type: str,
        object_name: str = "object",
        location: str = "target",
        target: str = "target",
        num_variations: int = 10,
        styles: List[AnnotationStyle] = None,
    ) -> List[LanguageAnnotation]:
        """Generate language annotations from templates."""
        if styles is None:
            styles = list(AnnotationStyle)

        annotations = []
        templates = cls.TEMPLATES.get(task_type, cls.TEMPLATES.get("pick_place"))

        # Get object/location variations
        object_variations = cls.OBJECT_SYNONYMS.get(
            object_name.lower(), [object_name]
        )
        location_variations = cls.LOCATION_SYNONYMS.get(
            location.lower(), [location]
        )

        for style in styles:
            style_templates = templates.get(style, templates.get(AnnotationStyle.IMPERATIVE, []))

            for template in style_templates:
                # Generate with variations
                for obj_var in object_variations[:2]:
                    for loc_var in location_variations[:2]:
                        text = template.format(
                            object=obj_var,
                            location=loc_var,
                            target=target,
                        )
                        annotations.append(LanguageAnnotation(
                            text=text,
                            style=style,
                            task_type=task_type,
                            confidence=1.0,
                        ))

        # Shuffle and limit
        random.shuffle(annotations)
        return annotations[:num_variations]


class LLMLanguageGenerator:
    """LLM-powered language annotation generation."""

    GENERATION_PROMPT = """Generate {num_variations} natural language instructions for a robot manipulation task.

Task: {task_description}
Object: {object_name}
Target location: {location}

Requirements:
1. Vary the phrasing and word choice
2. Include different levels of detail (brief to detailed)
3. Use different tones (formal, casual, technical)
4. All instructions should describe the same task
5. Each instruction should be on a new line
6. Do not number the instructions

Examples of good variations:
- "Pick up the red cup and place it on the counter"
- "Grab the cup, then set it on the countertop"
- "Take the cup and put it over there on the counter"
- "Carefully grasp the cup and transfer it to the kitchen counter"

Generate {num_variations} variations:"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.client = None

        if HAVE_LLM:
            try:
                self.client = create_llm_client()
            except Exception as e:
                if self.verbose:
                    print(f"[LANG-GEN] LLM client unavailable: {e}")

    def generate(
        self,
        task_description: str,
        object_name: str = "object",
        location: str = "target",
        num_variations: int = 10,
    ) -> List[LanguageAnnotation]:
        """Generate annotations using LLM."""
        if not self.client:
            return []

        prompt = self.GENERATION_PROMPT.format(
            num_variations=num_variations,
            task_description=task_description,
            object_name=object_name,
            location=location,
        )

        try:
            response = self.client.generate(prompt, max_tokens=1000)
            text = response.text if hasattr(response, 'text') else str(response)

            # Parse response
            annotations = []
            for line in text.strip().split("\n"):
                line = line.strip()
                if line and not line.startswith("#") and len(line) > 5:
                    # Remove numbering if present
                    if line[0].isdigit() and line[1] in ".):":
                        line = line[2:].strip()
                    elif line[0].isdigit() and line[1].isdigit() and line[2] in ".):":
                        line = line[3:].strip()

                    # Remove quotes if present
                    if line.startswith('"') and line.endswith('"'):
                        line = line[1:-1]
                    if line.startswith("'") and line.endswith("'"):
                        line = line[1:-1]

                    if line:
                        annotations.append(LanguageAnnotation(
                            text=line,
                            style=AnnotationStyle.IMPERATIVE,  # LLM default
                            task_type="generated",
                            confidence=0.9,
                            metadata={"source": "llm"},
                        ))

            return annotations[:num_variations]

        except Exception as e:
            if self.verbose:
                print(f"[LANG-GEN] LLM generation failed: {e}")
            return []


class LanguageAnnotator:
    """
    Complete language annotation system for episodes.

    Combines templates + LLM for comprehensive coverage.
    """

    def __init__(
        self,
        use_llm: bool = True,
        num_variations: int = 10,
        styles: List[AnnotationStyle] = None,
        verbose: bool = True,
    ):
        self.use_llm = use_llm and HAVE_LLM
        self.num_variations = num_variations
        self.styles = styles or [
            AnnotationStyle.IMPERATIVE,
            AnnotationStyle.CASUAL,
            AnnotationStyle.DETAILED,
        ]
        self.verbose = verbose

        self.llm_generator = LLMLanguageGenerator(verbose=verbose) if self.use_llm else None

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[LANG-ANNOTATOR] {msg}")

    def annotate_task(
        self,
        task_id: str,
        task_type: str,
        task_description: str,
        object_name: str = "object",
        location: str = "target",
        target: str = "target",
    ) -> TaskAnnotations:
        """Generate all annotations for a task."""
        annotations = TaskAnnotations(
            task_id=task_id,
            task_type=task_type,
            base_description=task_description,
        )

        # Template-based annotations
        template_annotations = LanguageTemplates.generate(
            task_type=task_type,
            object_name=object_name,
            location=location,
            target=target,
            num_variations=self.num_variations,
            styles=self.styles,
        )
        annotations.annotations.extend(template_annotations)

        # LLM-based annotations (if available)
        if self.llm_generator and self.use_llm:
            llm_annotations = self.llm_generator.generate(
                task_description=task_description,
                object_name=object_name,
                location=location,
                num_variations=5,  # Fewer from LLM (more expensive)
            )
            annotations.annotations.extend(llm_annotations)

        # Deduplicate
        seen = set()
        unique_annotations = []
        for ann in annotations.annotations:
            if ann.text.lower() not in seen:
                seen.add(ann.text.lower())
                unique_annotations.append(ann)
        annotations.annotations = unique_annotations[:self.num_variations]

        self.log(f"Generated {len(annotations.annotations)} annotations for task {task_id}")
        return annotations

    def annotate_episodes(
        self,
        episodes_meta_path: Path,
        output_path: Path,
    ) -> Dict[str, List[str]]:
        """Annotate all tasks from episodes metadata."""
        # Load episodes metadata
        with open(episodes_meta_path) as f:
            tasks_data = []
            for line in f:
                if line.strip():
                    tasks_data.append(json.loads(line))

        # Generate annotations for each task
        all_annotations = {}

        for task in tasks_data:
            task_id = str(task.get("task_index", 0))
            task_type = self._infer_task_type(task.get("task", ""))
            task_desc = task.get("task", "manipulation task")

            # Extract object and location from description
            object_name, location = self._extract_objects(task_desc)

            task_annotations = self.annotate_task(
                task_id=task_id,
                task_type=task_type,
                task_description=task_desc,
                object_name=object_name,
                location=location,
            )

            all_annotations[task_id] = [a.text for a in task_annotations.annotations]

        # Save annotations
        with open(output_path, "w") as f:
            json.dump(all_annotations, f, indent=2)

        self.log(f"Saved annotations to {output_path}")
        return all_annotations

    def _infer_task_type(self, description: str) -> str:
        """Infer task type from description."""
        desc_lower = description.lower()

        task_keywords = {
            "pick_place": ["pick", "place", "put", "move", "transfer"],
            "pick": ["pick up", "grab", "take", "lift"],
            "place": ["place", "put down", "set"],
            "open_drawer": ["open drawer", "pull drawer"],
            "close_drawer": ["close drawer", "push drawer"],
            "open_door": ["open door", "open cabinet"],
            "pour": ["pour", "empty", "fill"],
            "push": ["push", "slide"],
            "stack": ["stack", "pile"],
            "insert": ["insert", "put in", "fit"],
        }

        for task_type, keywords in task_keywords.items():
            for keyword in keywords:
                if keyword in desc_lower:
                    return task_type

        return "pick_place"  # Default

    def _extract_objects(self, description: str) -> Tuple[str, str]:
        """Extract object and location from description."""
        desc_lower = description.lower()

        # Common objects
        objects = [
            "cup", "mug", "bottle", "bowl", "plate", "box",
            "can", "jar", "drawer", "door", "cabinet", "pan",
            "pot", "spoon", "fork", "knife", "glass",
        ]

        # Common locations
        locations = [
            "counter", "table", "shelf", "sink", "stove",
            "countertop", "surface", "tray", "rack",
        ]

        found_object = "object"
        found_location = "target"

        for obj in objects:
            if obj in desc_lower:
                found_object = obj
                break

        for loc in locations:
            if loc in desc_lower:
                found_location = loc
                break

        return found_object, found_location


def integrate_with_lerobot_export(
    episodes_dir: Path,
    annotations: Dict[str, List[str]],
) -> None:
    """
    Integrate language annotations into LeRobot dataset.

    Adds 'language_instruction' field to episode parquet files.
    """
    import pyarrow.parquet as pq
    import pyarrow as pa

    data_dir = episodes_dir / "data"

    for chunk_dir in data_dir.glob("chunk-*"):
        for parquet_file in chunk_dir.glob("episode_*.parquet"):
            # Read existing data
            table = pq.read_table(parquet_file)
            df = table.to_pandas()

            # Add language annotations
            task_index = df.get("task_index", [0] * len(df))

            language_instructions = []
            for idx in task_index:
                task_annotations = annotations.get(str(idx), ["perform the task"])
                # Random selection for each row
                language_instructions.append(random.choice(task_annotations))

            df["language_instruction"] = language_instructions

            # Write back
            new_table = pa.Table.from_pandas(df)
            pq.write_table(new_table, parquet_file)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate language annotations for robot manipulation episodes"
    )
    parser.add_argument(
        "--episodes-dir",
        type=Path,
        required=True,
        help="Path to LeRobot episodes directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for annotations JSON",
    )
    parser.add_argument(
        "--num-variations",
        type=int,
        default=10,
        help="Number of annotation variations per task",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        default=True,
        help="Use LLM for additional variations",
    )
    parser.add_argument(
        "--integrate",
        action="store_true",
        help="Integrate annotations into LeRobot parquet files",
    )

    args = parser.parse_args()

    output_path = args.output or args.episodes_dir / "language_annotations.json"
    tasks_path = args.episodes_dir / "meta" / "tasks.jsonl"

    annotator = LanguageAnnotator(
        use_llm=args.use_llm,
        num_variations=args.num_variations,
    )

    annotations = annotator.annotate_episodes(
        episodes_meta_path=tasks_path,
        output_path=output_path,
    )

    if args.integrate:
        print("Integrating annotations into LeRobot dataset...")
        integrate_with_lerobot_export(args.episodes_dir, annotations)
        print("Integration complete!")

    print(f"\nGenerated {sum(len(v) for v in annotations.values())} total annotations")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
