#!/usr/bin/env python3
"""
Default Language Annotations for Genie Sim 3.0 & Arena.

Previously $10,000-$25,000 upsell - NOW INCLUDED BY DEFAULT!

Generates natural language instructions for VLA training.

Features (DEFAULT - FREE):
- Template-based instruction generation
- LLM-powered variation generation (Gemini)
- Multi-style annotations (imperative, descriptive, casual, detailed)
- 10+ variations per task
- Automatic LeRobot integration

Required for: OpenVLA, Pi0, RT-2, PaLM-E training.

Output:
- language_annotations_config.json - Annotation configuration
"""

import json
from pathlib import Path
from typing import Dict


def create_default_language_annotations_exporter(
    scene_id: str,
    output_dir: Path,
) -> Dict[str, Path]:
    """Create language annotations config (DEFAULT)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "language_annotations_config.json"
    with open(config_path, "w") as f:
        json.dump({
            "enabled": True,
            "scene_id": scene_id,
            "annotation_config": {
                "num_variations_per_task": 10,
                "use_llm": True,
                "llm_provider": "gemini",
                "styles": [
                    "imperative",
                    "descriptive",
                    "casual",
                    "detailed",
                    "minimal",
                ],
                "template_based": True,
                "llm_augmentation": True,
            },
            "task_templates": {
                "pick_place": "Pick up the {object} and place it on the {location}",
                "pick": "Pick up the {object}",
                "place": "Place the object on the {location}",
                "open_drawer": "Open the {object}",
                "close_drawer": "Close the {object}",
                "pour": "Pour from the {object} into the {target}",
                "push": "Push the {object}",
                "stack": "Stack the {object} on the {target}",
            },
            "output_data": {
                "annotations_file": "language_annotations.json",
                "per_episode_annotations": "episodes/language/",
            },
            "integration": {
                "add_to_lerobot_dataset": True,
                "parquet_column": "language_instruction",
            },
            "value": "Previously $10,000-$25,000 upsell - NOW FREE BY DEFAULT",
        }, f, indent=2)

    return {"language_annotations_config": config_path}
