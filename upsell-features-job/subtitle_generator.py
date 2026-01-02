#!/usr/bin/env python3
"""
Subtitle Generator for Robot Episodes.

This module generates subtitle files (SRT, VTT, JSON) synchronized with
robot episode playback. Subtitles describe robot actions at each step.

Useful for:
- Accessible episode review
- Training video-text alignment models
- Creating captioned tutorial content
- Documentation and presentation

Supports multiple output formats:
- SRT (SubRip) - Most widely compatible
- VTT (WebVTT) - Web-native format with styling support
- JSON - Structured format for programmatic access

Usage:
    from subtitle_generator import SubtitleGenerator

    generator = SubtitleGenerator()
    result = generator.generate_episode_subtitles(
        episode_metadata=episode_meta,
        output_dir=Path("./subtitles"),
    )
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure repo root is in path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# =============================================================================
# Constants
# =============================================================================

class SubtitleFormat(str, Enum):
    """Supported subtitle formats."""
    SRT = "srt"
    VTT = "vtt"
    JSON = "json"


class CaptionStyle(str, Enum):
    """Caption styling presets."""
    MINIMAL = "minimal"  # Just actions
    DESCRIPTIVE = "descriptive"  # Full descriptions
    TECHNICAL = "technical"  # Include technical details
    INSTRUCTIONAL = "instructional"  # Tutorial-style


# Action templates for different phases
ACTION_TEMPLATES = {
    "approach": [
        "Approaching target object",
        "Moving toward {object}",
        "Positioning gripper near {object}",
    ],
    "grasp": [
        "Grasping {object}",
        "Closing gripper on {object}",
        "Securing grip on {object}",
    ],
    "lift": [
        "Lifting {object}",
        "Raising {object} from surface",
        "Elevating {object}",
    ],
    "move": [
        "Moving {object} to destination",
        "Transporting {object}",
        "Carrying {object} to target",
    ],
    "place": [
        "Placing {object}",
        "Setting down {object}",
        "Releasing {object} at destination",
    ],
    "release": [
        "Opening gripper",
        "Releasing {object}",
        "Letting go of {object}",
    ],
    "retract": [
        "Retracting arm",
        "Moving to neutral position",
        "Clearing workspace",
    ],
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SubtitleCue:
    """A single subtitle cue/caption."""
    cue_id: int
    start_time_seconds: float
    end_time_seconds: float
    text: str

    # Optional metadata
    action_type: str = ""
    object_name: str = ""
    confidence: float = 1.0

    # Styling (for VTT)
    position: Optional[str] = None  # e.g., "50%,80%"
    align: str = "center"
    style: str = ""

    def to_srt(self) -> str:
        """Convert to SRT format."""
        start = self._format_timestamp_srt(self.start_time_seconds)
        end = self._format_timestamp_srt(self.end_time_seconds)

        return f"{self.cue_id}\n{start} --> {end}\n{self.text}\n"

    def to_vtt(self) -> str:
        """Convert to VTT format."""
        start = self._format_timestamp_vtt(self.start_time_seconds)
        end = self._format_timestamp_vtt(self.end_time_seconds)

        # Build cue with optional styling
        cue_lines = []
        cue_lines.append(f"{start} --> {end}")

        if self.position or self.align != "center":
            settings = []
            if self.position:
                settings.append(f"position:{self.position}")
            if self.align:
                settings.append(f"align:{self.align}")
            cue_lines[0] += " " + " ".join(settings)

        cue_lines.append(self.text)

        return "\n".join(cue_lines) + "\n"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cue_id": self.cue_id,
            "start_time": self.start_time_seconds,
            "end_time": self.end_time_seconds,
            "duration": self.end_time_seconds - self.start_time_seconds,
            "text": self.text,
            "action_type": self.action_type,
            "object_name": self.object_name,
            "confidence": self.confidence,
        }

    def _format_timestamp_srt(self, seconds: float) -> str:
        """Format timestamp for SRT (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def _format_timestamp_vtt(self, seconds: float) -> str:
        """Format timestamp for VTT (HH:MM:SS.mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


@dataclass
class EpisodeSubtitles:
    """Complete subtitles for an episode."""
    episode_id: str
    scene_id: str
    task_description: str

    # Cues
    cues: List[SubtitleCue] = field(default_factory=list)

    # Output paths
    srt_path: Optional[str] = None
    vtt_path: Optional[str] = None
    json_path: Optional[str] = None

    # Episode timing
    duration_seconds: float = 0.0
    fps: float = 30.0

    # Generation metadata
    style: str = "descriptive"
    language: str = "en"
    generated_at: Optional[datetime] = None

    def to_srt(self) -> str:
        """Generate complete SRT file content."""
        return "\n".join(cue.to_srt() for cue in self.cues)

    def to_vtt(self) -> str:
        """Generate complete VTT file content."""
        lines = ["WEBVTT", "Kind: captions", f"Language: {self.language}", ""]

        # Add metadata header
        lines.append(f"NOTE Episode: {self.episode_id}")
        lines.append(f"NOTE Scene: {self.scene_id}")
        lines.append(f"NOTE Task: {self.task_description}")
        lines.append("")

        for cue in self.cues:
            lines.append(cue.to_vtt())

        return "\n".join(lines)

    def to_json(self) -> Dict[str, Any]:
        """Generate JSON structure."""
        return {
            "episode_id": self.episode_id,
            "scene_id": self.scene_id,
            "task_description": self.task_description,
            "duration_seconds": self.duration_seconds,
            "fps": self.fps,
            "style": self.style,
            "language": self.language,
            "cue_count": len(self.cues),
            "cues": [cue.to_dict() for cue in self.cues],
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
        }

    def to_dict(self) -> Dict[str, Any]:
        return self.to_json()


@dataclass
class SubtitleResult:
    """Result of subtitle generation."""
    scene_id: str
    success: bool
    episodes_subtitled: int = 0
    total_cues: int = 0
    output_dir: str = ""
    manifest_path: str = ""
    formats_generated: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# =============================================================================
# Subtitle Generator
# =============================================================================

class SubtitleGenerator:
    """
    Generates subtitles for robot episodes.

    Creates synchronized captions describing robot actions
    based on episode metadata and skill segments.
    """

    def __init__(
        self,
        style: str = "descriptive",
        language: str = "en",
        output_formats: Optional[List[str]] = None,
        min_cue_duration: float = 0.5,
        max_cue_duration: float = 5.0,
        verbose: bool = True,
    ):
        self.verbose = verbose
        self.style = CaptionStyle(style)
        self.language = language
        self.output_formats = [SubtitleFormat(f) for f in (output_formats or ["srt", "vtt", "json"])]
        self.min_cue_duration = min_cue_duration
        self.max_cue_duration = max_cue_duration

    def log(self, msg: str, level: str = "INFO") -> None:
        if self.verbose:
            print(f"[SUBTITLE-GEN] [{level}] {msg}")

    def generate_episode_subtitles(
        self,
        episode_metadata: Dict[str, Any],
        output_dir: Path,
        episode_id: Optional[str] = None,
    ) -> EpisodeSubtitles:
        """
        Generate subtitles for a single episode.

        Args:
            episode_metadata: Episode metadata with task info, skill segments
            output_dir: Directory to save subtitle files
            episode_id: Optional episode ID

        Returns:
            EpisodeSubtitles with file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get episode identifiers
        episode_id = episode_id or episode_metadata.get("episode_id", "episode_0")
        scene_id = episode_metadata.get("scene_id", "unknown_scene")
        task_description = episode_metadata.get("task", "manipulation task")
        episode_length = episode_metadata.get("length", 100)
        fps = episode_metadata.get("fps", 30.0)

        duration = episode_length / fps

        self.log(f"Generating subtitles for {episode_id} ({duration:.1f}s)")

        # Generate cues
        cues = self._generate_cues(episode_metadata, duration)

        # Create subtitles object
        subtitles = EpisodeSubtitles(
            episode_id=episode_id,
            scene_id=scene_id,
            task_description=task_description,
            cues=cues,
            duration_seconds=duration,
            fps=fps,
            style=self.style.value,
            language=self.language,
            generated_at=datetime.now(timezone.utc),
        )

        # Write output files
        for fmt in self.output_formats:
            output_path = output_dir / f"{episode_id}.{fmt.value}"

            try:
                if fmt == SubtitleFormat.SRT:
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(subtitles.to_srt())
                    subtitles.srt_path = str(output_path)

                elif fmt == SubtitleFormat.VTT:
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(subtitles.to_vtt())
                    subtitles.vtt_path = str(output_path)

                elif fmt == SubtitleFormat.JSON:
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(subtitles.to_json(), f, indent=2)
                    subtitles.json_path = str(output_path)

            except Exception as e:
                self.log(f"Failed to write {fmt.value}: {e}", "ERROR")

        return subtitles

    def _generate_cues(
        self,
        episode_metadata: Dict[str, Any],
        total_duration: float,
    ) -> List[SubtitleCue]:
        """Generate subtitle cues from episode metadata."""
        cues = []
        cue_id = 1

        task_description = episode_metadata.get("task", "manipulation task")
        fps = episode_metadata.get("fps", 30.0)

        # Extract object name from task description
        object_name = self._extract_object_name(task_description)

        # Check for skill segments
        skill_segments = episode_metadata.get("skill_segments", [])

        if skill_segments:
            # Generate cues from skill segments
            for segment in skill_segments:
                segment_name = segment.get("name", "action")
                description = segment.get("description", "")
                start_frame = segment.get("start_frame", 0)
                end_frame = segment.get("end_frame", 100)

                start_time = start_frame / fps
                end_time = end_frame / fps

                # Clamp to valid range
                start_time = max(0, min(start_time, total_duration))
                end_time = max(start_time + self.min_cue_duration, min(end_time, total_duration))

                # Generate caption text
                caption = self._generate_caption(segment_name, object_name, description)

                cues.append(SubtitleCue(
                    cue_id=cue_id,
                    start_time_seconds=start_time,
                    end_time_seconds=end_time,
                    text=caption,
                    action_type=segment_name,
                    object_name=object_name,
                ))
                cue_id += 1
        else:
            # Generate default phase-based cues
            phases = self._get_default_phases(total_duration)

            for phase_name, start_time, end_time in phases:
                caption = self._generate_caption(phase_name, object_name, "")

                cues.append(SubtitleCue(
                    cue_id=cue_id,
                    start_time_seconds=start_time,
                    end_time_seconds=end_time,
                    text=caption,
                    action_type=phase_name,
                    object_name=object_name,
                ))
                cue_id += 1

        # Add intro cue
        if cues and cues[0].start_time_seconds > 0.5:
            intro_cue = SubtitleCue(
                cue_id=0,
                start_time_seconds=0.0,
                end_time_seconds=min(2.0, cues[0].start_time_seconds - 0.1),
                text=f"Task: {task_description}",
                action_type="intro",
            )
            cues.insert(0, intro_cue)
            # Renumber cues
            for i, cue in enumerate(cues):
                cue.cue_id = i + 1

        return cues

    def _extract_object_name(self, task_description: str) -> str:
        """Extract object name from task description."""
        # Common patterns
        keywords = ["pick up", "grasp", "grab", "lift", "move", "place", "put"]

        task_lower = task_description.lower()

        for keyword in keywords:
            if keyword in task_lower:
                # Get text after keyword
                idx = task_lower.find(keyword) + len(keyword)
                remaining = task_description[idx:].strip()

                # Get first word or phrase
                if remaining:
                    # Remove articles
                    for article in ["the ", "a ", "an "]:
                        if remaining.lower().startswith(article):
                            remaining = remaining[len(article):]

                    # Get first few words
                    words = remaining.split()[:3]
                    return " ".join(words).rstrip(",.;")

        return "object"

    def _generate_caption(
        self,
        action_type: str,
        object_name: str,
        description: str,
    ) -> str:
        """Generate caption text based on style."""
        action_type = action_type.lower().replace("_", " ")

        # Get template if available
        templates = ACTION_TEMPLATES.get(action_type.split()[0], None)

        if description and self.style in [CaptionStyle.DESCRIPTIVE, CaptionStyle.INSTRUCTIONAL]:
            return description

        if templates:
            # Use first template (could randomize for variety)
            template = templates[0]
            caption = template.format(object=object_name)
        else:
            # Generic caption
            caption = f"{action_type.title()}"
            if object_name != "object":
                caption += f" {object_name}"

        if self.style == CaptionStyle.TECHNICAL:
            caption = f"[{action_type.upper()}] {caption}"

        return caption

    def _get_default_phases(
        self,
        total_duration: float,
    ) -> List[Tuple[str, float, float]]:
        """Get default action phases for an episode."""
        phases = [
            ("approach", 0.0, 0.20),
            ("grasp", 0.20, 0.35),
            ("lift", 0.35, 0.50),
            ("move", 0.50, 0.75),
            ("place", 0.75, 0.90),
            ("retract", 0.90, 1.00),
        ]

        return [
            (name, start * total_duration, end * total_duration)
            for name, start, end in phases
        ]

    def generate_scene_subtitles(
        self,
        episodes_dir: Path,
        output_dir: Path,
        max_episodes: int = -1,
    ) -> SubtitleResult:
        """
        Generate subtitles for all episodes in a scene.

        Args:
            episodes_dir: Directory containing episode data
            output_dir: Directory to save subtitle files
            max_episodes: Maximum episodes to subtitle (-1 for all)

        Returns:
            SubtitleResult with summary
        """
        episodes_dir = Path(episodes_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        scene_id = episodes_dir.parent.name
        self.log(f"Generating subtitles for scene {scene_id}")

        # Find episode metadata
        episodes_meta_path = episodes_dir / "meta" / "episodes.jsonl"
        tasks_path = episodes_dir / "meta" / "tasks.jsonl"

        episodes = []
        errors = []

        # Load episodes metadata
        if episodes_meta_path.exists():
            with open(episodes_meta_path) as f:
                for line in f:
                    try:
                        episodes.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if max_episodes > 0:
            episodes = episodes[:max_episodes]

        # Load tasks
        tasks = {}
        if tasks_path.exists():
            with open(tasks_path) as f:
                for line in f:
                    try:
                        task = json.loads(line)
                        tasks[task.get("task_index", 0)] = task.get("task", "")
                    except json.JSONDecodeError:
                        continue

        # Generate subtitles
        all_subtitles = []
        total_cues = 0

        for ep in episodes:
            episode_id = ep.get("episode_index", 0)
            task_idx = ep.get("task_index", 0)

            # Build metadata
            metadata = {
                "episode_id": f"episode_{episode_id:06d}",
                "scene_id": scene_id,
                "task": tasks.get(task_idx, "manipulation task"),
                "length": ep.get("length", 100),
                "fps": ep.get("fps", 30.0),
                "skill_segments": ep.get("skill_segments", []),
            }

            try:
                subtitles = self.generate_episode_subtitles(
                    metadata,
                    output_dir / f"episode_{episode_id:06d}",
                )
                all_subtitles.append(subtitles)
                total_cues += len(subtitles.cues)
            except Exception as e:
                errors.append(f"Episode {episode_id}: {str(e)}")
                self.log(f"Failed to subtitle episode {episode_id}: {e}", "ERROR")

        # Write manifest
        manifest = {
            "scene_id": scene_id,
            "episodes_subtitled": len(all_subtitles),
            "total_cues": total_cues,
            "formats": [f.value for f in self.output_formats],
            "style": self.style.value,
            "language": self.language,
            "generated_at": datetime.now(timezone.utc).isoformat() + "Z",
            "episodes": [s.to_dict() for s in all_subtitles],
        }

        manifest_path = output_dir / "subtitle_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        return SubtitleResult(
            scene_id=scene_id,
            success=len(errors) == 0,
            episodes_subtitled=len(all_subtitles),
            total_cues=total_cues,
            output_dir=str(output_dir),
            manifest_path=str(manifest_path),
            formats_generated=[f.value for f in self.output_formats],
            errors=errors,
        )


# =============================================================================
# Helper Functions
# =============================================================================

def generate_subtitles(
    episodes_dir: Path,
    output_dir: Path,
    style: str = "descriptive",
    formats: Optional[List[str]] = None,
    max_episodes: int = -1,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function for generating subtitles.

    Args:
        episodes_dir: Directory containing episode data
        output_dir: Directory to save subtitle files
        style: Caption style (minimal, descriptive, technical, instructional)
        formats: Output formats (srt, vtt, json)
        max_episodes: Maximum episodes to subtitle
        verbose: Enable verbose output

    Returns:
        Result dictionary
    """
    generator = SubtitleGenerator(
        style=style,
        output_formats=formats,
        verbose=verbose,
    )

    result = generator.generate_scene_subtitles(
        episodes_dir=episodes_dir,
        output_dir=output_dir,
        max_episodes=max_episodes,
    )

    return {
        "success": result.success,
        "scene_id": result.scene_id,
        "episodes_subtitled": result.episodes_subtitled,
        "total_cues": result.total_cues,
        "output_dir": result.output_dir,
        "manifest_path": result.manifest_path,
        "formats_generated": result.formats_generated,
        "errors": result.errors,
    }


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate subtitles for robot episodes"
    )

    parser.add_argument(
        "--episodes-dir",
        type=Path,
        required=True,
        help="Directory containing episode data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save subtitle files",
    )
    parser.add_argument(
        "--style",
        choices=["minimal", "descriptive", "technical", "instructional"],
        default="descriptive",
        help="Caption style",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["srt", "vtt", "json"],
        default=["srt", "vtt", "json"],
        help="Output formats",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language code",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=-1,
        help="Maximum episodes to subtitle (-1 for all)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )

    args = parser.parse_args()

    generator = SubtitleGenerator(
        style=args.style,
        language=args.language,
        output_formats=args.formats,
        verbose=not args.quiet,
    )

    result = generator.generate_scene_subtitles(
        episodes_dir=args.episodes_dir,
        output_dir=args.output_dir,
        max_episodes=args.max_episodes,
    )

    print("\n" + "=" * 60)
    print("SUBTITLE GENERATION COMPLETE")
    print("=" * 60)
    print(f"Scene: {result.scene_id}")
    print(f"Episodes subtitled: {result.episodes_subtitled}")
    print(f"Total cues: {result.total_cues}")
    print(f"Formats: {', '.join(result.formats_generated)}")
    print(f"Output: {result.output_dir}")

    if result.errors:
        print("\nErrors:")
        for error in result.errors:
            print(f"  - {error}")

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
