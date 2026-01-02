#!/usr/bin/env python3
"""
Audio Narration Generator for Episodes.

This module generates text-to-speech audio narrations for robot episodes.
Audio narrations describe what the robot is doing at each step, useful for:
- Training VLA models with audio modality
- Creating accessible review content
- Tutorial generation for customers

Supports multiple TTS backends:
1. Google Cloud Text-to-Speech (production)
2. Local TTS via pyttsx3 (fallback)

Usage:
    from audio_narrator import AudioNarrator

    narrator = AudioNarrator(tts_provider="google")
    result = narrator.generate_episode_narration(
        episode_metadata=episode_meta,
        output_dir=Path("./audio"),
    )
"""

from __future__ import annotations

import io
import json
import os
import sys
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure repo root is in path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Try to import Google Cloud TTS
try:
    from google.cloud import texttospeech
    HAVE_GOOGLE_TTS = True
except ImportError:
    HAVE_GOOGLE_TTS = False
    texttospeech = None

# Try to import local TTS (pyttsx3)
try:
    import pyttsx3
    HAVE_LOCAL_TTS = True
except ImportError:
    HAVE_LOCAL_TTS = False
    pyttsx3 = None

# Try to import audio processing libraries
try:
    from pydub import AudioSegment
    HAVE_PYDUB = True
except ImportError:
    HAVE_PYDUB = False
    AudioSegment = None


# =============================================================================
# Constants
# =============================================================================

class TTSProvider(str, Enum):
    """Available TTS providers."""
    GOOGLE = "google"
    LOCAL = "local"
    MOCK = "mock"


class VoiceGender(str, Enum):
    """Voice gender options."""
    MALE = "MALE"
    FEMALE = "FEMALE"
    NEUTRAL = "NEUTRAL"


class AudioFormat(str, Enum):
    """Output audio formats."""
    MP3 = "mp3"
    WAV = "wav"
    OGG = "ogg"


# Voice presets for different use cases
VOICE_PRESETS = {
    "narrator": {
        "google": {
            "language_code": "en-US",
            "name": "en-US-Neural2-D",
            "gender": VoiceGender.MALE,
        },
        "description": "Clear, professional male narrator voice",
    },
    "instructor": {
        "google": {
            "language_code": "en-US",
            "name": "en-US-Neural2-F",
            "gender": VoiceGender.FEMALE,
        },
        "description": "Instructional female voice",
    },
    "robot": {
        "google": {
            "language_code": "en-US",
            "name": "en-US-Standard-B",
            "gender": VoiceGender.MALE,
        },
        "description": "Slightly robotic/synthetic voice",
    },
}

# Speaking rate and pitch defaults
DEFAULT_SPEAKING_RATE = 1.0
DEFAULT_PITCH = 0.0


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class NarrationSegment:
    """A single narration segment."""
    segment_id: str
    text: str
    start_time_seconds: float
    end_time_seconds: float

    # Audio file path (after generation)
    audio_path: Optional[str] = None

    # Metadata
    word_count: int = 0
    character_count: int = 0
    estimated_duration_seconds: float = 0.0

    def __post_init__(self):
        self.word_count = len(self.text.split())
        self.character_count = len(self.text)
        # Estimate ~150 words per minute
        self.estimated_duration_seconds = self.word_count / 2.5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodeNarration:
    """Complete narration for an episode."""
    episode_id: str
    scene_id: str
    task_description: str

    # Segments
    segments: List[NarrationSegment] = field(default_factory=list)

    # Combined audio
    combined_audio_path: Optional[str] = None

    # Generation metadata
    voice_preset: str = "narrator"
    language_code: str = "en-US"
    tts_provider: str = "google"

    # Statistics
    total_duration_seconds: float = 0.0
    total_word_count: int = 0

    # Timestamps
    generated_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["segments"] = [s.to_dict() for s in self.segments]
        if self.generated_at:
            data["generated_at"] = self.generated_at.isoformat()
        return data


@dataclass
class NarrationResult:
    """Result of narration generation."""
    scene_id: str
    success: bool
    episodes_narrated: int = 0
    total_audio_duration_seconds: float = 0.0
    output_dir: str = ""
    manifest_path: str = ""
    errors: List[str] = field(default_factory=list)


# =============================================================================
# Narration Script Generator
# =============================================================================

class NarrationScriptGenerator:
    """
    Generates narration scripts from episode metadata.

    Creates timestamped narration text that describes what the robot
    is doing at each stage of the episode.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[NARRATION-SCRIPT] {msg}")

    def generate_script(
        self,
        episode_metadata: Dict[str, Any],
        include_timestamps: bool = True,
    ) -> List[NarrationSegment]:
        """
        Generate narration script from episode metadata.

        Args:
            episode_metadata: Episode metadata with task info, keypoints, etc.
            include_timestamps: Whether to include timing information

        Returns:
            List of NarrationSegment objects
        """
        segments = []

        task_description = episode_metadata.get("task", "manipulation task")
        episode_length = episode_metadata.get("length", 100)
        fps = episode_metadata.get("fps", 30.0)

        total_duration = episode_length / fps

        # 1. Introduction segment
        intro_text = self._generate_intro(task_description)
        segments.append(NarrationSegment(
            segment_id="intro",
            text=intro_text,
            start_time_seconds=0.0,
            end_time_seconds=min(3.0, total_duration * 0.1),
        ))

        # 2. Get skill segments if available
        skill_segments = episode_metadata.get("skill_segments", [])

        if skill_segments:
            # Generate narration for each skill segment
            for i, skill in enumerate(skill_segments):
                skill_name = skill.get("name", f"step_{i+1}")
                skill_description = skill.get("description", "")
                start_frame = skill.get("start_frame", 0)
                end_frame = skill.get("end_frame", episode_length)

                start_time = start_frame / fps
                end_time = end_frame / fps

                narration_text = self._generate_skill_narration(
                    skill_name, skill_description, i + 1, len(skill_segments)
                )

                segments.append(NarrationSegment(
                    segment_id=f"skill_{i}",
                    text=narration_text,
                    start_time_seconds=start_time,
                    end_time_seconds=end_time,
                ))
        else:
            # Generate generic segments based on episode phases
            segments.extend(self._generate_phase_segments(
                task_description, total_duration
            ))

        # 3. Conclusion segment
        conclusion_text = self._generate_conclusion(task_description)
        segments.append(NarrationSegment(
            segment_id="conclusion",
            text=conclusion_text,
            start_time_seconds=total_duration - 2.0,
            end_time_seconds=total_duration,
        ))

        return segments

    def _generate_intro(self, task_description: str) -> str:
        """Generate introduction narration."""
        templates = [
            f"The robot is about to {task_description.lower()}.",
            f"In this episode, the robot will {task_description.lower()}.",
            f"Watch as the robot performs the task: {task_description.lower()}.",
            f"Demonstrating: {task_description}.",
        ]
        # Use task hash to select consistent template
        idx = hash(task_description) % len(templates)
        return templates[idx]

    def _generate_skill_narration(
        self,
        skill_name: str,
        description: str,
        step_num: int,
        total_steps: int,
    ) -> str:
        """Generate narration for a skill segment."""
        # Clean up skill name
        skill_name = skill_name.replace("_", " ").title()

        if description:
            return f"Step {step_num} of {total_steps}: {skill_name}. {description}"
        else:
            return f"Step {step_num} of {total_steps}: {skill_name}."

    def _generate_phase_segments(
        self,
        task_description: str,
        total_duration: float,
    ) -> List[NarrationSegment]:
        """Generate generic phase-based narration segments."""
        phases = [
            ("approach", "The robot approaches the target object.", 0.1, 0.25),
            ("grasp", "The gripper closes to grasp the object securely.", 0.25, 0.4),
            ("lift", "The robot lifts the object carefully.", 0.4, 0.55),
            ("move", "The robot moves the object to the target location.", 0.55, 0.75),
            ("place", "The robot places the object at the destination.", 0.75, 0.9),
        ]

        segments = []
        for phase_id, text, start_ratio, end_ratio in phases:
            segments.append(NarrationSegment(
                segment_id=phase_id,
                text=text,
                start_time_seconds=total_duration * start_ratio,
                end_time_seconds=total_duration * end_ratio,
            ))

        return segments

    def _generate_conclusion(self, task_description: str) -> str:
        """Generate conclusion narration."""
        return "Task completed successfully."


# =============================================================================
# Audio Narrator
# =============================================================================

class AudioNarrator:
    """
    Generates audio narrations for robot episodes.

    Uses text-to-speech to convert narration scripts into audio files,
    with support for multiple TTS backends and voice options.
    """

    def __init__(
        self,
        tts_provider: str = "google",
        voice_preset: str = "narrator",
        language_code: str = "en-US",
        speaking_rate: float = DEFAULT_SPEAKING_RATE,
        pitch: float = DEFAULT_PITCH,
        output_format: str = "mp3",
        verbose: bool = True,
    ):
        self.verbose = verbose
        self.voice_preset = voice_preset
        self.language_code = language_code
        self.speaking_rate = speaking_rate
        self.pitch = pitch
        self.output_format = AudioFormat(output_format)

        # Determine TTS provider
        self.tts_provider = TTSProvider(tts_provider)
        self._init_tts_client()

        # Script generator
        self.script_generator = NarrationScriptGenerator(verbose=verbose)

    def log(self, msg: str, level: str = "INFO") -> None:
        if self.verbose:
            print(f"[AUDIO-NARRATOR] [{level}] {msg}")

    def _init_tts_client(self) -> None:
        """Initialize TTS client based on provider."""
        self.tts_client = None

        if self.tts_provider == TTSProvider.GOOGLE:
            if HAVE_GOOGLE_TTS:
                try:
                    self.tts_client = texttospeech.TextToSpeechClient()
                    self.log("Google Cloud TTS initialized")
                except Exception as e:
                    self.log(f"Failed to init Google TTS: {e}", "WARNING")
                    self.tts_provider = TTSProvider.MOCK
            else:
                self.log("Google Cloud TTS not available, using mock", "WARNING")
                self.tts_provider = TTSProvider.MOCK

        elif self.tts_provider == TTSProvider.LOCAL:
            if HAVE_LOCAL_TTS:
                try:
                    self.tts_client = pyttsx3.init()
                    self.log("Local TTS (pyttsx3) initialized")
                except Exception as e:
                    self.log(f"Failed to init local TTS: {e}", "WARNING")
                    self.tts_provider = TTSProvider.MOCK
            else:
                self.log("pyttsx3 not available, using mock", "WARNING")
                self.tts_provider = TTSProvider.MOCK

    def generate_episode_narration(
        self,
        episode_metadata: Dict[str, Any],
        output_dir: Path,
        episode_id: Optional[str] = None,
    ) -> EpisodeNarration:
        """
        Generate complete narration for an episode.

        Args:
            episode_metadata: Episode metadata with task info
            output_dir: Directory to save audio files
            episode_id: Optional episode ID (derived from metadata if not provided)

        Returns:
            EpisodeNarration with audio file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get episode identifiers
        episode_id = episode_id or episode_metadata.get("episode_id", "episode_0")
        scene_id = episode_metadata.get("scene_id", "unknown_scene")
        task_description = episode_metadata.get("task", "manipulation task")

        self.log(f"Generating narration for episode {episode_id}")

        # Generate script
        segments = self.script_generator.generate_script(episode_metadata)

        # Generate audio for each segment
        total_duration = 0.0
        total_words = 0

        for segment in segments:
            audio_path = output_dir / f"{episode_id}_{segment.segment_id}.{self.output_format.value}"

            try:
                duration = self._synthesize_speech(segment.text, audio_path)
                segment.audio_path = str(audio_path)
                total_duration += duration
                total_words += segment.word_count
            except Exception as e:
                self.log(f"Failed to generate audio for {segment.segment_id}: {e}", "ERROR")

        # Combine audio segments (if pydub available)
        combined_path = None
        if HAVE_PYDUB and len(segments) > 1:
            combined_path = output_dir / f"{episode_id}_narration.{self.output_format.value}"
            try:
                self._combine_audio_segments(segments, combined_path)
            except Exception as e:
                self.log(f"Failed to combine audio: {e}", "WARNING")
                combined_path = None

        narration = EpisodeNarration(
            episode_id=episode_id,
            scene_id=scene_id,
            task_description=task_description,
            segments=segments,
            combined_audio_path=str(combined_path) if combined_path else None,
            voice_preset=self.voice_preset,
            language_code=self.language_code,
            tts_provider=self.tts_provider.value,
            total_duration_seconds=total_duration,
            total_word_count=total_words,
            generated_at=datetime.now(timezone.utc),
        )

        return narration

    def _synthesize_speech(self, text: str, output_path: Path) -> float:
        """
        Synthesize speech from text.

        Returns:
            Audio duration in seconds
        """
        if self.tts_provider == TTSProvider.GOOGLE:
            return self._synthesize_google(text, output_path)
        elif self.tts_provider == TTSProvider.LOCAL:
            return self._synthesize_local(text, output_path)
        else:
            return self._synthesize_mock(text, output_path)

    def _synthesize_google(self, text: str, output_path: Path) -> float:
        """Synthesize using Google Cloud TTS."""
        if not self.tts_client:
            raise RuntimeError("Google TTS client not initialized")

        # Get voice settings from preset
        preset = VOICE_PRESETS.get(self.voice_preset, VOICE_PRESETS["narrator"])
        google_config = preset.get("google", {})

        # Build request
        synthesis_input = texttospeech.SynthesisInput(text=text)

        voice = texttospeech.VoiceSelectionParams(
            language_code=google_config.get("language_code", self.language_code),
            name=google_config.get("name", "en-US-Neural2-D"),
        )

        # Audio config
        if self.output_format == AudioFormat.MP3:
            audio_encoding = texttospeech.AudioEncoding.MP3
        elif self.output_format == AudioFormat.WAV:
            audio_encoding = texttospeech.AudioEncoding.LINEAR16
        else:
            audio_encoding = texttospeech.AudioEncoding.OGG_OPUS

        audio_config = texttospeech.AudioConfig(
            audio_encoding=audio_encoding,
            speaking_rate=self.speaking_rate,
            pitch=self.pitch,
        )

        # Make request
        response = self.tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config,
        )

        # Write audio file
        with open(output_path, "wb") as f:
            f.write(response.audio_content)

        # Estimate duration from word count (~150 words/minute adjusted by rate)
        word_count = len(text.split())
        duration = (word_count / 2.5) / self.speaking_rate

        return duration

    def _synthesize_local(self, text: str, output_path: Path) -> float:
        """Synthesize using local TTS (pyttsx3)."""
        if not self.tts_client:
            raise RuntimeError("Local TTS not initialized")

        # pyttsx3 can save to file
        self.tts_client.save_to_file(text, str(output_path))
        self.tts_client.runAndWait()

        # Estimate duration
        word_count = len(text.split())
        duration = word_count / 2.5

        return duration

    def _synthesize_mock(self, text: str, output_path: Path) -> float:
        """Create mock audio file (for testing)."""
        # Create a placeholder file with metadata
        mock_data = {
            "mock": True,
            "text": text,
            "word_count": len(text.split()),
            "voice_preset": self.voice_preset,
        }

        # Write as JSON with audio extension (for testing)
        with open(output_path, "w") as f:
            json.dump(mock_data, f, indent=2)

        # Estimate duration
        word_count = len(text.split())
        duration = word_count / 2.5

        return duration

    def _combine_audio_segments(
        self,
        segments: List[NarrationSegment],
        output_path: Path,
    ) -> None:
        """Combine audio segments into a single file."""
        if not HAVE_PYDUB:
            raise RuntimeError("pydub not available for audio combining")

        combined = AudioSegment.empty()

        for segment in segments:
            if segment.audio_path and Path(segment.audio_path).exists():
                try:
                    audio = AudioSegment.from_file(segment.audio_path)
                    combined += audio
                except Exception as e:
                    self.log(f"Failed to load segment {segment.segment_id}: {e}", "WARNING")

        # Export combined audio
        export_format = self.output_format.value
        if export_format == "mp3":
            combined.export(output_path, format="mp3")
        elif export_format == "wav":
            combined.export(output_path, format="wav")
        else:
            combined.export(output_path, format="ogg")

    def generate_scene_narrations(
        self,
        episodes_dir: Path,
        output_dir: Path,
        max_episodes: int = -1,
    ) -> NarrationResult:
        """
        Generate narrations for all episodes in a scene.

        Args:
            episodes_dir: Directory containing episode data
            output_dir: Directory to save audio files
            max_episodes: Maximum episodes to narrate (-1 for all)

        Returns:
            NarrationResult with summary
        """
        episodes_dir = Path(episodes_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        scene_id = episodes_dir.parent.name
        self.log(f"Generating narrations for scene {scene_id}")

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

        # Load tasks for descriptions
        tasks = {}
        if tasks_path.exists():
            with open(tasks_path) as f:
                for line in f:
                    try:
                        task = json.loads(line)
                        tasks[task.get("task_index", 0)] = task.get("task", "")
                    except json.JSONDecodeError:
                        continue

        # Generate narrations
        narrations = []
        total_duration = 0.0

        for ep in episodes:
            episode_id = ep.get("episode_index", 0)
            task_idx = ep.get("task_index", 0)

            # Build metadata for narration
            metadata = {
                "episode_id": f"episode_{episode_id:06d}",
                "scene_id": scene_id,
                "task": tasks.get(task_idx, "manipulation task"),
                "length": ep.get("length", 100),
                "fps": ep.get("fps", 30.0),
                "skill_segments": ep.get("skill_segments", []),
            }

            try:
                narration = self.generate_episode_narration(
                    metadata,
                    output_dir / f"episode_{episode_id:06d}",
                )
                narrations.append(narration)
                total_duration += narration.total_duration_seconds
            except Exception as e:
                errors.append(f"Episode {episode_id}: {str(e)}")
                self.log(f"Failed to narrate episode {episode_id}: {e}", "ERROR")

        # Write manifest
        manifest = {
            "scene_id": scene_id,
            "episodes_narrated": len(narrations),
            "total_duration_seconds": total_duration,
            "voice_preset": self.voice_preset,
            "tts_provider": self.tts_provider.value,
            "generated_at": datetime.now(timezone.utc).isoformat() + "Z",
            "narrations": [n.to_dict() for n in narrations],
        }

        manifest_path = output_dir / "narration_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        return NarrationResult(
            scene_id=scene_id,
            success=len(errors) == 0,
            episodes_narrated=len(narrations),
            total_audio_duration_seconds=total_duration,
            output_dir=str(output_dir),
            manifest_path=str(manifest_path),
            errors=errors,
        )


# =============================================================================
# Helper Functions
# =============================================================================

def generate_audio_narration(
    episodes_dir: Path,
    output_dir: Path,
    voice_preset: str = "narrator",
    max_episodes: int = -1,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function for generating audio narrations.

    Args:
        episodes_dir: Directory containing episode data
        output_dir: Directory to save audio files
        voice_preset: Voice preset to use
        max_episodes: Maximum episodes to narrate
        verbose: Enable verbose output

    Returns:
        Result dictionary
    """
    narrator = AudioNarrator(
        voice_preset=voice_preset,
        verbose=verbose,
    )

    result = narrator.generate_scene_narrations(
        episodes_dir=episodes_dir,
        output_dir=output_dir,
        max_episodes=max_episodes,
    )

    return {
        "success": result.success,
        "scene_id": result.scene_id,
        "episodes_narrated": result.episodes_narrated,
        "total_duration_seconds": result.total_audio_duration_seconds,
        "output_dir": result.output_dir,
        "manifest_path": result.manifest_path,
        "errors": result.errors,
    }


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate audio narrations for robot episodes"
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
        help="Directory to save audio files",
    )
    parser.add_argument(
        "--voice-preset",
        choices=list(VOICE_PRESETS.keys()),
        default="narrator",
        help="Voice preset to use",
    )
    parser.add_argument(
        "--tts-provider",
        choices=["google", "local", "mock"],
        default="google",
        help="TTS provider",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=-1,
        help="Maximum episodes to narrate (-1 for all)",
    )
    parser.add_argument(
        "--speaking-rate",
        type=float,
        default=1.0,
        help="Speaking rate (0.5-2.0)",
    )
    parser.add_argument(
        "--format",
        choices=["mp3", "wav", "ogg"],
        default="mp3",
        help="Output audio format",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )

    args = parser.parse_args()

    narrator = AudioNarrator(
        tts_provider=args.tts_provider,
        voice_preset=args.voice_preset,
        speaking_rate=args.speaking_rate,
        output_format=args.format,
        verbose=not args.quiet,
    )

    result = narrator.generate_scene_narrations(
        episodes_dir=args.episodes_dir,
        output_dir=args.output_dir,
        max_episodes=args.max_episodes,
    )

    print("\n" + "=" * 60)
    print("AUDIO NARRATION COMPLETE")
    print("=" * 60)
    print(f"Scene: {result.scene_id}")
    print(f"Episodes narrated: {result.episodes_narrated}")
    print(f"Total duration: {result.total_audio_duration_seconds:.1f}s")
    print(f"Output: {result.output_dir}")

    if result.errors:
        print("\nErrors:")
        for error in result.errors:
            print(f"  - {error}")

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
