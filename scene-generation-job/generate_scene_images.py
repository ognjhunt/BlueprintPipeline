#!/usr/bin/env python3
"""
Scene Image Generator for BlueprintPipeline.

This job generates realistic scene images for different environment archetypes
using Gemini 3.0 Pro Image. Images are designed to be wide-angle shots that
capture the ENTIRE scene with no blind spots, suitable for 3D reconstruction.

Pipeline flow:
1. Query Firestore for generation history to ensure diversity
2. Use Gemini 3.0 Pro Preview to generate diverse prompts based on archetype
3. Generate scene images using Gemini 3.0 Pro Image (Nano Banana Pro)
4. Upload images to GCS to trigger downstream pipeline
5. Record generation metadata in Firestore for future diversity

Scheduled to run daily at 8:00 AM to create 10 new scenes.

Environment Variables:
    GEMINI_API_KEY: API key for Gemini models
    BUCKET: GCS bucket for scene storage
    SCENES_PER_RUN: Number of scenes to generate (default: 10)
    DRY_RUN: If "true", skip actual generation (for testing)
    FIRESTORE_PROJECT: GCP project for Firestore (optional, uses default)
"""

import json
import os
import sys
import time
import uuid
import datetime
import hashlib
import io
import base64
import random
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("[SCENE-GEN] ERROR: google-genai package not installed", file=sys.stderr)
    sys.exit(1)

try:
    from google.cloud import firestore
    from google.cloud import storage
    HAVE_CLOUD_DEPS = True
except ImportError:
    HAVE_CLOUD_DEPS = False
    firestore = None
    storage = None

from PIL import Image

from tools.gcs_upload import upload_blob_from_filename
from tools.metrics.pipeline_metrics import get_metrics
from tools.source_asset_checksums import write_source_checksums


# ============================================================================
# Constants
# ============================================================================

GCS_ROOT = Path("/mnt/gcs")
LOGGER = logging.getLogger("scene-generation-job")

# Gemini models
GEMINI_PRO_MODEL = "gemini-3-pro-preview"  # For prompt diversification
GEMINI_IMAGE_MODEL = "gemini-3-pro-image-preview"  # For image generation (Nano Banana Pro)

# Generation settings
DEFAULT_SCENES_PER_RUN = 10
DEFAULT_ASPECT_RATIO = "16:9"
DEFAULT_IMAGE_SIZE = "2K"
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5

# Firestore collection names
FIRESTORE_COLLECTION_HISTORY = "scene_generation_history"
FIRESTORE_COLLECTION_PROMPTS = "scene_generation_prompts"
JOB_NAME = "scene-generation-job"


# ============================================================================
# Error Handling
# ============================================================================

class SceneGenerationErrorCode(str, Enum):
    """Unique error codes for scene generation failures."""
    CONFIG_MISSING_API_KEY = "SCENEGEN-0001"
    FIRESTORE_INIT_FAILED = "SCENEGEN-1001"
    FIRESTORE_QUERY_FAILED = "SCENEGEN-1002"
    FIRESTORE_COVERAGE_FAILED = "SCENEGEN-1003"
    FIRESTORE_COUNTS_FAILED = "SCENEGEN-1004"
    FIRESTORE_WRITE_FAILED = "SCENEGEN-1005"
    PROMPT_DIVERSIFICATION_FAILED = "SCENEGEN-2001"
    GCS_CLIENT_UNAVAILABLE = "SCENEGEN-3001"
    GCS_UPLOAD_FAILED = "SCENEGEN-3002"
    PIPELINE_TRIGGER_FAILED = "SCENEGEN-3003"
    IMAGE_GENERATION_FAILED = "SCENEGEN-4001"
    CHECKSUM_WRITE_FAILED = "SCENEGEN-4002"
    UNKNOWN_ARCHETYPE = "SCENEGEN-5001"
    BATCH_FAILED = "SCENEGEN-9001"


@dataclass
class SceneGenerationIssue:
    """Structured error/warning payload for scene generation."""
    code: SceneGenerationErrorCode
    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    fatal: bool = False


class SceneGenerationJobError(RuntimeError):
    """Fatal error for scene-generation job."""
    def __init__(
        self,
        code: SceneGenerationErrorCode,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.context = context or {}


class SceneGenerationBatchError(SceneGenerationJobError):
    """Batch error that includes results and summary."""
    def __init__(
        self,
        message: str,
        results: List["SceneGenerationResult"],
        summary: Dict[str, Any],
        issues: List[SceneGenerationIssue]
    ) -> None:
        super().__init__(SceneGenerationErrorCode.BATCH_FAILED, message, {"failed": summary.get("failed", 0)})
        self.results = results
        self.summary = summary
        self.issues = issues


def _log_issue(level: int, issue: SceneGenerationIssue, exc: Optional[BaseException] = None) -> None:
    """Log issue with error code and context, emit metrics for errors."""
    if exc:
        LOGGER.log(
            level,
            "%s | code=%s | context=%s",
            issue.message,
            issue.code.value,
            issue.context,
            exc_info=True,
        )
    else:
        LOGGER.log(
            level,
            "%s | code=%s | context=%s",
            issue.message,
            issue.code.value,
            issue.context,
        )

    if level >= logging.ERROR:
        try:
            metrics = get_metrics()
            labels = {"job": JOB_NAME, "error_type": issue.code.value}
            scene_id = issue.context.get("scene_id")
            if scene_id:
                labels["scene_id"] = scene_id
            metrics.errors_total.inc(labels=labels)
        except Exception:
            LOGGER.debug("Metrics emission failed for %s.", issue.code.value, exc_info=True)


# ============================================================================
# Environment Archetypes
# ============================================================================

class EnvironmentArchetype(str, Enum):
    """Supported environment archetypes matching the marketplace offerings."""
    KITCHEN = "kitchen"
    GROCERY = "grocery"
    WAREHOUSE = "warehouse"
    LOADING_DOCK = "loading_dock"
    LAB = "lab"
    OFFICE = "office"
    UTILITY_ROOM = "utility_room"
    HOME_LAUNDRY = "home_laundry"


# Archetype descriptions for prompt generation
ARCHETYPE_DESCRIPTIONS = {
    EnvironmentArchetype.KITCHEN: {
        "name": "Commercial Kitchen / Prep Line",
        "description": "Commercial food preparation environments including prep lines, dish pits, quick-serve stations, and service pass-throughs",
        "key_elements": [
            "stainless steel prep tables", "commercial dishwashers", "sinks with spray nozzles",
            "refrigerators and freezers", "ovens and ranges", "warming stations",
            "utensil racks", "cutting boards", "storage shelves", "ventilation hoods",
            "floor drains", "hand washing stations", "condiment stations"
        ],
        "articulated_objects": [
            "dishwasher doors and racks", "refrigerator doors", "oven doors",
            "cabinet doors", "drawers", "warming lamp arms"
        ],
        "lighting_conditions": ["bright fluorescent", "under-cabinet task lighting", "pass-through window lighting"],
        "floor_types": ["tile with grout lines", "commercial rubber mats", "sealed concrete"],
        "policy_focus": ["dish loading/unloading", "food prep manipulation", "appliance operation"]
    },
    EnvironmentArchetype.GROCERY: {
        "name": "Grocery / Retail Aisle",
        "description": "Grocery store and retail environments with stocked shelving, refrigeration, and checkout areas",
        "key_elements": [
            "gondola shelving units", "refrigerated display cases", "freezer sections",
            "shopping carts and baskets", "price tags and labels", "barcode scanners",
            "checkout counters", "conveyor belts", "produce displays", "endcap promotions"
        ],
        "articulated_objects": [
            "refrigerator doors", "freezer doors", "hinged shelf labels",
            "checkout dividers", "shopping cart handles"
        ],
        "lighting_conditions": ["bright overhead fluorescent", "refrigerator case lighting", "accent spotlights"],
        "floor_types": ["polished concrete", "vinyl tile", "rubber anti-fatigue mats"],
        "policy_focus": ["product stocking", "shelf organization", "item retrieval"]
    },
    EnvironmentArchetype.WAREHOUSE: {
        "name": "Warehouse / Fulfillment Center",
        "description": "Industrial warehouse environments with racking systems, pallets, and logistics infrastructure",
        "key_elements": [
            "pallet racking systems", "tote bins and containers", "conveyor systems",
            "forklifts and pallet jacks", "shipping labels", "barcode systems",
            "loading zones", "safety barriers", "floor markings", "staging areas"
        ],
        "articulated_objects": [
            "roll-up doors", "conveyor gates", "adjustable shelf brackets",
            "pallet jack handles", "safety gate latches"
        ],
        "lighting_conditions": ["high bay LED", "natural skylights", "motion-activated zones"],
        "floor_types": ["sealed concrete with markings", "epoxy coating", "loading dock plates"],
        "policy_focus": ["pallet handling", "tote picking", "package sorting"]
    },
    EnvironmentArchetype.LOADING_DOCK: {
        "name": "Loading Dock / Shipping Bay",
        "description": "Industrial loading dock environments with dock levelers, restraints, and staging areas",
        "key_elements": [
            "dock levelers and plates", "dock bumpers", "truck restraints",
            "overhead doors", "dock lights", "pallet positions", "staging pallets",
            "dock seals", "safety bollards", "height clearance signs"
        ],
        "articulated_objects": [
            "overhead doors", "dock leveler plates", "truck restraint hooks",
            "dock light arms", "liftgate mechanisms"
        ],
        "lighting_conditions": ["outdoor daylight transition", "overhead dock lights", "trailer interior"],
        "floor_types": ["diamond plate steel", "sealed concrete", "dock bumper rubber"],
        "policy_focus": ["truck loading/unloading", "pallet staging", "dock equipment operation"]
    },
    EnvironmentArchetype.LAB: {
        "name": "Laboratory / Cleanroom",
        "description": "Scientific laboratory environments with precision equipment, safety features, and controlled conditions",
        "key_elements": [
            "lab benches with sinks", "fume hoods", "gloveboxes", "biosafety cabinets",
            "centrifuges", "microscopes", "sample racks", "pipette holders",
            "chemical storage cabinets", "emergency eyewash stations", "gas lines"
        ],
        "articulated_objects": [
            "fume hood sashes", "glovebox ports", "cabinet doors",
            "centrifuge lids", "drawer pulls", "gas valve handles"
        ],
        "lighting_conditions": ["bright even overhead", "task lighting", "UV-filtered zones"],
        "floor_types": ["epoxy resin", "static-dissipative tile", "vinyl sheet"],
        "policy_focus": ["precision sample handling", "equipment operation", "safety procedures"]
    },
    EnvironmentArchetype.OFFICE: {
        "name": "Office / Workspace",
        "description": "Office environments with desks, storage, and collaborative spaces",
        "key_elements": [
            "desks with monitors", "office chairs", "filing cabinets", "bookcases",
            "conference tables", "whiteboards", "printers and copiers", "phone systems",
            "cable management", "task lighting", "plants and decor"
        ],
        "articulated_objects": [
            "desk drawers", "filing cabinet drawers", "door handles",
            "adjustable monitor arms", "cabinet doors", "chair mechanisms"
        ],
        "lighting_conditions": ["overhead panels", "natural window light", "task lamps"],
        "floor_types": ["commercial carpet", "vinyl plank", "raised floor tiles"],
        "policy_focus": ["drawer manipulation", "document handling", "equipment interaction"]
    },
    EnvironmentArchetype.UTILITY_ROOM: {
        "name": "Utility Room / Mechanical Space",
        "description": "Back-of-house utility spaces with electrical panels, HVAC, and building systems",
        "key_elements": [
            "electrical panels and breakers", "HVAC units", "water heaters",
            "pipe runs and valves", "conduit and wiring", "meters and gauges",
            "safety signage", "fire suppression", "access panels", "tool storage"
        ],
        "articulated_objects": [
            "breaker switches", "valve handles", "access panel doors",
            "circuit breaker toggles", "damper controls", "service disconnects"
        ],
        "lighting_conditions": ["utility fluorescent", "emergency lighting", "task flashlight"],
        "floor_types": ["bare concrete", "rubber utility mats", "raised grating"],
        "policy_focus": ["panel interaction", "valve operation", "inspection routines"]
    },
    EnvironmentArchetype.HOME_LAUNDRY: {
        "name": "Home Laundry Room",
        "description": "Residential laundry environments with washers, dryers, and folding areas",
        "key_elements": [
            "front-load washer", "front-load dryer", "stacked units", "laundry sink",
            "folding table or counter", "hanging rod", "laundry baskets and hampers",
            "detergent storage", "ironing board", "drying rack", "lint traps"
        ],
        "articulated_objects": [
            "washer door", "dryer door", "detergent dispenser",
            "cabinet doors", "drawer pulls", "ironing board legs"
        ],
        "lighting_conditions": ["residential overhead", "natural window", "under-cabinet"],
        "floor_types": ["tile", "vinyl", "concrete with paint"],
        "policy_focus": ["laundry sorting", "garment folding", "appliance loading"]
    }
}

# Distribution weights for balanced scene generation
ARCHETYPE_WEIGHTS = {
    EnvironmentArchetype.KITCHEN: 0.18,      # High demand
    EnvironmentArchetype.GROCERY: 0.15,
    EnvironmentArchetype.WAREHOUSE: 0.17,    # High demand
    EnvironmentArchetype.LOADING_DOCK: 0.08,
    EnvironmentArchetype.LAB: 0.12,
    EnvironmentArchetype.OFFICE: 0.12,
    EnvironmentArchetype.UTILITY_ROOM: 0.08,
    EnvironmentArchetype.HOME_LAUNDRY: 0.10,
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class SceneGenerationRequest:
    """Request for generating a single scene image."""
    scene_id: str
    archetype: EnvironmentArchetype
    prompt: str
    variation_hints: List[str] = field(default_factory=list)
    diversity_seed: Optional[str] = None


@dataclass
class SceneGenerationResult:
    """Result of generating a single scene image."""
    scene_id: str
    archetype: EnvironmentArchetype
    success: bool
    image_path: Optional[str] = None
    gcs_uri: Optional[str] = None
    prompt_used: str = ""
    generation_time_seconds: float = 0.0
    error: Optional[str] = None
    error_code: Optional[str] = None
    error_context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationHistoryEntry:
    """Entry in the generation history for tracking diversity."""
    scene_id: str
    archetype: str
    prompt_hash: str
    prompt_summary: str  # First 200 chars
    variation_tags: List[str]
    generated_at: datetime.datetime  # UTC timestamp
    success: bool


@dataclass
class FirestoreWriteError:
    """Structured error for Firestore write failures."""
    message: str
    exception_type: str
    details: Optional[str] = None


@dataclass
class FirestoreWriteResult:
    """Structured result for Firestore write attempts."""
    success: bool
    operation: str
    documents_written: int = 0
    errors: List[FirestoreWriteError] = field(default_factory=list)


# ============================================================================
# Firestore History Tracking
# ============================================================================

class GenerationHistoryTracker:
    """Tracks generation history in Firestore for diversity."""

    def __init__(self, project_id: Optional[str] = None):
        """Initialize Firestore client."""
        self.enabled = HAVE_CLOUD_DEPS and firestore is not None
        self.db = None

        if self.enabled:
            try:
                if project_id:
                    self.db = firestore.Client(project=project_id)
                else:
                    self.db = firestore.Client()
                print("[SCENE-GEN] Firestore history tracking enabled")
            except Exception as e:
                issue = SceneGenerationIssue(
                    code=SceneGenerationErrorCode.FIRESTORE_INIT_FAILED,
                    message="Firestore unavailable; disabling history tracking.",
                    context={"project_id": project_id or "default"},
                )
                _log_issue(logging.WARNING, issue, exc=e)
                self.enabled = False

    def get_recent_prompts(
        self,
        archetype: EnvironmentArchetype,
        days: int = 30,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get recent prompts for an archetype to avoid repetition."""
        if not self.enabled or not self.db:
            return []

        try:
            cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)

            query = (
                self.db.collection(FIRESTORE_COLLECTION_HISTORY)
                .where("archetype", "==", archetype.value)
                .where("generated_at", ">=", cutoff)
                .order_by("generated_at", direction=firestore.Query.DESCENDING)
                .limit(limit)
            )

            docs = query.stream()
            return [doc.to_dict() for doc in docs]

        except Exception as e:
            issue = SceneGenerationIssue(
                code=SceneGenerationErrorCode.FIRESTORE_QUERY_FAILED,
                message="Failed to query Firestore history; proceeding without history.",
                context={"archetype": archetype.value, "days": days, "limit": limit},
            )
            _log_issue(logging.WARNING, issue, exc=e)
            return []

    def get_variation_coverage(
        self,
        archetype: EnvironmentArchetype,
        days: int = 30
    ) -> Dict[str, int]:
        """Get coverage counts for variation tags."""
        if not self.enabled or not self.db:
            return {}

        try:
            recent = self.get_recent_prompts(archetype, days)
            coverage = {}

            for entry in recent:
                for tag in entry.get("variation_tags", []):
                    coverage[tag] = coverage.get(tag, 0) + 1

            return coverage

        except Exception as e:
            issue = SceneGenerationIssue(
                code=SceneGenerationErrorCode.FIRESTORE_COVERAGE_FAILED,
                message="Failed to compute variation coverage; proceeding without coverage.",
                context={"archetype": archetype.value, "days": days},
            )
            _log_issue(logging.WARNING, issue, exc=e)
            return {}

    def _build_history_payload(self, entry: GenerationHistoryEntry) -> Dict[str, Any]:
        data = asdict(entry)
        # Always store a Firestore timestamp to avoid string ordering issues
        data["generated_at"] = firestore.SERVER_TIMESTAMP if firestore is not None else entry.generated_at
        return data

    def record_generations(self, entries: List[GenerationHistoryEntry]) -> FirestoreWriteResult:
        """Record one or more generations in history."""
        if not entries:
            return FirestoreWriteResult(success=True, operation="noop", documents_written=0)
        if not self.enabled or not self.db:
            return FirestoreWriteResult(
                success=False,
                operation="history_write",
                documents_written=0,
                errors=[
                    FirestoreWriteError(
                        message="Firestore history tracking is disabled or unavailable.",
                        exception_type="FirestoreUnavailable",
                    )
                ],
            )

        try:
            if len(entries) == 1:
                entry = entries[0]
                doc_ref = self.db.collection(FIRESTORE_COLLECTION_HISTORY).document(entry.scene_id)
                doc_ref.set(self._build_history_payload(entry))
                return FirestoreWriteResult(
                    success=True,
                    operation="history_set",
                    documents_written=1,
                )

            batch = self.db.batch()
            for entry in entries:
                doc_ref = self.db.collection(FIRESTORE_COLLECTION_HISTORY).document(entry.scene_id)
                batch.set(doc_ref, self._build_history_payload(entry))
            batch.commit()
            return FirestoreWriteResult(
                success=True,
                operation="history_batch_set",
                documents_written=len(entries),
            )

        except Exception as e:
            issue = SceneGenerationIssue(
                code=SceneGenerationErrorCode.FIRESTORE_WRITE_FAILED,
                message="Failed to record history entries in Firestore.",
                context={"documents": len(entries)},
                fatal=False,
            )
            _log_issue(logging.ERROR, issue, exc=e)
            return FirestoreWriteResult(
                success=False,
                operation="history_write",
                documents_written=0,
                errors=[
                    FirestoreWriteError(
                        message=str(e),
                        exception_type=type(e).__name__,
                        details=repr(e),
                    )
                ],
            )

    def record_generation(self, entry: GenerationHistoryEntry) -> FirestoreWriteResult:
        """Record a generation in history."""
        return self.record_generations([entry])

    def get_archetype_counts(self, days: int = 7) -> Dict[str, int]:
        """Get generation counts per archetype for load balancing."""
        if not self.enabled or not self.db:
            return {}

        try:
            cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)

            counts = {}
            for archetype in EnvironmentArchetype:
                query = (
                    self.db.collection(FIRESTORE_COLLECTION_HISTORY)
                    .where("archetype", "==", archetype.value)
                    .where("generated_at", ">=", cutoff)
                    .where("success", "==", True)
                )
                # Count documents
                docs = list(query.stream())
                counts[archetype.value] = len(docs)

            return counts

        except Exception as e:
            issue = SceneGenerationIssue(
                code=SceneGenerationErrorCode.FIRESTORE_COUNTS_FAILED,
                message="Failed to get archetype counts; using default weights.",
                context={"days": days},
            )
            _log_issue(logging.WARNING, issue, exc=e)
            return {}


# ============================================================================
# Prompt Diversification Engine
# ============================================================================

class PromptDiversifier:
    """Uses Gemini 3.0 Pro Preview to generate diverse scene prompts."""

    def __init__(self, client):
        """Initialize with Gemini client."""
        self.client = client

    def generate_diverse_prompt(
        self,
        archetype: EnvironmentArchetype,
        recent_prompts: List[Dict[str, Any]],
        variation_coverage: Dict[str, int],
        target_variations: Optional[List[str]] = None
    ) -> Tuple[str, List[str]]:
        """
        Generate a diverse prompt for scene image generation.

        Uses Gemini Pro to analyze recent generations and create a prompt
        that explores new variations while staying true to the archetype.

        Returns:
            Tuple of (prompt_text, variation_tags)
        """
        archetype_info = ARCHETYPE_DESCRIPTIONS[archetype]

        # Build context about recent generations
        recent_summaries = []
        for entry in recent_prompts[:20]:
            summary = entry.get("prompt_summary", "")[:100]
            tags = entry.get("variation_tags", [])
            recent_summaries.append(f"- {summary}... (tags: {', '.join(tags[:3])})")

        recent_context = "\n".join(recent_summaries) if recent_summaries else "No recent generations."

        # Build coverage context
        coverage_context = ""
        if variation_coverage:
            low_coverage = [k for k, v in variation_coverage.items() if v < 3]
            high_coverage = [k for k, v in variation_coverage.items() if v >= 5]
            coverage_context = f"""
Variation Coverage Analysis:
- Under-represented (generate more of these): {', '.join(low_coverage[:10]) if low_coverage else 'None'}
- Well-covered (avoid unless necessary): {', '.join(high_coverage[:10]) if high_coverage else 'None'}
"""

        # Target specific variations if requested
        target_context = ""
        if target_variations:
            target_context = f"\nSpecifically try to include: {', '.join(target_variations)}"

        diversification_prompt = f"""You are an expert in generating prompts for AI image generation of photorealistic indoor environments for robotics simulation.

## Task
Generate a unique, detailed prompt for creating a photorealistic image of a {archetype_info['name']} environment. The image must be suitable for 3D reconstruction and robotics training.

## Environment Type: {archetype.value}
{archetype_info['description']}

## Key Elements to Consider
{json.dumps(archetype_info['key_elements'], indent=2)}

## Articulated Objects (important for robotics)
{json.dumps(archetype_info['articulated_objects'], indent=2)}

## Lighting Conditions
{json.dumps(archetype_info['lighting_conditions'], indent=2)}

## Floor Types
{json.dumps(archetype_info['floor_types'], indent=2)}

## Policy Focus Areas
{json.dumps(archetype_info['policy_focus'], indent=2)}

## Recent Generations (avoid similarity)
{recent_context}

{coverage_context}
{target_context}

## Requirements for the Generated Prompt

1. **Wide-angle coverage**: The prompt must describe a scene that captures the ENTIRE environment with no blind spots. Think of a photographer doing a real estate shoot.

2. **Photorealistic quality**: Describe the scene as if photographing a real location with a professional camera.

3. **Consistent lighting**: Specify realistic lighting conditions appropriate for the environment.

4. **Rich detail**: Include specific objects, materials, textures, and environmental details.

5. **Diversity**: Generate something DIFFERENT from recent generations. Vary:
   - Layout and arrangement
   - Time of day / lighting mood
   - Occupancy state (busy vs. quiet, clean vs. in-use)
   - Specific equipment/furniture variants
   - Camera angle and perspective

6. **Robotics relevance**: Ensure the scene includes objects and configurations relevant to robotic manipulation tasks.

## Output Format

Return a JSON object with exactly these fields:
{{
    "prompt": "The complete image generation prompt (2-4 paragraphs, very detailed)",
    "variation_tags": ["tag1", "tag2", "tag3", "tag4", "tag5"],
    "scene_summary": "One sentence summary of what makes this scene unique"
}}

The variation_tags should capture the key unique aspects of this scene for tracking diversity.
Examples: "morning_light", "busy_state", "modern_equipment", "cluttered", "high_angle", "compact_space"

Return ONLY the JSON, no additional text."""

        try:
            response = self.client.models.generate_content(
                model=GEMINI_PRO_MODEL,
                contents=[diversification_prompt],
                config=types.GenerateContentConfig(
                    temperature=0.9,  # Higher for diversity
                    max_output_tokens=2000,
                    response_mime_type="application/json",
                ),
            )

            response_text = response.text.strip()

            # Clean up response
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]

            result = json.loads(response_text)

            prompt = result.get("prompt", "")
            tags = result.get("variation_tags", [])

            # Enhance prompt with standard requirements
            enhanced_prompt = self._enhance_prompt_for_generation(prompt, archetype)

            return enhanced_prompt, tags

        except Exception as e:
            issue = SceneGenerationIssue(
                code=SceneGenerationErrorCode.PROMPT_DIVERSIFICATION_FAILED,
                message="Prompt diversification failed; using fallback prompt.",
                context={"archetype": archetype.value},
                fatal=False,
            )
            _log_issue(logging.WARNING, issue, exc=e)
            # Fall back to template-based generation
            return self._fallback_prompt(archetype), ["fallback", archetype.value]

    def _enhance_prompt_for_generation(self, base_prompt: str, archetype: EnvironmentArchetype) -> str:
        """Add standard requirements to the base prompt for optimal image generation."""
        archetype_info = ARCHETYPE_DESCRIPTIONS[archetype]

        enhancement = f"""

Technical Requirements:
- Ultra-wide angle lens perspective (equivalent to 16-24mm full frame)
- Capture the ENTIRE room/space from corner to corner with no blind spots
- Professional architectural photography style
- Sharp focus throughout the entire scene (deep depth of field)
- Photorealistic quality with accurate materials, textures, and lighting
- Natural perspective without excessive distortion
- High dynamic range to capture both bright and shadow areas
- 8K resolution detail level
- No people visible in the scene
- Realistic wear and use patterns on surfaces and objects
- Accurate scale and proportions for all objects

Camera Position:
- Elevated position (approximately 1.5-2 meters height)
- Positioned to maximize visible floor area and wall coverage
- Slight downward angle to show floor and surfaces clearly
- Centered to avoid extreme perspective distortion
"""

        return base_prompt + enhancement

    def _fallback_prompt(self, archetype: EnvironmentArchetype) -> str:
        """Generate a fallback prompt when diversification fails."""
        info = ARCHETYPE_DESCRIPTIONS[archetype]

        elements = random.sample(info['key_elements'], min(6, len(info['key_elements'])))
        lighting = random.choice(info['lighting_conditions'])
        floor = random.choice(info['floor_types'])

        prompt = f"""A photorealistic wide-angle photograph of a {info['name']} environment.

The space features {', '.join(elements)}. The lighting consists of {lighting}, creating realistic shadows and highlights across the scene. The floor is {floor}, showing natural wear patterns from regular use.

The camera captures the entire room from an elevated corner position, ensuring complete coverage of all surfaces, equipment, and architectural features. Every detail is sharp and clearly visible, suitable for professional documentation.

The scene appears to be in active use with realistic clutter and arrangement, not a staged showroom. All equipment and furniture shows appropriate age and condition for a working environment."""

        return self._enhance_prompt_for_generation(prompt, archetype)


# ============================================================================
# Image Generation
# ============================================================================

class SceneImageGenerator:
    """Generates scene images using Gemini 3.0 Pro Image."""

    def __init__(self, client, output_dir: Path, gcs_bucket: Optional[str] = None):
        """Initialize generator."""
        self.client = client
        self.output_dir = output_dir
        self.gcs_bucket = gcs_bucket
        self.storage_client = None

        if gcs_bucket:
            if not HAVE_CLOUD_DEPS or storage is None:
                issue = SceneGenerationIssue(
                    code=SceneGenerationErrorCode.GCS_CLIENT_UNAVAILABLE,
                    message="GCS dependencies unavailable while bucket configured.",
                    context={"bucket": gcs_bucket},
                    fatal=True,
                )
                _log_issue(logging.ERROR, issue)
                raise SceneGenerationJobError(
                    code=issue.code,
                    message=issue.message,
                    context=issue.context,
                )

            try:
                self.storage_client = storage.Client()
            except Exception as e:
                issue = SceneGenerationIssue(
                    code=SceneGenerationErrorCode.GCS_CLIENT_UNAVAILABLE,
                    message="Failed to initialize GCS storage client.",
                    context={"bucket": gcs_bucket},
                    fatal=True,
                )
                _log_issue(logging.ERROR, issue, exc=e)
                raise SceneGenerationJobError(
                    code=issue.code,
                    message=issue.message,
                    context=issue.context,
                ) from e

    def generate_scene_image(
        self,
        request: SceneGenerationRequest,
        dry_run: bool = False
    ) -> SceneGenerationResult:
        """
        Generate a single scene image.

        Uses Gemini 3.0 Pro Image with:
        - Grounding with Google Search enabled
        - 2K resolution output
        - 16:9 aspect ratio
        """
        start_time = time.time()

        # Create output directory for this scene
        scene_dir = self.output_dir / request.scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)

        image_path = scene_dir / "source_image.png"

        if dry_run:
            print(f"[SCENE-GEN] [DRY-RUN] Would generate scene: {request.scene_id}")
            print(f"[SCENE-GEN] [DRY-RUN] Archetype: {request.archetype.value}")
            print(f"[SCENE-GEN] [DRY-RUN] Prompt preview: {request.prompt[:200]}...")

            return SceneGenerationResult(
                scene_id=request.scene_id,
                archetype=request.archetype,
                success=True,
                image_path=str(image_path),
                prompt_used=request.prompt,
                generation_time_seconds=0.0,
                metadata={"dry_run": True}
            )

        print(f"[SCENE-GEN] Generating scene: {request.scene_id}")
        print(f"[SCENE-GEN]   Archetype: {request.archetype.value}")
        print(f"[SCENE-GEN]   Model: {GEMINI_IMAGE_MODEL}")

        # Retry loop for robustness
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                # Build content with prompt
                contents = [
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(text=request.prompt),
                        ],
                    ),
                ]

                # Enable Google Search grounding
                tools = [
                    types.Tool(googleSearch=types.GoogleSearch()),
                ]

                # Configure image generation
                generate_config = types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                    image_config=types.ImageConfig(
                        aspect_ratio=DEFAULT_ASPECT_RATIO,
                        image_size=DEFAULT_IMAGE_SIZE,
                    ),
                    tools=tools,
                )

                # Generate with streaming
                image_data = None
                for chunk in self.client.models.generate_content_stream(
                    model=GEMINI_IMAGE_MODEL,
                    contents=contents,
                    config=generate_config,
                ):
                    if (
                        chunk.candidates is None
                        or chunk.candidates[0].content is None
                        or chunk.candidates[0].content.parts is None
                    ):
                        continue

                    part = chunk.candidates[0].content.parts[0]
                    if hasattr(part, 'inline_data') and part.inline_data and part.inline_data.data:
                        image_data = part.inline_data.data
                        break

                if not image_data:
                    raise ValueError("No image data in Gemini response")

                # Save image
                if isinstance(image_data, str):
                    image_bytes = base64.b64decode(image_data)
                else:
                    image_bytes = image_data

                img = Image.open(io.BytesIO(image_bytes))
                img.save(str(image_path), format='PNG')

                checksum_path = scene_dir / "source_checksums.json"
                try:
                    write_source_checksums(checksum_path, scene_dir, [image_path])
                except Exception as e:
                    issue = SceneGenerationIssue(
                        code=SceneGenerationErrorCode.CHECKSUM_WRITE_FAILED,
                        message="Failed to write source asset checksum file.",
                        context={"scene_id": request.scene_id, "path": str(checksum_path)},
                        fatal=True,
                    )
                    _log_issue(logging.ERROR, issue, exc=e)
                    elapsed = time.time() - start_time
                    return SceneGenerationResult(
                        scene_id=request.scene_id,
                        archetype=request.archetype,
                        success=False,
                        image_path=str(image_path),
                        prompt_used=request.prompt,
                        generation_time_seconds=elapsed,
                        error=issue.message,
                        error_code=issue.code.value,
                        error_context=issue.context,
                    )

                # Upload to GCS if configured
                gcs_uri = None
                if self.gcs_bucket and self.storage_client:
                    gcs_uri, upload_issue = self._upload_to_gcs(
                        request.scene_id,
                        image_path,
                        checksum_path,
                    )
                    if upload_issue:
                        _log_issue(logging.ERROR, upload_issue)
                        elapsed = time.time() - start_time
                        return SceneGenerationResult(
                            scene_id=request.scene_id,
                            archetype=request.archetype,
                            success=False,
                            image_path=str(image_path),
                            prompt_used=request.prompt,
                            generation_time_seconds=elapsed,
                            error=upload_issue.message,
                            error_code=upload_issue.code.value,
                            error_context=upload_issue.context,
                        )

                elapsed = time.time() - start_time
                print(f"[SCENE-GEN] Generated: {request.scene_id} ({elapsed:.1f}s)")

                return SceneGenerationResult(
                    scene_id=request.scene_id,
                    archetype=request.archetype,
                    success=True,
                    image_path=str(image_path),
                    gcs_uri=gcs_uri,
                    prompt_used=request.prompt,
                    generation_time_seconds=elapsed,
                    metadata={
                        "variation_hints": request.variation_hints,
                        "image_size": DEFAULT_IMAGE_SIZE,
                        "aspect_ratio": DEFAULT_ASPECT_RATIO,
                    }
                )

            except Exception as e:
                last_error = str(e)
                issue = SceneGenerationIssue(
                    code=SceneGenerationErrorCode.IMAGE_GENERATION_FAILED,
                    message="Scene image generation attempt failed.",
                    context={
                        "scene_id": request.scene_id,
                        "archetype": request.archetype.value,
                        "attempt": attempt + 1,
                        "max_retries": MAX_RETRIES,
                    },
                    fatal=False,
                )
                _log_issue(logging.WARNING, issue, exc=e)
                try:
                    metrics = get_metrics()
                    metrics.retries_total.inc(labels={"job": JOB_NAME, "scene_id": request.scene_id})
                except Exception:
                    LOGGER.debug("Retry metrics emission failed.", exc_info=True)

                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY_SECONDS * (attempt + 1))

        # All retries failed
        elapsed = time.time() - start_time
        return SceneGenerationResult(
            scene_id=request.scene_id,
            archetype=request.archetype,
            success=False,
            prompt_used=request.prompt,
            generation_time_seconds=elapsed,
            error=last_error,
            error_code=SceneGenerationErrorCode.IMAGE_GENERATION_FAILED.value,
            error_context={"scene_id": request.scene_id, "archetype": request.archetype.value},
        )

    def _upload_to_gcs(
        self,
        scene_id: str,
        local_path: Path,
        checksum_path: Path,
    ) -> Tuple[Optional[str], Optional[SceneGenerationIssue]]:
        """Upload image to GCS and return URI."""
        try:
            bucket = self.storage_client.bucket(self.gcs_bucket)
            blob_path = f"scenes/{scene_id}/source_image.png"
            blob = bucket.blob(blob_path)

            gcs_uri = f"gs://{self.gcs_bucket}/{blob_path}"
            result = upload_blob_from_filename(
                blob,
                local_path,
                gcs_uri,
                logger=LOGGER,
                verify_upload=True,
            )

            if not result.success:
                issue = SceneGenerationIssue(
                    code=SceneGenerationErrorCode.GCS_UPLOAD_FAILED,
                    message="GCS upload failed after retries.",
                    context={
                        "scene_id": scene_id,
                        "bucket": self.gcs_bucket,
                        "attempts": result.attempts,
                        "error": result.error,
                    },
                    fatal=True,
                )
                return None, issue

            print(f"[SCENE-GEN] Uploaded to: {gcs_uri}")

            checksum_blob_path = f"scenes/{scene_id}/source_checksums.json"
            checksum_blob = bucket.blob(checksum_blob_path)
            checksum_uri = f"gs://{self.gcs_bucket}/{checksum_blob_path}"
            checksum_result = upload_blob_from_filename(
                checksum_blob,
                checksum_path,
                checksum_uri,
                logger=LOGGER,
                content_type="application/json",
                verify_upload=True,
            )
            if not checksum_result.success:
                issue = SceneGenerationIssue(
                    code=SceneGenerationErrorCode.GCS_UPLOAD_FAILED,
                    message="GCS checksum upload failed after retries.",
                    context={
                        "scene_id": scene_id,
                        "bucket": self.gcs_bucket,
                        "attempts": checksum_result.attempts,
                        "error": checksum_result.error,
                    },
                    fatal=True,
                )
                return None, issue
            return gcs_uri, None

        except Exception as e:
            issue = SceneGenerationIssue(
                code=SceneGenerationErrorCode.GCS_UPLOAD_FAILED,
                message="GCS upload failed with exception.",
                context={"scene_id": scene_id, "bucket": self.gcs_bucket, "exception": repr(e)},
                fatal=True,
            )
            return None, issue

    def trigger_pipeline(self, scene_id: str) -> bool:
        """Trigger downstream pipeline by writing completion marker."""
        if not self.gcs_bucket or not self.storage_client:
            issue = SceneGenerationIssue(
                code=SceneGenerationErrorCode.PIPELINE_TRIGGER_FAILED,
                message="Cannot trigger pipeline; GCS not configured.",
                context={"scene_id": scene_id},
                fatal=True,
            )
            _log_issue(logging.ERROR, issue)
            return False

        try:
            bucket = self.storage_client.bucket(self.gcs_bucket)

            # Write the source image marker that triggers 3D-RE-GEN
            marker_path = f"scenes/{scene_id}/.scene_generation_complete"
            marker_blob = bucket.blob(marker_path)

            marker_content = json.dumps({
                "scene_id": scene_id,
                "completed_at": datetime.datetime.utcnow().isoformat() + "Z",
                "source": "scene-generation-job",
                "ready_for": "3d-regen"
            }, indent=2)

            marker_uri = f"gs://{self.gcs_bucket}/{marker_path}"
            with tempfile.TemporaryDirectory() as temp_dir:
                marker_file = Path(temp_dir) / ".scene_generation_complete"
                marker_file.write_text(marker_content)
                result = upload_blob_from_filename(
                    marker_blob,
                    marker_file,
                    marker_uri,
                    logger=LOGGER,
                    content_type="application/json",
                    verify_upload=True,
                )

            if not result.success:
                issue = SceneGenerationIssue(
                    code=SceneGenerationErrorCode.PIPELINE_TRIGGER_FAILED,
                    message="Failed to trigger pipeline after retries.",
                    context={
                        "scene_id": scene_id,
                        "bucket": self.gcs_bucket,
                        "attempts": result.attempts,
                        "error": result.error,
                    },
                    fatal=True,
                )
                _log_issue(logging.ERROR, issue)
                return False

            print(f"[SCENE-GEN] Pipeline trigger written: {marker_path}")
            return True

        except Exception as e:
            issue = SceneGenerationIssue(
                code=SceneGenerationErrorCode.PIPELINE_TRIGGER_FAILED,
                message="Failed to trigger pipeline with exception.",
                context={"scene_id": scene_id, "bucket": self.gcs_bucket},
                fatal=True,
            )
            _log_issue(logging.ERROR, issue, exc=e)
            return False


# ============================================================================
# Archetype Selection
# ============================================================================

def select_archetypes_for_batch(
    count: int,
    history_tracker: Optional[GenerationHistoryTracker] = None
) -> List[EnvironmentArchetype]:
    """
    Select archetypes for a batch of generations.

    Uses history to balance coverage while respecting target weights.
    """
    # Get recent counts if history is available
    recent_counts = {}
    if history_tracker:
        recent_counts = history_tracker.get_archetype_counts(days=7)

    # Calculate adjusted weights based on coverage
    adjusted_weights = {}
    total_recent = sum(recent_counts.values()) or 1

    for archetype in EnvironmentArchetype:
        base_weight = ARCHETYPE_WEIGHTS.get(archetype, 0.1)
        recent_count = recent_counts.get(archetype.value, 0)

        # Reduce weight for over-represented archetypes
        if total_recent > 0:
            current_ratio = recent_count / total_recent
            target_ratio = base_weight

            if current_ratio > target_ratio * 1.5:
                # Over-represented: reduce weight
                adjusted_weights[archetype] = base_weight * 0.5
            elif current_ratio < target_ratio * 0.5:
                # Under-represented: increase weight
                adjusted_weights[archetype] = base_weight * 1.5
            else:
                adjusted_weights[archetype] = base_weight
        else:
            adjusted_weights[archetype] = base_weight

    # Normalize weights
    total_weight = sum(adjusted_weights.values())
    normalized = {k: v / total_weight for k, v in adjusted_weights.items()}

    # Select archetypes
    archetypes = list(normalized.keys())
    weights = list(normalized.values())

    selected = random.choices(archetypes, weights=weights, k=count)

    return selected


# ============================================================================
# Main Processing
# ============================================================================

def generate_scene_batch(
    count: int = DEFAULT_SCENES_PER_RUN,
    bucket: Optional[str] = None,
    dry_run: bool = False,
    specific_archetypes: Optional[List[str]] = None,
    output_dir: Optional[Path] = None
) -> Tuple[List[SceneGenerationResult], Dict[str, Any]]:
    """
    Generate a batch of scene images.

    Args:
        count: Number of scenes to generate
        bucket: GCS bucket for storage
        dry_run: If True, skip actual generation
        specific_archetypes: Optional list of specific archetypes to generate
        output_dir: Local output directory

    Returns:
        Tuple of (list of results, summary dict)
    """
    print(f"[SCENE-GEN] Starting batch generation: {count} scenes")

    # Initialize output directory
    if output_dir is None:
        if bucket:
            output_dir = GCS_ROOT / "scenes"
        else:
            output_dir = Path("/tmp/scene_generation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize clients
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        issue = SceneGenerationIssue(
            code=SceneGenerationErrorCode.CONFIG_MISSING_API_KEY,
            message="GEMINI_API_KEY environment variable is required.",
            context={},
            fatal=True,
        )
        _log_issue(logging.ERROR, issue)
        raise SceneGenerationJobError(issue.code, issue.message, issue.context)

    client = genai.Client(api_key=api_key)

    # Initialize tracking
    firestore_project = os.getenv("FIRESTORE_PROJECT")
    history_tracker = GenerationHistoryTracker(project_id=firestore_project)

    # Initialize prompt diversifier
    diversifier = PromptDiversifier(client)

    # Initialize image generator
    generator = SceneImageGenerator(
        client=client,
        output_dir=output_dir,
        gcs_bucket=bucket
    )

    # Select archetypes
    if specific_archetypes:
        archetypes = []
        for name in specific_archetypes:
            try:
                archetypes.append(EnvironmentArchetype(name))
            except ValueError:
                issue = SceneGenerationIssue(
                    code=SceneGenerationErrorCode.UNKNOWN_ARCHETYPE,
                    message="Unknown archetype provided.",
                    context={"archetype": name},
                    fatal=True,
                )
                _log_issue(logging.ERROR, issue)
                raise SceneGenerationJobError(issue.code, issue.message, issue.context)

        # Repeat to fill count
        while len(archetypes) < count:
            archetypes.extend(archetypes[:count - len(archetypes)])
        archetypes = archetypes[:count]
    else:
        archetypes = select_archetypes_for_batch(count, history_tracker)

    print(f"[SCENE-GEN] Selected archetypes: {[a.value for a in archetypes]}")

    # Generate scenes
    results: List[SceneGenerationResult] = []
    history_entries: List[GenerationHistoryEntry] = []

    for i, archetype in enumerate(archetypes):
        print(f"\n[SCENE-GEN] === Scene {i + 1}/{count} ===")

        # Generate unique scene ID
        scene_id = f"{archetype.value}_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        # Get history for diversity
        recent_prompts = history_tracker.get_recent_prompts(archetype)
        variation_coverage = history_tracker.get_variation_coverage(archetype)

        # Generate diverse prompt
        prompt, variation_tags = diversifier.generate_diverse_prompt(
            archetype=archetype,
            recent_prompts=recent_prompts,
            variation_coverage=variation_coverage
        )

        # Create generation request
        request = SceneGenerationRequest(
            scene_id=scene_id,
            archetype=archetype,
            prompt=prompt,
            variation_hints=variation_tags
        )

        # Generate image
        result = generator.generate_scene_image(request, dry_run=dry_run)
        results.append(result)

        if not result.success and result.error_code not in {
            SceneGenerationErrorCode.GCS_UPLOAD_FAILED.value,
            SceneGenerationErrorCode.PIPELINE_TRIGGER_FAILED.value,
        }:
            code = SceneGenerationErrorCode.IMAGE_GENERATION_FAILED
            if result.error_code:
                try:
                    code = SceneGenerationErrorCode(result.error_code)
                except ValueError:
                    code = SceneGenerationErrorCode.IMAGE_GENERATION_FAILED
            issue = SceneGenerationIssue(
                code=code,
                message=result.error or "Scene generation failed.",
                context=result.error_context or {"scene_id": result.scene_id},
                fatal=True,
            )
            _log_issue(logging.ERROR, issue)

        # Record in history
        if result.success:
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
            history_entry = GenerationHistoryEntry(
                scene_id=scene_id,
                archetype=archetype.value,
                prompt_hash=prompt_hash,
                prompt_summary=prompt[:200],
                variation_tags=variation_tags,
                generated_at=datetime.datetime.now(datetime.timezone.utc),
                success=True
            )
            history_entries.append(history_entry)

            # Trigger downstream pipeline
            if not dry_run:
                triggered = generator.trigger_pipeline(scene_id)
                if not triggered:
                    result.success = False
                    result.error = "Pipeline trigger failed."
                    result.error_code = SceneGenerationErrorCode.PIPELINE_TRIGGER_FAILED.value
                    result.error_context = {"scene_id": scene_id, "bucket": bucket}

        # Delay between generations to avoid rate limiting
        if not dry_run and i < count - 1:
            time.sleep(2.0)

    # Compute summary
    successful = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success)
    total_time = sum(r.generation_time_seconds for r in results)

    archetype_counts = {}
    for r in results:
        arch = r.archetype.value
        archetype_counts[arch] = archetype_counts.get(arch, 0) + 1

    history_write_result = history_tracker.record_generations(history_entries)
    if history_write_result.errors:
        issue = SceneGenerationIssue(
            code=SceneGenerationErrorCode.FIRESTORE_WRITE_FAILED,
            message="Failed to record history entries.",
            context={"errors": [error.message for error in history_write_result.errors]},
            fatal=False,
        )
        _log_issue(logging.WARNING, issue)

    summary = {
        "total_attempted": len(results),
        "successful": successful,
        "failed": failed,
        "total_generation_time_seconds": total_time,
        "average_time_per_scene_seconds": total_time / len(results) if results else 0,
        "archetype_distribution": archetype_counts,
        "image_model": GEMINI_IMAGE_MODEL,
        "prompt_model": GEMINI_PRO_MODEL,
        "dry_run": dry_run,
        "history_write": asdict(history_write_result),
    }

    if failed > 0:
        issues = []
        for result in results:
            if result.success:
                continue
            code = SceneGenerationErrorCode.IMAGE_GENERATION_FAILED
            if result.error_code:
                try:
                    code = SceneGenerationErrorCode(result.error_code)
                except ValueError:
                    code = SceneGenerationErrorCode.IMAGE_GENERATION_FAILED
            issues.append(
                SceneGenerationIssue(
                    code=code,
                    message=result.error or "Scene generation failed.",
                    context=result.error_context or {"scene_id": result.scene_id},
                    fatal=True,
                )
            )
        raise SceneGenerationBatchError(
            message="One or more scenes failed to generate or publish.",
            results=results,
            summary=summary,
            issues=issues,
        )

    return results, summary


# ============================================================================
# Entry Point
# ============================================================================

def main():
    """Main entry point for the scene-generation job."""

    # Get configuration from environment
    bucket = os.getenv("BUCKET", "")
    scenes_per_run = int(os.getenv("SCENES_PER_RUN", str(DEFAULT_SCENES_PER_RUN)))
    dry_run = os.getenv("DRY_RUN", "").lower() in {"1", "true", "yes"}
    specific_archetypes_str = os.getenv("ARCHETYPES", "")

    specific_archetypes = None
    if specific_archetypes_str:
        specific_archetypes = [a.strip() for a in specific_archetypes_str.split(",")]

    print(f"[SCENE-GEN] Starting scene generation job")
    print(f"[SCENE-GEN] Scenes per run: {scenes_per_run}")
    print(f"[SCENE-GEN] Bucket: {bucket or '(local only)'}")
    print(f"[SCENE-GEN] Prompt model: {GEMINI_PRO_MODEL}")
    print(f"[SCENE-GEN] Image model: {GEMINI_IMAGE_MODEL}")
    if specific_archetypes:
        print(f"[SCENE-GEN] Specific archetypes: {specific_archetypes}")
    if dry_run:
        print(f"[SCENE-GEN] DRY RUN MODE - no actual generation")

    try:
        metrics = get_metrics()
        with metrics.track_job(JOB_NAME, scene_id="batch"):
            results, summary = generate_scene_batch(
                count=scenes_per_run,
                bucket=bucket or None,
                dry_run=dry_run,
                specific_archetypes=specific_archetypes
            )

        print(f"\n[SCENE-GEN] === Generation Complete ===")
        print(f"[SCENE-GEN] Successful: {summary['successful']}/{summary['total_attempted']}")
        print(f"[SCENE-GEN] Failed: {summary['failed']}")
        print(f"[SCENE-GEN] Total time: {summary['total_generation_time_seconds']:.1f}s")
        print(f"[SCENE-GEN] Archetype distribution: {json.dumps(summary['archetype_distribution'])}")

        # Write summary report
        report_path = Path("/tmp/scene_generation_report.json")
        with report_path.open("w") as f:
            json.dump(
                {
                    "summary": summary,
                    "results": [asdict(r) for r in results],
                    "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
                },
                f,
                indent=2,
                default=str,
            )
        print(f"[SCENE-GEN] Report written to: {report_path}")

        if summary['successful'] > 0:
            print("[SCENE-GEN] SUCCESS")
            sys.exit(0)
        else:
            print("[SCENE-GEN] FAILURE: No scenes generated successfully")
            sys.exit(1)

    except SceneGenerationBatchError as e:
        issue = SceneGenerationIssue(
            code=e.code,
            message="Scene generation batch failed.",
            context={"failed": e.summary.get("failed", 0)},
            fatal=True,
        )
        _log_issue(logging.ERROR, issue)
        report_path = Path("/tmp/scene_generation_report.json")
        with report_path.open("w") as f:
            json.dump(
                {
                    "summary": e.summary,
                    "results": [asdict(r) for r in e.results],
                    "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
                    "issues": [asdict(issue) for issue in e.issues],
                },
                f,
                indent=2,
                default=str,
            )
        print(f"[SCENE-GEN] Report written to: {report_path}")
        sys.exit(1)

    except SceneGenerationJobError as e:
        issue = SceneGenerationIssue(
            code=e.code,
            message=str(e),
            context=e.context,
            fatal=True,
        )
        _log_issue(logging.ERROR, issue)
        sys.exit(1)

    except Exception as e:
        print(f"[SCENE-GEN] ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    from tools.startup_validation import validate_and_fail_fast

    validate_and_fail_fast(job_name="SCENE-GENERATION", validate_gcs=True)
    main()
