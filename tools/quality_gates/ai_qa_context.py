"""AI-Guided QA Context Generation.

Uses LLM to analyze pipeline outputs and generate intelligent QA guidance
for human reviewers. Tells humans exactly what to check and why.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import LLM client
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from tools.llm_client.client import create_llm_client, LLMProvider
except ImportError:
    create_llm_client = None
    LLMProvider = None


@dataclass
class QAContextItem:
    """Single QA context item for human review."""
    category: str
    title: str
    description: str
    priority: str  # critical, high, medium, low
    what_to_check: List[str]
    why_it_matters: str
    estimated_time: str  # e.g., "2-5 minutes"
    auto_checkable: bool = False
    passed_auto_check: Optional[bool] = None


@dataclass
class QAContext:
    """Complete QA context for a checkpoint."""
    checkpoint: str
    scene_id: str
    timestamp: str = ""
    summary: str = ""
    items: List[QAContextItem] = field(default_factory=list)
    total_review_time: str = ""
    critical_count: int = 0
    auto_passed_count: int = 0
    requires_human_review: bool = True

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + "Z"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "scene_id": self.scene_id,
            "timestamp": self.timestamp,
            "summary": self.summary,
            "total_review_time": self.total_review_time,
            "critical_count": self.critical_count,
            "auto_passed_count": self.auto_passed_count,
            "requires_human_review": self.requires_human_review,
            "items": [
                {
                    "category": item.category,
                    "title": item.title,
                    "description": item.description,
                    "priority": item.priority,
                    "what_to_check": item.what_to_check,
                    "why_it_matters": item.why_it_matters,
                    "estimated_time": item.estimated_time,
                    "auto_checkable": item.auto_checkable,
                    "passed_auto_check": item.passed_auto_check,
                }
                for item in self.items
            ],
        }

    def to_email_format(self) -> str:
        """Format context for email notification."""
        lines = [
            f"QA Review Required: {self.checkpoint}",
            f"Scene: {self.scene_id}",
            f"Estimated Review Time: {self.total_review_time}",
            "",
            "=" * 50,
            self.summary,
            "=" * 50,
            "",
        ]

        # Group by priority
        for priority in ["critical", "high", "medium", "low"]:
            priority_items = [i for i in self.items if i.priority == priority]
            if priority_items:
                lines.append(f"\n{priority.upper()} PRIORITY ITEMS:")
                lines.append("-" * 40)

                for item in priority_items:
                    status = ""
                    if item.auto_checkable:
                        status = " [AUTO-PASSED]" if item.passed_auto_check else " [AUTO-FAILED]"

                    lines.append(f"\n{item.title}{status}")
                    lines.append(f"  Category: {item.category}")
                    lines.append(f"  Time: {item.estimated_time}")
                    lines.append(f"  Why: {item.why_it_matters}")
                    lines.append("  What to Check:")
                    for check in item.what_to_check:
                        lines.append(f"    - {check}")

        return "\n".join(lines)


class QAContextGenerator:
    """Generates intelligent QA context using AI."""

    # Checkpoint-specific prompts
    CHECKPOINT_PROMPTS = {
        "manifest_validated": """
You are reviewing a scene manifest for a robotics simulation pipeline.
The manifest defines objects, their transforms, and asset references.

Analyze the manifest and identify:
1. Objects that might have incorrect scales (too big/small for their category)
2. Objects with unusual positions (floating, intersecting)
3. Missing or invalid asset references
4. Physics hints that seem implausible for the object type

Focus on issues that would cause simulation failures or unrealistic behavior.
""",

        "simready_complete": """
You are reviewing physics properties estimated by AI for simulation objects.
These properties will be used in physics simulation and must be realistic.

Analyze the physics data and identify:
1. Mass values that seem incorrect for the object type and size
2. Friction coefficients that don't match expected materials
3. Collision proxies that might cause physics issues
4. Center of mass positions that could cause unstable behavior

Focus on properties critical for manipulation tasks (grasping, placing, pushing).
""",

        "usd_assembled": """
You are reviewing a USD scene assembled for robotics simulation.
The scene will be loaded in Isaac Sim for episode generation.

Analyze the USD structure and identify:
1. Missing or broken asset references
2. Objects without proper physics APIs applied
3. Hierarchy issues that could affect simulation
4. Material/texture problems that could affect visual observations

Focus on issues that would prevent the scene from loading or simulating correctly.
""",

        "episodes_generated": """
You are reviewing generated training episodes for robot learning.
These episodes will be used to train manipulation policies.

Analyze the episode data and identify:
1. Trajectories with collisions or failures
2. Episodes with unusual timing or dynamics
3. Quality score distributions and outliers
4. Task completion patterns and success rates

Focus on episode quality for sim-to-real transfer.
""",

        "scene_ready": """
You are performing a final review before delivering a simulation-ready scene.
This is the last checkpoint before the scene is used for robot training.

Perform a comprehensive review covering:
1. Overall scene quality and completeness
2. Physics simulation stability
3. Episode generation success rate
4. Visual quality for observation learning
5. Any blockers for sim-to-real transfer

This is a critical gate - be thorough but prioritize actionable feedback.
""",
    }

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.llm_client = None

        # Try to initialize LLM client
        if create_llm_client:
            try:
                self.llm_client = create_llm_client()
                self.log("LLM client initialized for AI-guided QA")
            except Exception as e:
                self.log(f"LLM client unavailable: {e}")

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[QA-CONTEXT] {msg}")

    def generate(
        self,
        checkpoint: str,
        scene_id: str,
        context_data: Dict[str, Any],
        images: Optional[List[Path]] = None,
    ) -> QAContext:
        """Generate QA context for a checkpoint.

        Args:
            checkpoint: Pipeline checkpoint name
            scene_id: Scene identifier
            context_data: Relevant data for the checkpoint
            images: Optional images to analyze (screenshots, renders)

        Returns:
            QAContext with review guidance
        """
        self.log(f"Generating QA context for {checkpoint}")

        # Try AI-powered generation first
        if self.llm_client:
            try:
                return self._generate_with_ai(checkpoint, scene_id, context_data, images)
            except Exception as e:
                self.log(f"AI generation failed, using fallback: {e}")

        # Fallback to rule-based generation
        return self._generate_fallback(checkpoint, scene_id, context_data)

    def _generate_with_ai(
        self,
        checkpoint: str,
        scene_id: str,
        context_data: Dict[str, Any],
        images: Optional[List[Path]] = None,
    ) -> QAContext:
        """Generate context using LLM."""

        # Build prompt
        base_prompt = self.CHECKPOINT_PROMPTS.get(
            checkpoint,
            "Review the following data and identify issues that require human attention."
        )

        prompt = f"""
{base_prompt}

Scene ID: {scene_id}
Checkpoint: {checkpoint}

Data to analyze:
```json
{json.dumps(context_data, indent=2, default=str)[:8000]}  # Truncate if too long
```

Generate a structured QA review guide in JSON format:
{{
    "summary": "Brief overall assessment",
    "total_review_time": "Estimated time (e.g., '5-10 minutes')",
    "items": [
        {{
            "category": "Category (e.g., 'Physics', 'Geometry', 'Assets')",
            "title": "Short title",
            "description": "What this item is about",
            "priority": "critical|high|medium|low",
            "what_to_check": ["Specific check 1", "Specific check 2"],
            "why_it_matters": "Why this is important for sim-to-real transfer",
            "estimated_time": "1-2 minutes"
        }}
    ]
}}

Focus on actionable items that require human judgment. Prioritize issues that:
1. Could cause simulation failures
2. Would affect sim-to-real transfer
3. Cannot be automatically validated

Return only valid JSON, no additional text.
"""

        # Call LLM
        response = self.llm_client.generate(
            prompt=prompt,
            images=[str(img) for img in images] if images else None,
            json_output=True,
            temperature=0.3,  # Lower temperature for structured output
        )

        # Parse response
        try:
            data = response.parse_json()
        except json.JSONDecodeError:
            # Try to extract JSON from response
            text = response.text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
            else:
                raise ValueError("Could not parse LLM response as JSON")

        # Build QAContext
        items = []
        for item_data in data.get("items", []):
            items.append(QAContextItem(
                category=item_data.get("category", "General"),
                title=item_data.get("title", "Review Item"),
                description=item_data.get("description", ""),
                priority=item_data.get("priority", "medium"),
                what_to_check=item_data.get("what_to_check", []),
                why_it_matters=item_data.get("why_it_matters", ""),
                estimated_time=item_data.get("estimated_time", "2-5 minutes"),
            ))

        critical_count = sum(1 for i in items if i.priority == "critical")

        return QAContext(
            checkpoint=checkpoint,
            scene_id=scene_id,
            summary=data.get("summary", "Review required"),
            items=items,
            total_review_time=data.get("total_review_time", "10-15 minutes"),
            critical_count=critical_count,
            requires_human_review=critical_count > 0 or len(items) > 0,
        )

    def _generate_fallback(
        self,
        checkpoint: str,
        scene_id: str,
        context_data: Dict[str, Any],
    ) -> QAContext:
        """Generate context using rule-based logic (fallback)."""

        items = []

        if checkpoint == "manifest_validated":
            manifest = context_data.get("manifest", {})
            objects = manifest.get("objects", [])

            items.append(QAContextItem(
                category="Scene Completeness",
                title="Object Count Verification",
                description=f"Scene has {len(objects)} objects",
                priority="high",
                what_to_check=[
                    "Count matches expected objects in source image",
                    "All major furniture/items are represented",
                    "No duplicate or extra objects",
                ],
                why_it_matters="Missing objects lead to incomplete training scenarios",
                estimated_time="2-3 minutes",
            ))

            items.append(QAContextItem(
                category="Geometry",
                title="Scale and Position Check",
                description="Verify object scales are realistic",
                priority="critical",
                what_to_check=[
                    "Countertops are ~0.9m high",
                    "Doors are ~2m high",
                    "Objects don't intersect or float",
                    "Scale relationships are correct (mug < pot < fridge)",
                ],
                why_it_matters="Wrong scales cause sim-to-real transfer failures",
                estimated_time="3-5 minutes",
            ))

        elif checkpoint == "simready_complete":
            items.append(QAContextItem(
                category="Physics",
                title="Mass Estimation Review",
                description="Review AI-estimated masses",
                priority="high",
                what_to_check=[
                    "Heavy items (appliances, furniture) have realistic masses",
                    "Light items (utensils, cups) are appropriately light",
                    "Mass values support stable grasping",
                ],
                why_it_matters="Wrong masses cause unrealistic dynamics and grasp failures",
                estimated_time="3-5 minutes",
            ))

            items.append(QAContextItem(
                category="Physics",
                title="Friction Properties",
                description="Review surface friction estimates",
                priority="medium",
                what_to_check=[
                    "Slippery surfaces (glass, metal) have low friction",
                    "Grippy surfaces (rubber, wood) have higher friction",
                    "Values support stable placement",
                ],
                why_it_matters="Friction affects grasp stability and object placement",
                estimated_time="2-3 minutes",
            ))

        elif checkpoint == "usd_assembled":
            items.append(QAContextItem(
                category="USD Structure",
                title="Scene Loads Correctly",
                description="Verify scene loads in USD viewer or Isaac Sim",
                priority="critical",
                what_to_check=[
                    "Scene opens without errors",
                    "All objects are visible",
                    "No missing textures (pink/magenta areas)",
                    "No broken hierarchies",
                ],
                why_it_matters="Scene must load correctly for any downstream work",
                estimated_time="2-3 minutes",
            ))

            items.append(QAContextItem(
                category="Physics",
                title="Collision Proxies",
                description="Verify collision shapes are correct",
                priority="high",
                what_to_check=[
                    "Collision shapes approximate visual geometry",
                    "No gaps in collision (robot hand could pass through)",
                    "Articulated objects have correct joint limits",
                ],
                why_it_matters="Bad collisions cause unrealistic interactions",
                estimated_time="5-10 minutes",
            ))

        elif checkpoint == "episodes_generated":
            stats = context_data.get("episode_stats", {})

            items.append(QAContextItem(
                category="Episode Quality",
                title="Quality Score Distribution",
                description=f"Average quality: {stats.get('average_quality_score', 'N/A')}",
                priority="high",
                what_to_check=[
                    "Average quality score > 0.7",
                    "No large clusters of low-quality episodes",
                    "Failed episodes have clear patterns",
                ],
                why_it_matters="Low quality episodes hurt training efficiency",
                estimated_time="3-5 minutes",
            ))

            items.append(QAContextItem(
                category="Episode Quality",
                title="Sample Episode Review",
                description="Watch 2-3 episodes end-to-end",
                priority="critical",
                what_to_check=[
                    "Robot movements are smooth and realistic",
                    "Objects are manipulated correctly",
                    "No physics glitches (explosions, teleporting)",
                    "Task completion looks natural",
                ],
                why_it_matters="Visual review catches issues metrics miss",
                estimated_time="5-10 minutes",
            ))

        elif checkpoint == "scene_ready":
            items.append(QAContextItem(
                category="Final Review",
                title="End-to-End Validation",
                description="Final check before delivery",
                priority="critical",
                what_to_check=[
                    "Scene loads in Isaac Sim",
                    "Physics runs for 100+ steps without issues",
                    "Episodes exist and have good quality",
                    "All required outputs are present",
                ],
                why_it_matters="This is the last gate before customer delivery",
                estimated_time="10-15 minutes",
            ))

        # Calculate totals
        critical_count = sum(1 for i in items if i.priority == "critical")
        total_time = sum(
            int(i.estimated_time.split("-")[1].split()[0])
            for i in items
            if "-" in i.estimated_time
        )

        return QAContext(
            checkpoint=checkpoint,
            scene_id=scene_id,
            summary=f"Review required for {checkpoint}: {len(items)} items, {critical_count} critical",
            items=items,
            total_review_time=f"{total_time}-{total_time + 5} minutes",
            critical_count=critical_count,
            requires_human_review=True,
        )


# Convenience function
def generate_qa_context(
    checkpoint: str,
    scene_id: str,
    context_data: Dict[str, Any],
    images: Optional[List[Path]] = None,
    verbose: bool = True,
) -> QAContext:
    """Generate QA context for a checkpoint.

    Args:
        checkpoint: Pipeline checkpoint name
        scene_id: Scene identifier
        context_data: Relevant data for the checkpoint
        images: Optional images to analyze
        verbose: Print progress

    Returns:
        QAContext with review guidance
    """
    generator = QAContextGenerator(verbose=verbose)
    return generator.generate(checkpoint, scene_id, context_data, images)
