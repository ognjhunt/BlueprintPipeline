"""Cost Tracking for Pipeline Operations.

Track and analyze costs for:
- API calls (Gemini, Genie Sim)
- Compute resources (Cloud Run, Cloud Build)
- Storage (GCS)
- External services

Provides per-scene cost visibility and optimization insights.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Pricing constants (as of January 2026)
# Note: Update these based on actual pricing
PRICING = {
    # Gemini API pricing (per 1K tokens)
    "gemini_input_per_1k": 0.00125,  # $0.00125 per 1K input tokens
    "gemini_output_per_1k": 0.005,   # $0.005 per 1K output tokens

    # Cloud Run pricing (per vCPU-second and GB-second)
    "cloud_run_vcpu_second": 0.00002400,  # $0.000024 per vCPU-second
    "cloud_run_memory_gb_second": 0.00000250,  # $0.0000025 per GB-second

    # Cloud Build pricing (per build-minute)
    "cloud_build_minute": 0.003,  # $0.003 per build-minute

    # GCS pricing (per GB per month)
    "gcs_storage_gb_month": 0.020,  # $0.02 per GB per month
    "gcs_operation_class_a": 0.005 / 10000,  # $0.005 per 10K operations
    "gcs_operation_class_b": 0.0004 / 10000,  # $0.0004 per 10K operations

    # Genie Sim API (placeholder - need actual pricing)
    "geniesim_job": 0.10,  # $0.10 per job (PLACEHOLDER)
    "geniesim_episode": 0.002,  # $0.002 per episode (PLACEHOLDER)
}


@dataclass
class CostEntry:
    """Individual cost entry."""
    timestamp: str
    scene_id: str
    category: str  # gemini, cloud_run, gcs, geniesim, etc.
    subcategory: str  # e.g., "input_tokens", "vcpu_time", "storage"
    amount: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "scene_id": self.scene_id,
            "category": self.category,
            "subcategory": self.subcategory,
            "amount": self.amount,
            "metadata": self.metadata,
        }


@dataclass
class CostBreakdown:
    """Cost breakdown for a scene or period."""
    total: float = 0.0

    # API costs
    gemini: float = 0.0
    geniesim: float = 0.0
    other_apis: float = 0.0

    # Compute costs
    cloud_run: float = 0.0
    cloud_build: float = 0.0

    # Storage costs
    gcs_storage: float = 0.0
    gcs_operations: float = 0.0

    # Detailed breakdown
    by_job: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "api_costs": {
                "gemini": self.gemini,
                "geniesim": self.geniesim,
                "other_apis": self.other_apis,
            },
            "compute_costs": {
                "cloud_run": self.cloud_run,
                "cloud_build": self.cloud_build,
            },
            "storage_costs": {
                "gcs_storage": self.gcs_storage,
                "gcs_operations": self.gcs_operations,
            },
            "by_job": self.by_job,
        }


class CostTracker:
    """Track API and compute costs per scene.

    Example:
        tracker = CostTracker()

        # Track Gemini call
        tracker.track_gemini_call("scene_001", tokens_in=1000, tokens_out=500)

        # Track compute
        tracker.track_compute("scene_001", "regen3d-job", duration_seconds=120)

        # Get costs
        costs = tracker.get_scene_cost("scene_001")
        print(f"Total cost: ${costs.total:.4f}")
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        custom_pricing: Optional[Dict[str, float]] = None,
        enable_logging: bool = True,
    ):
        """Initialize cost tracker.

        Args:
            data_dir: Directory for storing cost data
            custom_pricing: Override default pricing
            enable_logging: Whether to log cost tracking
        """
        self.data_dir = Path(data_dir or os.getenv("COST_DATA_DIR", "./cost_data"))
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.pricing = PRICING.copy()
        if custom_pricing:
            self.pricing.update(custom_pricing)

        self.enable_logging = enable_logging

        # In-memory cache
        self.entries: List[CostEntry] = []

        # Load existing data
        self._load_data()

    def log(self, msg: str) -> None:
        """Log if enabled."""
        if self.enable_logging:
            logger.info(f"[COST] {msg}")

    def track_gemini_call(
        self,
        scene_id: str,
        tokens_in: int,
        tokens_out: int,
        operation: str = "",
    ) -> float:
        """Track Gemini API call cost.

        Args:
            scene_id: Scene identifier
            tokens_in: Number of input tokens
            tokens_out: Number of output tokens
            operation: Optional operation name

        Returns:
            Cost of this call
        """
        # Calculate cost
        cost_in = (tokens_in / 1000) * self.pricing["gemini_input_per_1k"]
        cost_out = (tokens_out / 1000) * self.pricing["gemini_output_per_1k"]
        total_cost = cost_in + cost_out

        # Record
        self._record(
            scene_id=scene_id,
            category="gemini",
            subcategory="api_call",
            amount=total_cost,
            metadata={
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "operation": operation,
                "cost_in": cost_in,
                "cost_out": cost_out,
            }
        )

        self.log(
            f"Gemini call (scene: {scene_id}): "
            f"{tokens_in} in, {tokens_out} out = ${total_cost:.4f}"
        )

        return total_cost

    def track_compute(
        self,
        scene_id: str,
        job_name: str,
        duration_seconds: float,
        vcpu_count: int = 1,
        memory_gb: float = 2.0,
    ) -> float:
        """Track Cloud Run compute cost.

        Args:
            scene_id: Scene identifier
            job_name: Job name
            duration_seconds: Duration in seconds
            vcpu_count: Number of vCPUs
            memory_gb: Memory in GB

        Returns:
            Cost of this compute
        """
        # Calculate cost
        vcpu_cost = duration_seconds * vcpu_count * self.pricing["cloud_run_vcpu_second"]
        memory_cost = duration_seconds * memory_gb * self.pricing["cloud_run_memory_gb_second"]
        total_cost = vcpu_cost + memory_cost

        # Record
        self._record(
            scene_id=scene_id,
            category="cloud_run",
            subcategory=job_name,
            amount=total_cost,
            metadata={
                "duration_seconds": duration_seconds,
                "vcpu_count": vcpu_count,
                "memory_gb": memory_gb,
                "vcpu_cost": vcpu_cost,
                "memory_cost": memory_cost,
            }
        )

        self.log(
            f"Cloud Run {job_name} (scene: {scene_id}): "
            f"{duration_seconds:.1f}s × {vcpu_count} vCPU = ${total_cost:.4f}"
        )

        return total_cost

    def track_geniesim_job(
        self,
        scene_id: str,
        job_id: str,
        episode_count: int,
        duration_seconds: float = 0,
    ) -> float:
        """Track Genie Sim API cost.

        Args:
            scene_id: Scene identifier
            job_id: Genie Sim job ID
            episode_count: Number of episodes generated
            duration_seconds: Optional job duration

        Returns:
            Cost of this job
        """
        # Calculate cost (job base cost + per-episode cost)
        job_cost = self.pricing["geniesim_job"]
        episode_cost = episode_count * self.pricing["geniesim_episode"]
        total_cost = job_cost + episode_cost

        # Record
        self._record(
            scene_id=scene_id,
            category="geniesim",
            subcategory="job",
            amount=total_cost,
            metadata={
                "job_id": job_id,
                "episode_count": episode_count,
                "duration_seconds": duration_seconds,
                "job_cost": job_cost,
                "episode_cost": episode_cost,
            }
        )

        self.log(
            f"Genie Sim job {job_id} (scene: {scene_id}): "
            f"{episode_count} episodes = ${total_cost:.4f}"
        )

        return total_cost

    def track_storage(
        self,
        scene_id: str,
        bytes_written: int,
        operation_count: int = 1,
        operation_class: str = "A",
    ) -> float:
        """Track GCS storage cost.

        Args:
            scene_id: Scene identifier
            bytes_written: Number of bytes written
            operation_count: Number of operations
            operation_class: Operation class ("A" or "B")

        Returns:
            Cost of storage operations
        """
        # Calculate cost
        # Note: Monthly storage cost is amortized per scene
        gb = bytes_written / (1024 ** 3)
        storage_cost = gb * self.pricing["gcs_storage_gb_month"] / 30  # Per day

        if operation_class == "A":
            op_cost = operation_count * self.pricing["gcs_operation_class_a"]
        else:
            op_cost = operation_count * self.pricing["gcs_operation_class_b"]

        total_cost = storage_cost + op_cost

        # Record
        self._record(
            scene_id=scene_id,
            category="gcs",
            subcategory="storage",
            amount=total_cost,
            metadata={
                "bytes_written": bytes_written,
                "gb": gb,
                "operation_count": operation_count,
                "operation_class": operation_class,
                "storage_cost": storage_cost,
                "operation_cost": op_cost,
            }
        )

        self.log(
            f"GCS storage (scene: {scene_id}): "
            f"{gb:.3f} GB + {operation_count} ops = ${total_cost:.4f}"
        )

        return total_cost

    def track_cloud_build(
        self,
        scene_id: str,
        build_name: str,
        duration_minutes: float,
    ) -> float:
        """Track Cloud Build cost.

        Args:
            scene_id: Scene identifier
            build_name: Build name
            duration_minutes: Build duration in minutes

        Returns:
            Cost of this build
        """
        total_cost = duration_minutes * self.pricing["cloud_build_minute"]

        # Record
        self._record(
            scene_id=scene_id,
            category="cloud_build",
            subcategory=build_name,
            amount=total_cost,
            metadata={
                "duration_minutes": duration_minutes,
            }
        )

        self.log(
            f"Cloud Build {build_name} (scene: {scene_id}): "
            f"{duration_minutes:.1f} min = ${total_cost:.4f}"
        )

        return total_cost

    def get_scene_cost(self, scene_id: str) -> CostBreakdown:
        """Get cost breakdown for a scene.

        Args:
            scene_id: Scene identifier

        Returns:
            CostBreakdown for the scene
        """
        scene_entries = [e for e in self.entries if e.scene_id == scene_id]

        breakdown = CostBreakdown()

        for entry in scene_entries:
            breakdown.total += entry.amount

            # Category totals
            if entry.category == "gemini":
                breakdown.gemini += entry.amount
            elif entry.category == "geniesim":
                breakdown.geniesim += entry.amount
            elif entry.category == "cloud_run":
                breakdown.cloud_run += entry.amount
                # By job
                job_name = entry.subcategory
                breakdown.by_job[job_name] = breakdown.by_job.get(job_name, 0) + entry.amount
            elif entry.category == "cloud_build":
                breakdown.cloud_build += entry.amount
            elif entry.category == "gcs":
                if entry.subcategory == "storage":
                    breakdown.gcs_storage += entry.amount
                else:
                    breakdown.gcs_operations += entry.amount

        return breakdown

    def get_period_cost(
        self,
        days: int = 30,
        customer_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get cost breakdown for a time period.

        Args:
            days: Number of days to include
            customer_id: Optional customer filter

        Returns:
            Period cost breakdown
        """
        end = datetime.utcnow()
        start = end - timedelta(days=days)

        # Filter entries
        period_entries = [
            e for e in self.entries
            if e.timestamp >= start.isoformat()
        ]

        # Aggregate
        total = sum(e.amount for e in period_entries)
        by_category = {}
        by_scene = {}

        for entry in period_entries:
            # By category
            by_category[entry.category] = by_category.get(entry.category, 0) + entry.amount

            # By scene
            by_scene[entry.scene_id] = by_scene.get(entry.scene_id, 0) + entry.amount

        # Get top scenes
        top_scenes = sorted(
            by_scene.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        return {
            "period": {
                "start": start.isoformat() + "Z",
                "end": end.isoformat() + "Z",
                "days": days,
            },
            "total": total,
            "by_category": by_category,
            "scene_count": len(by_scene),
            "avg_per_scene": total / len(by_scene) if by_scene else 0,
            "top_scenes": [
                {"scene_id": scene_id, "cost": cost}
                for scene_id, cost in top_scenes
            ],
        }

    def get_optimization_insights(self, scene_id: str) -> Dict[str, Any]:
        """Get cost optimization insights for a scene.

        Args:
            scene_id: Scene identifier

        Returns:
            Optimization insights
        """
        breakdown = self.get_scene_cost(scene_id)

        insights = {
            "total_cost": breakdown.total,
            "recommendations": [],
        }

        # Check Gemini usage
        if breakdown.gemini > 0.50:  # More than $0.50 on Gemini
            insights["recommendations"].append({
                "category": "gemini",
                "message": "High Gemini API costs. Consider caching physics estimates for similar objects.",
                "potential_savings": breakdown.gemini * 0.3,  # Estimate 30% savings
            })

        # Check Genie Sim usage
        if breakdown.geniesim > 1.00:  # More than $1.00 on Genie Sim
            insights["recommendations"].append({
                "category": "geniesim",
                "message": "High Genie Sim costs. Consider reducing episode count or using batch submission.",
                "potential_savings": breakdown.geniesim * 0.2,
            })

        # Check compute efficiency
        if breakdown.cloud_run > 0.20:  # More than $0.20 on compute
            total_duration = sum(
                e.metadata.get("duration_seconds", 0)
                for e in self.entries
                if e.scene_id == scene_id and e.category == "cloud_run"
            )
            if total_duration > 3600:  # More than 1 hour
                insights["recommendations"].append({
                    "category": "cloud_run",
                    "message": "Long processing time. Consider parallel processing or algorithm optimization.",
                    "potential_savings": breakdown.cloud_run * 0.4,
                })

        return insights

    def _record(
        self,
        scene_id: str,
        category: str,
        subcategory: str,
        amount: float,
        metadata: Dict[str, Any],
    ) -> None:
        """Record a cost entry."""
        entry = CostEntry(
            timestamp=datetime.utcnow().isoformat() + "Z",
            scene_id=scene_id,
            category=category,
            subcategory=subcategory,
            amount=amount,
            metadata=metadata,
        )

        self.entries.append(entry)
        self._save_entry(entry)

    def _load_data(self) -> None:
        """Load existing cost data."""
        entries_file = self.data_dir / "cost_entries.jsonl"
        if entries_file.exists():
            try:
                with open(entries_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            entry = CostEntry(**data)
                            self.entries.append(entry)
            except Exception as e:
                logger.warning(f"Failed to load cost data: {e}")

    def _save_entry(self, entry: CostEntry) -> None:
        """Append entry to cost log."""
        entries_file = self.data_dir / "cost_entries.jsonl"
        try:
            with open(entries_file, 'a') as f:
                f.write(json.dumps(entry.to_dict()) + '\n')
        except Exception as e:
            logger.warning(f"Failed to save cost entry: {e}")


# Global singleton instance
_cost_tracker: Optional[CostTracker] = None


def get_cost_tracker() -> CostTracker:
    """Get global cost tracker instance.

    Returns:
        Global CostTracker instance

    Example:
        from tools.cost_tracking import get_cost_tracker

        tracker = get_cost_tracker()
        tracker.track_gemini_call("scene_001", 1000, 500)
    """
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker()
    return _cost_tracker
