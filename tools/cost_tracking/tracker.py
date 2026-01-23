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

from tools.config.production_mode import resolve_production_mode

logger = logging.getLogger(__name__)


# Pricing constants (as of January 2026)
# Note: Update these based on actual pricing.
# These defaults are intended for development only; production must provide
# a full pricing configuration.
PLACEHOLDER_GENIESIM_JOB_COST = 0.10
PLACEHOLDER_GENIESIM_EPISODE_COST = 0.002

GENIESIM_PRICING_ENV_VARS = ("GENIESIM_JOB_COST", "GENIESIM_EPISODE_COST")
GENIESIM_GPU_RATE_TABLE_ENV = "GENIESIM_GPU_RATE_TABLE"
GENIESIM_GPU_RATE_TABLE_PATH_ENV = "GENIESIM_GPU_RATE_TABLE_PATH"
GENIESIM_GPU_HOURLY_RATE_ENV = "GENIESIM_GPU_HOURLY_RATE"
GENIESIM_GPU_REGION_ENV = "GENIESIM_GPU_REGION"
GENIESIM_GPU_NODE_TYPE_ENV = "GENIESIM_GPU_NODE_TYPE"
COST_TRACKING_PRICING_ENV = "COST_TRACKING_PRICING_JSON"
COST_TRACKING_PRICING_PATH_ENV = "COST_TRACKING_PRICING_PATH"
DEFAULT_PRICING_PATH = Path(__file__).with_name("pricing_defaults.json")
COST_ALERT_PER_SCENE_ENV = "COST_ALERT_PER_SCENE_USD"
COST_ALERT_TOTAL_ENV = "COST_ALERT_TOTAL_USD"

DEFAULT_GENIESIM_GPU_RATES = {
    "default": {
        "g5.xlarge": 1.006,
        "g5.2xlarge": 1.212,
        "g5.12xlarge": 4.384,
        "a2-highgpu-1g": 1.685,
    },
}


def _parse_float(value: Any, *, env_var: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid float for {env_var}: {value!r}") from exc


def _parse_optional_threshold(env_var: str) -> Optional[float]:
    value = os.getenv(env_var)
    if value is None or value == "":
        return None
    threshold = _parse_float(value, env_var=env_var)
    if threshold <= 0:
        logger.warning("Ignoring non-positive %s=%s for cost alert threshold.", env_var, value)
        return None
    return threshold


def _sanitize_pricing_payload(parsed: Any, *, source: str) -> Dict[str, float]:
    if not isinstance(parsed, dict):
        raise ValueError(f"Pricing data in {source} must be a mapping.")
    sanitized: Dict[str, float] = {}
    for key, value in parsed.items():
        if not isinstance(key, str):
            raise ValueError(f"Pricing keys must be strings in {source}.")
        sanitized[key] = _parse_float(value, env_var=source)
    return sanitized


def _parse_pricing_payload(payload: str, *, source: str) -> Dict[str, float]:
    try:
        parsed = json.loads(payload)
        return _sanitize_pricing_payload(parsed, source=source)
    except json.JSONDecodeError:
        pass
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ValueError(
            f"Pricing data in {source} is not valid JSON and PyYAML is not available."
        ) from exc
    try:
        parsed = yaml.safe_load(payload)
    except Exception as exc:  # pragma: no cover - yaml errors are library-specific
        raise ValueError(f"Invalid pricing YAML in {source}: {exc}") from exc
    return _sanitize_pricing_payload(parsed, source=source)


def _load_pricing_defaults() -> tuple[Dict[str, float], Dict[str, Any]]:
    if not DEFAULT_PRICING_PATH.exists():
        raise FileNotFoundError(f"Pricing defaults not found: {DEFAULT_PRICING_PATH}")
    try:
        parsed = json.loads(DEFAULT_PRICING_PATH.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Pricing defaults file is not valid JSON: {DEFAULT_PRICING_PATH}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"Pricing defaults must be a JSON object: {DEFAULT_PRICING_PATH}")
    if "pricing" not in parsed:
        raise ValueError(f"Pricing defaults must include a 'pricing' block: {DEFAULT_PRICING_PATH}")
    pricing = _sanitize_pricing_payload(parsed["pricing"], source=str(DEFAULT_PRICING_PATH))
    metadata = {key: value for key, value in parsed.items() if key != "pricing"}
    return pricing, metadata


def _validate_pricing_config(pricing: Dict[str, float], *, source: str) -> None:
    missing_fields = REQUIRED_PRICING_FIELDS - pricing.keys()
    if missing_fields:
        missing_list = ", ".join(sorted(missing_fields))
        raise ValueError(f"Pricing config in {source} is missing fields: {missing_list}.")
    _validate_pricing_values(pricing, source=source, required_fields=REQUIRED_PRICING_FIELDS)


def _validate_pricing_values(
    pricing: Dict[str, float],
    *,
    source: str,
    required_fields: Optional[frozenset[str]] = None,
) -> None:
    required = required_fields or frozenset(pricing.keys())
    missing_fields = required - pricing.keys()
    if missing_fields:
        missing_list = ", ".join(sorted(missing_fields))
        raise ValueError(f"Pricing data from {source} is missing fields: {missing_list}.")
    negative_fields = [key for key, value in pricing.items() if value < 0]
    if negative_fields:
        negative_list = ", ".join(sorted(negative_fields))
        raise ValueError(f"Pricing data from {source} has negative values for: {negative_list}.")


DEFAULT_PRICING, DEFAULT_PRICING_METADATA = _load_pricing_defaults()
REQUIRED_PRICING_FIELDS = frozenset(DEFAULT_PRICING.keys())
_validate_pricing_values(
    DEFAULT_PRICING,
    source=str(DEFAULT_PRICING_PATH),
    required_fields=REQUIRED_PRICING_FIELDS,
)


def _load_custom_pricing_from_env() -> tuple[Dict[str, float], Optional[str]]:
    pricing_json = os.getenv(COST_TRACKING_PRICING_ENV)
    pricing_path = os.getenv(COST_TRACKING_PRICING_PATH_ENV)
    if pricing_json:
        pricing = _parse_pricing_payload(pricing_json, source=COST_TRACKING_PRICING_ENV)
        _validate_pricing_config(pricing, source=COST_TRACKING_PRICING_ENV)
        return pricing, COST_TRACKING_PRICING_ENV
    if pricing_path:
        path = Path(pricing_path)
        if not path.exists():
            raise FileNotFoundError(f"Pricing config not found: {pricing_path}")
        pricing = _parse_pricing_payload(path.read_text(), source=pricing_path)
        _validate_pricing_config(pricing, source=pricing_path)
        return pricing, pricing_path
    return {}, None


def _is_production_env() -> bool:
    return resolve_production_mode()


def _load_geniesim_pricing_from_env() -> Dict[str, float]:
    pricing_overrides: Dict[str, float] = {}
    job_cost = os.getenv("GENIESIM_JOB_COST")
    episode_cost = os.getenv("GENIESIM_EPISODE_COST")
    if job_cost is not None:
        pricing_overrides["geniesim_job"] = _parse_float(job_cost, env_var="GENIESIM_JOB_COST")
    if episode_cost is not None:
        pricing_overrides["geniesim_episode"] = _parse_float(
            episode_cost,
            env_var="GENIESIM_EPISODE_COST",
        )
    return pricing_overrides


def _validate_geniesim_pricing(pricing: Dict[str, float]) -> None:
    _validate_pricing_values(
        {
            "geniesim_job": pricing.get("geniesim_job", DEFAULT_PRICING["geniesim_job"]),
            "geniesim_episode": pricing.get(
                "geniesim_episode",
                DEFAULT_PRICING["geniesim_episode"],
            ),
        },
        source="geniesim pricing",
    )
    if DEFAULT_PRICING_METADATA.get("geniesim_pricing_placeholder") is True:
        raise ValueError(
            "Genie Sim pricing defaults are marked as placeholders; provide overrides via "
            f"{COST_TRACKING_PRICING_ENV}, {COST_TRACKING_PRICING_PATH_ENV}, or "
            f"{', '.join(GENIESIM_PRICING_ENV_VARS)}."
        )
    placeholder_values = {
        "geniesim_job": PLACEHOLDER_GENIESIM_JOB_COST,
        "geniesim_episode": PLACEHOLDER_GENIESIM_EPISODE_COST,
    }
    placeholder_keys = [
        key for key, value in placeholder_values.items() if pricing.get(key) == value
    ]
    if placeholder_keys:
        placeholder_list = ", ".join(sorted(placeholder_keys))
        raise ValueError(
            "Genie Sim pricing uses placeholder values for: "
            f"{placeholder_list}. Provide real values via "
            f"{', '.join(GENIESIM_PRICING_ENV_VARS)} or a pricing JSON override."
        )


def _require_geniesim_env_vars_in_production() -> None:
    if _is_production_env():
        missing_env_vars = [
            env_var for env_var in GENIESIM_PRICING_ENV_VARS if not os.getenv(env_var)
        ]
        if missing_env_vars:
            missing_list = ", ".join(missing_env_vars)
            raise RuntimeError(
                "Missing required Genie Sim pricing environment variables in production: "
                f"{missing_list}."
            )


def _load_geniesim_gpu_rate_table() -> Dict[str, Any]:
    rate_table_json = os.getenv(GENIESIM_GPU_RATE_TABLE_ENV)
    rate_table_path = os.getenv(GENIESIM_GPU_RATE_TABLE_PATH_ENV)
    if rate_table_json:
        try:
            parsed = json.loads(rate_table_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid GPU rate table JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("GENIESIM_GPU_RATE_TABLE must be a JSON object.")
        return parsed
    if rate_table_path:
        path = Path(rate_table_path)
        if not path.exists():
            raise FileNotFoundError(f"GPU rate table config not found: {rate_table_path}")
        try:
            parsed = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid GPU rate table JSON in {rate_table_path}: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("GPU rate table config must be a JSON object.")
        return parsed
    return DEFAULT_GENIESIM_GPU_RATES.copy()


def _resolve_gpu_hourly_rate(
    *,
    rate_table: Dict[str, Any],
    region: Optional[str],
    node_type: Optional[str],
) -> Optional[float]:
    if not rate_table:
        return None
    region_key = region or "default"
    if isinstance(rate_table.get(region_key), dict):
        region_rates = rate_table.get(region_key, {})
        if node_type and node_type in region_rates:
            return _parse_float(region_rates[node_type], env_var="GENIESIM_GPU_RATE_TABLE")
        if "default" in region_rates:
            return _parse_float(region_rates["default"], env_var="GENIESIM_GPU_RATE_TABLE")
        return None
    if node_type and node_type in rate_table:
        return _parse_float(rate_table[node_type], env_var="GENIESIM_GPU_RATE_TABLE")
    if "default" in rate_table:
        return _parse_float(rate_table["default"], env_var="GENIESIM_GPU_RATE_TABLE")
    return None


def _extract_episode_count(job_metadata: Dict[str, Any]) -> Optional[int]:
    job_metrics = job_metadata.get("job_metrics", {})
    for key in ("episodes_collected", "total_episodes", "episodes_passed"):
        value = job_metrics.get(key)
        if isinstance(value, int) and value > 0:
            return value
    generation_params = job_metadata.get("generation_params", {})
    episodes_per_task = generation_params.get("episodes_per_task")
    num_variations = generation_params.get("num_variations")
    if isinstance(episodes_per_task, int) and isinstance(num_variations, int):
        return max(1, episodes_per_task * num_variations)
    return None


def _extract_duration_seconds(job_metadata: Dict[str, Any]) -> Optional[float]:
    job_metrics = job_metadata.get("job_metrics", {})
    duration = job_metrics.get("duration_seconds")
    if isinstance(duration, (int, float)) and duration > 0:
        return float(duration)
    return None


def _calculate_geniesim_episode_cost(
    *,
    job_metadata: Dict[str, Any],
    rate_table: Dict[str, Any],
) -> Optional[Dict[str, float]]:
    duration_seconds = _extract_duration_seconds(job_metadata)
    episode_count = _extract_episode_count(job_metadata)
    if duration_seconds is None or episode_count is None:
        return None
    region = (
        job_metadata.get("local_execution", {})
        .get("server_info", {})
        .get("region")
    ) or os.getenv(GENIESIM_GPU_REGION_ENV)
    node_type = (
        job_metadata.get("local_execution", {})
        .get("server_info", {})
        .get("node_type")
    ) or os.getenv(GENIESIM_GPU_NODE_TYPE_ENV)
    hourly_rate = _resolve_gpu_hourly_rate(
        rate_table=rate_table,
        region=region,
        node_type=node_type,
    )
    if hourly_rate is None:
        fallback_rate = os.getenv(GENIESIM_GPU_HOURLY_RATE_ENV)
        if fallback_rate:
            hourly_rate = _parse_float(fallback_rate, env_var=GENIESIM_GPU_HOURLY_RATE_ENV)
    if hourly_rate is None:
        return None
    duration_hours = duration_seconds / 3600.0
    total_cost = duration_hours * hourly_rate
    per_episode_cost = total_cost / episode_count if episode_count > 0 else 0.0
    return {
        "duration_seconds": duration_seconds,
        "episode_count": float(episode_count),
        "hourly_rate": hourly_rate,
        "total_cost": total_cost,
        "per_episode_cost": per_episode_cost,
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
        pricing_source: Optional[str] = None,
        enable_logging: bool = True,
    ):
        """Initialize cost tracker.

        Args:
            data_dir: Directory for storing cost data
            custom_pricing: Override default pricing
            pricing_source: Optional description of pricing source
            enable_logging: Whether to log cost tracking
        """
        self.data_dir = Path(data_dir or os.getenv("COST_DATA_DIR", "./cost_data"))
        self.data_dir.mkdir(parents=True, exist_ok=True)

        if custom_pricing and not pricing_source:
            pricing_source = "custom pricing"

        _require_geniesim_env_vars_in_production()

        if _is_production_env() and not custom_pricing:
            raise RuntimeError(
                "Cost tracking defaults are dev-only. Provide a full pricing config via "
                f"{COST_TRACKING_PRICING_ENV} or {COST_TRACKING_PRICING_PATH_ENV} in production."
            )

        self.pricing = DEFAULT_PRICING.copy()
        if custom_pricing:
            _validate_pricing_config(custom_pricing, source=pricing_source or "custom pricing")
            self.pricing.update(custom_pricing)
        geniesim_overrides = _load_geniesim_pricing_from_env()
        if geniesim_overrides:
            self.pricing.update(geniesim_overrides)
            pricing_source = (
                f"{pricing_source} + geniesim env"
                if pricing_source
                else "dev defaults + geniesim env"
            )

        _validate_pricing_values(
            self.pricing,
            source="effective pricing",
            required_fields=REQUIRED_PRICING_FIELDS,
        )
        _validate_geniesim_pricing(self.pricing)

        self.enable_logging = enable_logging
        self.pricing_source = pricing_source or "dev defaults"
        logger.info("Cost tracking pricing source: %s", self.pricing_source)

        # In-memory cache
        self.entries: List[CostEntry] = []
        self.scene_totals: Dict[str, float] = {}
        self.total_cost: float = 0.0
        self.scene_alert_threshold = _parse_optional_threshold(COST_ALERT_PER_SCENE_ENV)
        self.total_alert_threshold = _parse_optional_threshold(COST_ALERT_TOTAL_ENV)
        self.last_scene_alert_totals: Dict[str, float] = {}
        self.last_total_alert_total: float = 0.0

        self.metrics = None
        try:
            from tools.metrics.pipeline_metrics import get_metrics

            self.metrics = get_metrics()
        except Exception as exc:  # pragma: no cover - metrics optional
            logger.debug(f"Metrics unavailable for cost tracking: {exc}")

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
        job_metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Track Genie Sim API cost.

        Args:
            scene_id: Scene identifier
            job_id: Genie Sim job ID
            episode_count: Number of episodes generated
            duration_seconds: Optional job duration
            job_metadata: Optional Genie Sim job metadata payload

        Returns:
            Cost of this job
        """
        # Calculate cost (job base cost + per-episode cost)
        job_cost = self.pricing["geniesim_job"]
        episode_cost = episode_count * self.pricing["geniesim_episode"]
        runtime_pricing: Optional[Dict[str, float]] = None
        if job_metadata:
            rate_table = _load_geniesim_gpu_rate_table()
            runtime_pricing = _calculate_geniesim_episode_cost(
                job_metadata=job_metadata,
                rate_table=rate_table,
            )
        if runtime_pricing:
            runtime_episode_count = int(runtime_pricing["episode_count"])
            episode_cost = runtime_pricing["total_cost"]
            if runtime_episode_count > 0:
                episode_count = runtime_episode_count
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
                "runtime_pricing": runtime_pricing,
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
        self._update_scene_total(scene_id, amount, entry=entry)
        self._save_entry(entry)
        self._emit_cost_metric(entry)

    def _emit_cost_metric(self, entry: CostEntry) -> None:
        try:
            from tools.metrics.pipeline_metrics import get_metrics
        except Exception:
            return

        try:
            metrics = get_metrics()
            metrics.pipeline_cost_total_usd.inc(
                entry.amount,
                labels={
                    "job": f"{entry.category}:{entry.subcategory}",
                    "scene_id": entry.scene_id,
                    "status": "recorded",
                },
            )
        except Exception:
            logger.debug("Failed to emit cost metric for entry %s", entry)

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
                            self._update_scene_total(entry.scene_id, entry.amount, entry=None)
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

    def _update_scene_total(
        self,
        scene_id: str,
        amount: float,
        *,
        entry: Optional[CostEntry],
    ) -> None:
        """Update per-scene totals, emit metrics, and alert on thresholds."""
        self.scene_totals[scene_id] = self.scene_totals.get(scene_id, 0.0) + amount
        self.total_cost += amount
        if self.metrics:
            self.metrics.cost_per_scene.set(
                self.scene_totals[scene_id],
                labels={"job": "cost_tracking", "scene_id": scene_id},
            )
        if entry is None:
            return
        self._maybe_alert_thresholds(scene_id, entry)

    def _maybe_alert_thresholds(self, scene_id: str, entry: CostEntry) -> None:
        if not self.scene_alert_threshold and not self.total_alert_threshold:
            return
        try:
            from monitoring.alerting import send_alert
        except Exception:
            logger.debug("Alerting unavailable; skipping cost alerts.")
            return

        scene_total = self.scene_totals.get(scene_id, 0.0)
        total_cost = self.total_cost
        details = {
            "scene_id": scene_id,
            "category": entry.category,
            "job_name": entry.subcategory,
            "scene_total_usd": scene_total,
            "total_usd": total_cost,
        }

        if self.scene_alert_threshold is not None:
            last_scene_alert = self.last_scene_alert_totals.get(scene_id, 0.0)
            if last_scene_alert < self.scene_alert_threshold <= scene_total:
                send_alert(
                    event_type="cost_tracking.scene_threshold_exceeded",
                    summary=(
                        f"Scene {scene_id} cost ${scene_total:.2f} exceeded "
                        f"threshold ${self.scene_alert_threshold:.2f}"
                    ),
                    details={
                        **details,
                        "scene_threshold_usd": self.scene_alert_threshold,
                    },
                    severity="warning",
                )
                self.last_scene_alert_totals[scene_id] = scene_total

        if self.total_alert_threshold is not None:
            if self.last_total_alert_total < self.total_alert_threshold <= total_cost:
                send_alert(
                    event_type="cost_tracking.total_threshold_exceeded",
                    summary=(
                        f"Total pipeline cost ${total_cost:.2f} exceeded "
                        f"threshold ${self.total_alert_threshold:.2f}"
                    ),
                    details={
                        **details,
                        "total_threshold_usd": self.total_alert_threshold,
                    },
                    severity="warning",
                )
                self.last_total_alert_total = total_cost


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
        custom_pricing, pricing_source = _load_custom_pricing_from_env()
        _cost_tracker = CostTracker(
            custom_pricing=custom_pricing or None,
            pricing_source=pricing_source,
        )
    return _cost_tracker
