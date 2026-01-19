"""GPU cost estimation utilities for BlueprintPipeline."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

DEFAULT_INSTANCE_RATES: Dict[str, Dict[str, float]] = {
    "g5.xlarge": {"hourly_rate": 1.006, "gpu_count": 1},
    "g5.2xlarge": {"hourly_rate": 1.212, "gpu_count": 1},
    "g5.12xlarge": {"hourly_rate": 4.384, "gpu_count": 4},
    "a2-highgpu-1g": {"hourly_rate": 1.685, "gpu_count": 1},
}

BASE_STEP_CONFIG: Dict[str, Dict[str, Any]] = {
    "regen3d": {
        "duration_minutes": 45,
        "instance_type": "g5.xlarge",
    },
    "simready": {
        "duration_minutes": 20,
        "instance_type": "g5.xlarge",
    },
    "replicator": {
        "duration_minutes": 10,
        "instance_type": "g5.xlarge",
    },
    "variation-gen": {
        "duration_minutes": 25,
        "instance_type": "g5.xlarge",
    },
    "genie-sim-export": {
        "duration_minutes": 20,
        "instance_type": "g5.xlarge",
    },
    "genie-sim-import": {
        "duration_minutes": 15,
        "instance_type": "g5.xlarge",
    },
}

DWM_STEP_CONFIG: Dict[str, Dict[str, Any]] = {
    "dwm": {
        "duration_minutes": 30,
        "instance_type": "g5.2xlarge",
    },
    "dwm-inference": {
        "duration_minutes": 120,
        "instance_type": "g5.12xlarge",
    },
}

DREAM2FLOW_STEP_CONFIG: Dict[str, Dict[str, Any]] = {
    "dream2flow": {
        "duration_minutes": 20,
        "instance_type": "g5.2xlarge",
    },
    "dream2flow-inference": {
        "duration_minutes": 90,
        "instance_type": "g5.12xlarge",
    },
}


def _experimental_flags_from_env() -> Dict[str, bool]:
    from tools.config.env import parse_bool_env

    enable_experimental = parse_bool_env(
        os.getenv("ENABLE_EXPERIMENTAL_PIPELINE"),
        default=False,
    ) is True
    enable_dwm = enable_experimental or (
        parse_bool_env(os.getenv("ENABLE_DWM"), default=False) is True
    )
    enable_dream2flow = enable_experimental or (
        parse_bool_env(os.getenv("ENABLE_DREAM2FLOW"), default=False) is True
    )
    return {
        "enable_dwm": enable_dwm,
        "enable_dream2flow": enable_dream2flow,
    }


def build_step_config(
    *,
    include_dwm: bool,
    include_dream2flow: bool,
) -> Dict[str, Dict[str, Any]]:
    step_config = BASE_STEP_CONFIG.copy()
    if include_dwm:
        step_config.update(DWM_STEP_CONFIG)
    if include_dream2flow:
        step_config.update(DREAM2FLOW_STEP_CONFIG)
    return step_config


@dataclass
class EstimateConfig:
    """Configuration for GPU estimation."""

    instance_rates: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: DEFAULT_INSTANCE_RATES.copy()
    )
    step_config: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    default_instance_type: str = "g5.xlarge"

    def __post_init__(self) -> None:
        if not self.step_config:
            flags = _experimental_flags_from_env()
            self.step_config = build_step_config(
                include_dwm=flags["enable_dwm"],
                include_dream2flow=flags["enable_dream2flow"],
            )

    def merge_overrides(self, overrides: Dict[str, Any]) -> None:
        """Merge overrides from JSON configuration."""
        if not overrides:
            return
        rates = overrides.get("rates", {})
        for instance_type, data in rates.items():
            if instance_type in self.instance_rates:
                self.instance_rates[instance_type].update(data)
            else:
                self.instance_rates[instance_type] = data
        steps = overrides.get("steps", {})
        for step_name, data in steps.items():
            if step_name in self.step_config:
                self.step_config[step_name].update(data)
            else:
                self.step_config[step_name] = data
        default_instance = overrides.get("default_instance_type")
        if default_instance:
            self.default_instance_type = default_instance


@dataclass
class StepEstimate:
    """GPU estimate for a single step."""

    step: str
    instance_type: str
    duration_hours: float
    gpu_count: int
    gpu_hours: float
    cost: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "instance_type": self.instance_type,
            "duration_hours": self.duration_hours,
            "gpu_count": self.gpu_count,
            "gpu_hours": self.gpu_hours,
            "cost": self.cost,
        }


@dataclass
class EstimateSummary:
    """Aggregate GPU estimate summary."""

    total_gpu_hours: float
    total_cost: float
    steps: List[StepEstimate]
    missing_steps: List[str]
    missing_rates: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_gpu_hours": self.total_gpu_hours,
            "total_cost": self.total_cost,
            "steps": [step.to_dict() for step in self.steps],
            "missing_steps": self.missing_steps,
            "missing_rates": self.missing_rates,
        }


def _step_duration_hours(step_config: Dict[str, Any]) -> Optional[float]:
    if "duration_hours" in step_config:
        return float(step_config["duration_hours"])
    if "duration_minutes" in step_config:
        return float(step_config["duration_minutes"]) / 60.0
    return None


def estimate_gpu_costs(
    steps: Sequence[str],
    config: EstimateConfig,
) -> EstimateSummary:
    """Estimate GPU-hours and costs for the provided steps."""
    step_estimates: List[StepEstimate] = []
    missing_steps: List[str] = []
    missing_rates: List[str] = []
    total_gpu_hours = 0.0
    total_cost = 0.0

    for step in steps:
        step_config = config.step_config.get(step)
        if not step_config:
            missing_steps.append(step)
            continue

        duration_hours = _step_duration_hours(step_config)
        if duration_hours is None:
            missing_steps.append(step)
            continue

        instance_type = step_config.get("instance_type", config.default_instance_type)
        rate_data = config.instance_rates.get(instance_type)
        if not rate_data:
            missing_rates.append(instance_type)
            continue

        gpu_count = int(rate_data.get("gpu_count", 1))
        hourly_rate = float(rate_data.get("hourly_rate", 0.0))
        gpu_hours = duration_hours * gpu_count
        cost = duration_hours * hourly_rate

        step_estimates.append(
            StepEstimate(
                step=step,
                instance_type=instance_type,
                duration_hours=duration_hours,
                gpu_count=gpu_count,
                gpu_hours=gpu_hours,
                cost=cost,
            )
        )
        total_gpu_hours += gpu_hours
        total_cost += cost

    return EstimateSummary(
        total_gpu_hours=total_gpu_hours,
        total_cost=total_cost,
        steps=step_estimates,
        missing_steps=sorted(set(missing_steps)),
        missing_rates=sorted(set(missing_rates)),
    )


def load_estimate_config(
    config_path: Optional[Path],
    *,
    include_dwm: Optional[bool] = None,
    include_dream2flow: Optional[bool] = None,
) -> EstimateConfig:
    """Load estimate configuration from JSON file."""
    if include_dwm is None or include_dream2flow is None:
        flags = _experimental_flags_from_env()
        include_dwm = flags["enable_dwm"] if include_dwm is None else include_dwm
        include_dream2flow = (
            flags["enable_dream2flow"] if include_dream2flow is None else include_dream2flow
        )

    config = EstimateConfig(
        step_config=build_step_config(
            include_dwm=bool(include_dwm),
            include_dream2flow=bool(include_dream2flow),
        )
    )
    if not config_path:
        return config

    payload = json.loads(Path(config_path).read_text())
    config.merge_overrides(payload)
    return config


def format_estimate_summary(summary: EstimateSummary) -> str:
    """Format estimate summary for console output."""
    lines = [
        "GPU Cost Estimate",
        "-" * 60,
    ]
    for step in summary.steps:
        lines.append(
            f"{step.step:20} {step.gpu_hours:6.2f} GPU-hrs "
            f"on {step.instance_type} = ${step.cost:,.2f}"
        )
    lines.append("-" * 60)
    lines.append(
        f"Total: {summary.total_gpu_hours:.2f} GPU-hrs, "
        f"${summary.total_cost:,.2f}"
    )
    if summary.missing_steps:
        lines.append(f"Missing step configs: {', '.join(summary.missing_steps)}")
    if summary.missing_rates:
        lines.append(f"Missing rate configs: {', '.join(summary.missing_rates)}")
    return "\n".join(lines)


def resolve_steps_for_scene(
    scene_dir: Path,
    steps: Optional[Iterable[str]],
    enable_dwm: bool,
    enable_dream2flow: bool,
) -> List[str]:
    """Resolve pipeline steps for estimation."""
    if steps:
        return [str(step) for step in steps]

    from tools.run_local_pipeline import LocalPipelineRunner

    runner = LocalPipelineRunner(
        scene_dir=scene_dir,
        verbose=False,
        skip_interactive=True,
        environment_type="kitchen",
        enable_dwm=enable_dwm,
        enable_dream2flow=enable_dream2flow,
        disable_articulated_assets=True,
    )
    return [step.value for step in runner._resolve_default_steps()]


def _parse_steps(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate GPU-hours and costs for a local pipeline run",
    )
    parser.add_argument(
        "--scene-dir",
        required=True,
        help="Path to the scene directory (used to resolve default steps)",
    )
    parser.add_argument(
        "--steps",
        type=str,
        help="Comma-separated list of steps to estimate (default: resolved pipeline steps)",
    )
    parser.add_argument(
        "--enable-dwm",
        action="store_true",
        help="Include optional DWM steps when resolving default steps",
    )
    parser.add_argument(
        "--enable-dream2flow",
        action="store_true",
        help="Include optional Dream2Flow steps when resolving default steps",
    )
    parser.add_argument(
        "--enable-experimental",
        action="store_true",
        help="Enable experimental steps (DWM + Dream2Flow)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to JSON configuration for rates and step durations",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON output",
    )

    args = parser.parse_args()
    if args.enable_experimental:
        args.enable_dwm = True
        args.enable_dream2flow = True
    scene_dir = Path(args.scene_dir).resolve()
    steps = _parse_steps(args.steps)
    resolved_steps = resolve_steps_for_scene(
        scene_dir=scene_dir,
        steps=steps,
        enable_dwm=args.enable_dwm,
        enable_dream2flow=args.enable_dream2flow,
    )
    config = load_estimate_config(
        Path(args.config) if args.config else None,
        include_dwm=args.enable_dwm,
        include_dream2flow=args.enable_dream2flow,
    )
    summary = estimate_gpu_costs(resolved_steps, config)

    if args.json:
        print(json.dumps(summary.to_dict(), indent=2))
    else:
        print(format_estimate_summary(summary))


if __name__ == "__main__":
    main()
