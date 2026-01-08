#!/usr/bin/env python3
"""
Upsell Outputs Generator - Integrates all upsell features into default pipeline.

This job generates all the high-value outputs that robotics labs pay for:
1. scene_report.json - Quality metrics for procurement
2. asset_provenance.json - Legal audit trail
3. baseline_benchmarks.json - Success rate baselines
4. PyTorch DataLoaders - Plug-and-play training code

Run this job AFTER episode generation to produce the complete deliverable package.

Output Structure:
    scene_dir/
    ├── quality/
    │   ├── scene_report.json       # Comprehensive QA metrics
    │   └── validation_results.json # Detailed validation
    ├── legal/
    │   └── asset_provenance.json   # License audit trail
    ├── baselines/
    │   └── baseline_benchmarks.json # Expected success rates
    └── dataloaders/
        ├── README.md               # Usage instructions
        └── example_training.py     # Example training script
"""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class UpsellOutputConfig:
    """Configuration for upsell output generation."""

    # Input paths
    scene_dir: Path
    episodes_dir: Optional[Path] = None

    # Which outputs to generate
    generate_scene_report: bool = True
    generate_provenance: bool = True
    generate_baselines: bool = True
    generate_dataloader_examples: bool = True

    # Options
    run_baseline_evaluations: bool = False  # Actually run Isaac Sim
    verbose: bool = True

    def __post_init__(self):
        self.scene_dir = Path(self.scene_dir)
        if self.episodes_dir:
            self.episodes_dir = Path(self.episodes_dir)
        else:
            # Default to scene_dir/episodes
            self.episodes_dir = self.scene_dir / "episodes"


class UpsellOutputGenerator:
    """
    Generates all upsell outputs for a BlueprintPipeline scene.

    This is the main entry point for producing the complete deliverable package
    that justifies premium pricing to robotics labs.
    """

    def __init__(self, config: UpsellOutputConfig):
        self.config = config
        self.scene_id = config.scene_dir.name
        self.results: Dict[str, Any] = {}

    def log(self, msg: str) -> None:
        if self.config.verbose:
            print(f"[UPSELL-GEN] {msg}")

    def generate_all(self) -> Dict[str, Path]:
        """Generate all configured upsell outputs."""
        self.log(f"Generating upsell outputs for: {self.scene_id}")
        self.log(f"Scene directory: {self.config.scene_dir}")

        outputs = {}

        # 1. Scene Quality Report
        if self.config.generate_scene_report:
            self.log("Generating scene quality report...")
            report_path = self._generate_scene_report()
            outputs["scene_report"] = report_path

        # 2. Asset Provenance
        if self.config.generate_provenance:
            self.log("Generating asset provenance...")
            provenance_path = self._generate_provenance()
            outputs["asset_provenance"] = provenance_path

        # 3. Baseline Benchmarks
        if self.config.generate_baselines:
            self.log("Generating baseline benchmarks...")
            baselines_path = self._generate_baselines()
            outputs["baselines"] = baselines_path

        # 4. DataLoader Examples
        if self.config.generate_dataloader_examples:
            self.log("Generating DataLoader examples...")
            dataloader_dir = self._generate_dataloader_examples()
            outputs["dataloaders"] = dataloader_dir

        # 5. Generate summary manifest
        manifest_path = self._generate_upsell_manifest(outputs)
        outputs["manifest"] = manifest_path

        self.log(f"Generated {len(outputs)} upsell outputs")

        return outputs

    def _generate_scene_report(self) -> Path:
        """Generate scene quality report."""
        from tools.quality_reports import generate_scene_report

        output_path = self.config.scene_dir / "quality" / "scene_report.json"

        report = generate_scene_report(
            scene_dir=self.config.scene_dir,
            output_path=output_path,
            scene_id=self.scene_id,
            verbose=self.config.verbose,
        )

        self.results["scene_report"] = {
            "overall_score": report.overall_score,
            "passed": report.passed,
            "physics_score": report.physics_score,
            "asset_score": report.asset_score,
        }

        return output_path

    def _generate_provenance(self) -> Path:
        """Generate asset provenance."""
        from tools.quality_reports import generate_asset_provenance

        output_path = self.config.scene_dir / "legal" / "asset_provenance.json"

        provenance = generate_asset_provenance(
            scene_dir=self.config.scene_dir,
            output_path=output_path,
            scene_id=self.scene_id,
            verbose=self.config.verbose,
        )

        self.results["provenance"] = {
            "total_assets": provenance.total_assets,
            "commercial_ok": provenance.commercial_use_ok,
            "commercial_ok_count": provenance.commercial_ok_assets,
            "blockers": len(provenance.commercial_blockers),
        }

        return output_path

    def _generate_baselines(self) -> Path:
        """Generate baseline benchmarks."""
        from tools.baselines import generate_scene_baselines

        output_path = self.config.scene_dir / "baselines" / "baseline_benchmarks.json"

        baselines = generate_scene_baselines(
            scene_dir=self.config.scene_dir,
            output_path=output_path,
            scene_id=self.scene_id,
            run_evaluations=self.config.run_baseline_evaluations,
            verbose=self.config.verbose,
        )

        self.results["baselines"] = {
            "num_tasks": len(baselines.tasks),
            "scripted_success": baselines.overall_scripted_success,
            "heuristic_success": baselines.overall_heuristic_success,
            "pretrained_success": baselines.overall_pretrained_success,
        }

        return output_path

    def _generate_dataloader_examples(self) -> Path:
        """Generate DataLoader usage examples."""
        dataloader_dir = self.config.scene_dir / "dataloaders"
        dataloader_dir.mkdir(parents=True, exist_ok=True)

        # Copy PyTorch DataLoader module
        source_dataloader = REPO_ROOT / "episode-generation-job" / "pytorch_dataloaders.py"
        if source_dataloader.exists():
            shutil.copy(source_dataloader, dataloader_dir / "pytorch_dataloaders.py")

        # Generate README
        readme_content = f"""# PyTorch DataLoaders for {self.scene_id}

## Quick Start

```python
from pytorch_dataloaders import create_blueprint_dataloader

# Create training DataLoader with defaults
train_loader = create_blueprint_dataloader(
    "{self.config.episodes_dir}",
    split="train"
)

# Training loop
for batch in train_loader:
    images = batch["observation.images.wrist"]  # (B, T, C, H, W)
    state = batch["observation.state"]           # (B, T, state_dim)
    actions = batch["action"]                    # (B, chunk_size, action_dim)

    # Your training code here
    loss = model(images, state, actions)
    loss.backward()
```

## Custom Configuration

```python
from pytorch_dataloaders import create_blueprint_dataloader, DataLoadingConfig

config = DataLoadingConfig(
    cameras=["wrist", "overhead"],  # Multiple cameras
    image_size=(224, 224),          # Image resolution
    chunk_size=16,                  # Action chunk for Diffusion Policy
    batch_size=32,
    include_language=True,          # Include language instructions
)

train_loader = create_blueprint_dataloader(
    "{self.config.episodes_dir}",
    config=config,
    split="train"
)
```

## Available Data Fields

| Field | Shape | Description |
|-------|-------|-------------|
| `observation.images.<camera>` | (B, T, C, H, W) | RGB images |
| `observation.state` | (B, T, state_dim) | Robot state |
| `action` | (B, chunk_size, action_dim) | Actions to predict |
| `observation.depth.<camera>` | (B, H, W) | Depth (if enabled) |
| `language` | List[str] | Language instructions (if enabled) |

## Compatibility

This DataLoader is compatible with:
- Diffusion Policy
- ACT (Action Chunking Transformer)
- OpenVLA / Pi0 / SmolVLA
- Any PyTorch-based policy

## Support

Email: support@tryblueprint.io
"""

        readme_path = dataloader_dir / "README.md"
        readme_path.write_text(readme_content)

        # Generate example training script
        example_script = f'''#!/usr/bin/env python3
"""
Example Training Script for {self.scene_id}

This script demonstrates how to use BlueprintPipeline data
with a simple policy network.
"""

import torch
import torch.nn as nn
from pathlib import Path

# Import DataLoader
from pytorch_dataloaders import create_blueprint_dataloader, DataLoadingConfig


class SimplePolicy(nn.Module):
    """Simple MLP policy for demonstration."""

    def __init__(self, state_dim: int = 7, action_dim: int = 7, hidden_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # state: (B, T, state_dim) -> use last timestep
        x = state[:, -1]  # (B, state_dim)
        x = self.encoder(x)
        return self.action_head(x)


def main():
    # Configuration
    dataset_path = Path("{self.config.episodes_dir}")

    config = DataLoadingConfig(
        cameras=["wrist"],
        chunk_size=16,
        batch_size=32,
        num_workers=4,
    )

    # Create DataLoader
    print("Creating DataLoader...")
    train_loader = create_blueprint_dataloader(
        dataset_path,
        config=config,
        split="train"
    )
    print(f"Dataset: {{len(train_loader.dataset)}} samples")

    # Create model
    model = SimplePolicy()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    # Training loop
    print("Starting training...")
    for epoch in range(10):
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            state = batch["observation.state"]
            actions = batch["action"]

            # Forward pass
            predicted = model(state)
            target = actions[:, 0]  # First action in chunk

            loss = loss_fn(predicted, target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {{epoch}}, Batch {{batch_idx}}, Loss: {{loss.item():.4f}}")

        print(f"Epoch {{epoch}} complete. Avg Loss: {{total_loss / len(train_loader):.4f}}")

    print("Training complete!")


if __name__ == "__main__":
    main()
'''

        example_path = dataloader_dir / "example_training.py"
        example_path.write_text(example_script)

        self.results["dataloaders"] = {
            "readme": str(readme_path),
            "example_script": str(example_path),
        }

        return dataloader_dir

    def _generate_upsell_manifest(self, outputs: Dict[str, Path]) -> Path:
        """Generate manifest of all upsell outputs."""
        manifest = {
            "scene_id": self.scene_id,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "pipeline_version": "1.0.0",
            "outputs": {
                name: str(path) for name, path in outputs.items()
            },
            "results": self.results,
            "upsell_value": self._calculate_upsell_value(),
        }

        manifest_path = self.config.scene_dir / "upsell_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        return manifest_path

    def _calculate_upsell_value(self) -> Dict[str, Any]:
        """Calculate the upsell value of generated outputs."""
        # Pricing estimates based on analysis document
        value = {
            "scene_report": 500,  # QA documentation
            "provenance": 1000,  # Legal clearance
            "baselines": 2000,  # Benchmark/eval harness
            "dataloaders": 500,  # Training-ready data
        }

        total = sum(v for k, v in value.items() if k in self.results)

        # Premium if commercially cleared
        if self.results.get("provenance", {}).get("commercial_ok"):
            value["commercial_clearance_premium"] = 2000
            total += 2000

        # Premium if high quality score
        scene_score = self.results.get("scene_report", {}).get("overall_score", 0)
        if scene_score >= 80:
            value["quality_premium"] = 1000
            total += 1000

        value["total_estimated_value"] = total
        value["currency"] = "USD"

        return value


def generate_upsell_outputs(
    scene_dir: Path,
    episodes_dir: Optional[Path] = None,
    run_evaluations: bool = False,
    verbose: bool = True,
) -> Dict[str, Path]:
    """
    Convenience function to generate all upsell outputs.

    Args:
        scene_dir: Path to scene directory
        episodes_dir: Optional path to episodes (defaults to scene_dir/episodes)
        run_evaluations: Whether to run actual Isaac Sim evaluations
        verbose: Print progress

    Returns:
        Dictionary of output names to paths
    """
    config = UpsellOutputConfig(
        scene_dir=scene_dir,
        episodes_dir=episodes_dir,
        run_baseline_evaluations=run_evaluations,
        verbose=verbose,
    )

    generator = UpsellOutputGenerator(config)
    return generator.generate_all()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate upsell outputs")
    parser.add_argument("scene_dir", type=Path, help="Path to scene directory")
    parser.add_argument("--episodes-dir", type=Path, help="Path to episodes directory")
    parser.add_argument("--run-eval", action="store_true", help="Run Isaac Sim evaluations")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")

    args = parser.parse_args()

    outputs = generate_upsell_outputs(
        scene_dir=args.scene_dir,
        episodes_dir=args.episodes_dir,
        run_evaluations=args.run_eval,
        verbose=not args.quiet,
    )

    print("\nGenerated Upsell Outputs:")
    print("=" * 50)
    for name, path in outputs.items():
        print(f"  {name}: {path}")
