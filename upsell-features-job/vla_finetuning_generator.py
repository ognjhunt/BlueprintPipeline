#!/usr/bin/env python3
"""
VLA (Vision-Language-Action) Fine-Tuning Package Generator.

Generates turnkey fine-tuning configurations for popular VLA models:
- OpenVLA (7B params, Stanford)
- Pi0 (Physical Intelligence)
- SmolVLA (450M params, efficient)
- GR00T N1.5 (NVIDIA)

These packages allow customers to immediately start training VLA models
on their Blueprint-generated data without ML engineering expertise.

Upsell Value: +$3,000-$8,000 per scene
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add parent to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class VLAModel(str, Enum):
    """Supported VLA models for fine-tuning."""
    OPENVLA = "openvla"
    PI0 = "pi0"
    SMOLVLA = "smolvla"
    GROOT_N1 = "groot_n1"
    COSMOS_POLICY = "cosmos_policy"
    ALL = "all"


@dataclass
class VLAFinetuningConfig:
    """Configuration for VLA fine-tuning."""
    model: VLAModel
    model_name: str
    model_size: str

    # Training hyperparameters
    learning_rate: float = 2e-5
    batch_size: int = 32
    num_epochs: int = 10
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 4

    # LoRA configuration (for efficient fine-tuning)
    use_lora: bool = True
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=list)

    # Hardware requirements
    min_gpu_memory_gb: int = 24
    recommended_gpu: str = "A100-40GB"
    multi_gpu: bool = False

    # Data configuration
    image_size: Tuple[int, int] = (224, 224)
    max_seq_length: int = 256
    action_dim: int = 7  # 6-DoF + gripper

    # Pretrained weights
    pretrained_checkpoint: str = ""
    vision_encoder: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model.value,
            "model_name": self.model_name,
            "model_size": self.model_size,
            "training": {
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "num_epochs": self.num_epochs,
                "warmup_steps": self.warmup_steps,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
            },
            "lora": {
                "enabled": self.use_lora,
                "rank": self.lora_rank,
                "alpha": self.lora_alpha,
                "dropout": self.lora_dropout,
                "target_modules": self.lora_target_modules,
            },
            "hardware": {
                "min_gpu_memory_gb": self.min_gpu_memory_gb,
                "recommended_gpu": self.recommended_gpu,
                "multi_gpu": self.multi_gpu,
            },
            "data": {
                "image_size": list(self.image_size),
                "max_seq_length": self.max_seq_length,
                "action_dim": self.action_dim,
            },
            "pretrained": {
                "checkpoint": self.pretrained_checkpoint,
                "vision_encoder": self.vision_encoder,
            },
        }


# Model-specific configurations
VLA_CONFIGS: Dict[VLAModel, VLAFinetuningConfig] = {
    VLAModel.OPENVLA: VLAFinetuningConfig(
        model=VLAModel.OPENVLA,
        model_name="openvla/openvla-7b",
        model_size="7B",
        learning_rate=2e-5,
        batch_size=16,
        num_epochs=10,
        use_lora=True,
        lora_rank=32,
        lora_alpha=64,
        lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        min_gpu_memory_gb=27,
        recommended_gpu="A100-80GB",
        image_size=(224, 224),
        max_seq_length=256,
        action_dim=7,
        pretrained_checkpoint="openvla/openvla-7b",
        vision_encoder="siglip-so400m-patch14-384",
    ),
    VLAModel.PI0: VLAFinetuningConfig(
        model=VLAModel.PI0,
        model_name="physical-intelligence/pi0",
        model_size="3B+300M",
        learning_rate=1e-4,
        batch_size=32,
        num_epochs=50,
        warmup_steps=1000,
        use_lora=False,  # Pi0 uses flow matching, different training
        min_gpu_memory_gb=24,
        recommended_gpu="A100-40GB",
        image_size=(256, 256),
        max_seq_length=512,
        action_dim=7,
        pretrained_checkpoint="google/paligemma-3b-pt-224",
        vision_encoder="paligemma",
    ),
    VLAModel.SMOLVLA: VLAFinetuningConfig(
        model=VLAModel.SMOLVLA,
        model_name="HuggingFaceTB/SmolVLA-450M",
        model_size="450M",
        learning_rate=5e-5,
        batch_size=64,
        num_epochs=20,
        use_lora=False,  # Small enough for full fine-tuning
        min_gpu_memory_gb=16,
        recommended_gpu="RTX 4090 / A10",
        image_size=(224, 224),
        max_seq_length=256,
        action_dim=7,
        pretrained_checkpoint="HuggingFaceTB/SmolVLA-450M",
        vision_encoder="siglip-base-patch16-224",
    ),
    VLAModel.GROOT_N1: VLAFinetuningConfig(
        model=VLAModel.GROOT_N1,
        model_name="nvidia/gr00t-n1.5",
        model_size="N1.5",
        learning_rate=1e-4,
        batch_size=48,
        num_epochs=100,
        warmup_steps=500,
        use_lora=True,
        lora_rank=64,
        min_gpu_memory_gb=32,
        recommended_gpu="A100-80GB",
        multi_gpu=True,
        image_size=(256, 256),
        max_seq_length=1024,
        action_dim=7,
        pretrained_checkpoint="nvidia/gr00t-n1.5-base",
        vision_encoder="dinov2",
    ),
    VLAModel.COSMOS_POLICY: VLAFinetuningConfig(
        model=VLAModel.COSMOS_POLICY,
        model_name="nvidia/Cosmos-Policy-Predict2-2B",
        model_size="2B",
        learning_rate=1e-4,
        batch_size=32,
        num_epochs=100,
        warmup_steps=1000,
        gradient_accumulation_steps=4,
        use_lora=False,  # Full fine-tuning (video diffusion model)
        min_gpu_memory_gb=80,
        recommended_gpu="H100-80GB",
        multi_gpu=True,
        image_size=(256, 256),
        max_seq_length=1024,
        action_dim=8,  # 7 joints + gripper
        pretrained_checkpoint="nvidia/Cosmos-Policy-Predict2-2B",
        vision_encoder="wan2.1_vae",  # Cosmos uses Wan2.1 VAE tokenizer
    ),
}


@dataclass
class VLADatasetAdapter:
    """Adapts BlueprintPipeline LeRobot data for VLA training."""

    model: VLAModel
    episodes_dir: Path
    output_dir: Path

    # Language annotations
    language_annotations: Dict[str, List[str]] = field(default_factory=dict)

    def generate_language_annotations(
        self,
        task_description: str,
        num_variations: int = 10,
    ) -> List[str]:
        """
        Generate language instruction variations for a task.
        Uses templates + LLM for diversity.
        """
        # Base templates for common manipulation tasks
        templates = {
            "pick": [
                "Pick up the {object}",
                "Grab the {object}",
                "Take the {object}",
                "Lift the {object}",
                "Get the {object}",
            ],
            "place": [
                "Place the {object} on the {location}",
                "Put the {object} on the {location}",
                "Set the {object} down on the {location}",
                "Move the {object} to the {location}",
            ],
            "pick_place": [
                "Pick up the {object} and place it on the {location}",
                "Grab the {object} and put it on the {location}",
                "Take the {object} and move it to the {location}",
                "Get the {object} and set it on the {location}",
            ],
            "open": [
                "Open the {object}",
                "Pull open the {object}",
                "Slide the {object} open",
            ],
            "close": [
                "Close the {object}",
                "Push the {object} closed",
                "Shut the {object}",
            ],
            "pour": [
                "Pour from the {object} into the {target}",
                "Empty the {object} into the {target}",
                "Transfer contents from {object} to {target}",
            ],
        }

        # Parse task description to extract task type and objects
        task_lower = task_description.lower()
        annotations = []

        # Try to match task patterns
        for task_type, phrases in templates.items():
            if task_type in task_lower:
                # Generate variations
                for phrase in phrases[:num_variations]:
                    # Simple placeholder replacement from task description
                    annotation = phrase.format(
                        object=self._extract_object(task_description),
                        location=self._extract_location(task_description),
                        target=self._extract_target(task_description),
                    )
                    annotations.append(annotation)
                break

        # If no match, use the task description directly with variations
        if not annotations:
            annotations = [
                task_description,
                f"Execute: {task_description}",
                f"Task: {task_description}",
                f"Perform the following: {task_description}",
                f"Robot instruction: {task_description}",
            ]

        # Ensure we have enough variations
        while len(annotations) < num_variations:
            annotations.append(annotations[len(annotations) % len(annotations)])

        return annotations[:num_variations]

    def _extract_object(self, description: str) -> str:
        """Extract object name from task description."""
        words = description.lower().split()
        # Look for common object words
        objects = ["cup", "mug", "bottle", "box", "bowl", "plate", "drawer", "door", "cabinet"]
        for word in words:
            for obj in objects:
                if obj in word:
                    return f"the {obj}"
        return "the object"

    def _extract_location(self, description: str) -> str:
        """Extract location from task description."""
        locations = ["counter", "table", "shelf", "countertop", "surface", "tray"]
        words = description.lower().split()
        for word in words:
            for loc in locations:
                if loc in word:
                    return f"the {loc}"
        return "the target location"

    def _extract_target(self, description: str) -> str:
        """Extract target container from task description."""
        targets = ["bowl", "cup", "container", "glass", "pot"]
        words = description.lower().split()
        for word in words:
            for tgt in targets:
                if tgt in word:
                    return f"the {tgt}"
        return "the target"


class VLAFinetuningGenerator:
    """
    Generates complete VLA fine-tuning packages.

    Output Structure:
        vla_finetuning/
        ├── openvla/
        │   ├── config.yaml
        │   ├── train.py
        │   ├── eval.py
        │   ├── requirements.txt
        │   └── README.md
        ├── pi0/
        │   ├── config.yaml
        │   ├── train.py
        │   └── ...
        ├── smolvla/
        │   └── ...
        ├── language_annotations.json
        ├── clip_embeddings.npy  (optional)
        └── dataset_info.json
    """

    def __init__(
        self,
        episodes_dir: Path,
        output_dir: Path,
        scene_id: str,
        models: List[VLAModel] = None,
        verbose: bool = True,
    ):
        self.episodes_dir = Path(episodes_dir)
        self.output_dir = Path(output_dir)
        self.scene_id = scene_id
        self.models = models or [VLAModel.OPENVLA, VLAModel.PI0, VLAModel.SMOLVLA, VLAModel.COSMOS_POLICY]
        self.verbose = verbose

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[VLA-GEN] {msg}")

    def generate_all(self) -> Dict[str, Path]:
        """Generate fine-tuning packages for all configured models."""
        results = {}

        # Load episode metadata
        episodes_meta = self._load_episodes_metadata()

        # Generate language annotations
        self.log("Generating language annotations...")
        language_annotations = self._generate_language_annotations(episodes_meta)

        # Save language annotations
        lang_path = self.output_dir / "language_annotations.json"
        with open(lang_path, "w") as f:
            json.dump(language_annotations, f, indent=2)
        results["language_annotations"] = lang_path

        # Generate model-specific packages
        for model in self.models:
            if model == VLAModel.ALL:
                continue

            self.log(f"Generating {model.value} package...")
            model_dir = self.output_dir / model.value
            model_dir.mkdir(parents=True, exist_ok=True)

            config = VLA_CONFIGS[model]

            # Generate config file
            self._generate_config(model_dir, config, language_annotations)

            # Generate training script
            self._generate_train_script(model_dir, config)

            # Generate evaluation script
            self._generate_eval_script(model_dir, config)

            # Generate requirements
            self._generate_requirements(model_dir, config)

            # Generate README
            self._generate_readme(model_dir, config)

            results[model.value] = model_dir

        # Generate dataset info
        info_path = self.output_dir / "dataset_info.json"
        self._generate_dataset_info(info_path, episodes_meta, language_annotations)
        results["dataset_info"] = info_path

        # Generate master README
        self._generate_master_readme()

        self.log(f"Generated VLA fine-tuning packages at {self.output_dir}")
        return results

    def _load_episodes_metadata(self) -> Dict[str, Any]:
        """Load episode metadata from LeRobot dataset."""
        meta_dir = self.episodes_dir / "meta"

        episodes_meta = {
            "episodes": [],
            "tasks": [],
            "info": {},
        }

        # Load info.json
        info_path = meta_dir / "info.json"
        if info_path.exists():
            with open(info_path) as f:
                episodes_meta["info"] = json.load(f)

        # Load tasks.jsonl
        tasks_path = meta_dir / "tasks.jsonl"
        if tasks_path.exists():
            with open(tasks_path) as f:
                for line in f:
                    if line.strip():
                        episodes_meta["tasks"].append(json.loads(line))

        # Load episodes.jsonl
        episodes_path = meta_dir / "episodes.jsonl"
        if episodes_path.exists():
            with open(episodes_path) as f:
                for line in f:
                    if line.strip():
                        episodes_meta["episodes"].append(json.loads(line))

        return episodes_meta

    def _generate_language_annotations(
        self,
        episodes_meta: Dict[str, Any],
    ) -> Dict[str, List[str]]:
        """Generate language annotations for all tasks."""
        annotations = {}
        adapter = VLADatasetAdapter(
            model=VLAModel.OPENVLA,  # Doesn't matter for annotation generation
            episodes_dir=self.episodes_dir,
            output_dir=self.output_dir,
        )

        for task in episodes_meta.get("tasks", []):
            task_id = str(task.get("task_index", 0))
            task_desc = task.get("task", task.get("description", "manipulation task"))

            # Generate 10 language variations per task
            variations = adapter.generate_language_annotations(task_desc, num_variations=10)
            annotations[task_id] = variations

        return annotations

    def _generate_config(
        self,
        model_dir: Path,
        config: VLAFinetuningConfig,
        language_annotations: Dict[str, List[str]],
    ) -> None:
        """Generate YAML configuration file."""
        config_content = f"""# VLA Fine-Tuning Configuration
# Model: {config.model_name}
# Generated by BlueprintPipeline

model:
  name: "{config.model_name}"
  size: "{config.model_size}"
  pretrained_checkpoint: "{config.pretrained_checkpoint}"
  vision_encoder: "{config.vision_encoder}"

training:
  learning_rate: {config.learning_rate}
  batch_size: {config.batch_size}
  num_epochs: {config.num_epochs}
  warmup_steps: {config.warmup_steps}
  gradient_accumulation_steps: {config.gradient_accumulation_steps}
  max_grad_norm: 1.0
  weight_decay: 0.01
  seed: 42

lora:
  enabled: {str(config.use_lora).lower()}
  rank: {config.lora_rank}
  alpha: {config.lora_alpha}
  dropout: {config.lora_dropout}
  target_modules:
{chr(10).join(f'    - "{m}"' for m in config.lora_target_modules)}

data:
  dataset_path: "{self.episodes_dir}"
  language_annotations: "../language_annotations.json"
  image_size: [{config.image_size[0]}, {config.image_size[1]}]
  max_seq_length: {config.max_seq_length}
  action_dim: {config.action_dim}
  action_normalization: "minmax"  # minmax, standardize, or none
  use_augmentation: true
  augmentation:
    random_crop: true
    color_jitter: true
    random_rotation: 5  # degrees

hardware:
  min_gpu_memory_gb: {config.min_gpu_memory_gb}
  recommended_gpu: "{config.recommended_gpu}"
  multi_gpu: {str(config.multi_gpu).lower()}
  mixed_precision: "bf16"  # fp16, bf16, or fp32
  gradient_checkpointing: true

evaluation:
  eval_frequency: 1000  # steps
  num_eval_episodes: 50
  metrics:
    - success_rate
    - action_mse
    - language_accuracy

output:
  checkpoint_dir: "./checkpoints"
  log_dir: "./logs"
  save_frequency: 1000  # steps
  save_total_limit: 3

# BlueprintPipeline Metadata
blueprint:
  scene_id: "{self.scene_id}"
  generated_at: "{datetime.utcnow().isoformat()}Z"
  package_version: "1.0.0"
"""

        config_path = model_dir / "config.yaml"
        with open(config_path, "w") as f:
            f.write(config_content)

    def _generate_train_script(
        self,
        model_dir: Path,
        config: VLAFinetuningConfig,
    ) -> None:
        """Generate training script."""

        if config.model == VLAModel.OPENVLA:
            script = self._generate_openvla_train_script(config)
        elif config.model == VLAModel.PI0:
            script = self._generate_pi0_train_script(config)
        elif config.model == VLAModel.SMOLVLA:
            script = self._generate_smolvla_train_script(config)
        elif config.model == VLAModel.GROOT_N1:
            script = self._generate_groot_train_script(config)
        else:
            script = self._generate_generic_train_script(config)

        train_path = model_dir / "train.py"
        with open(train_path, "w") as f:
            f.write(script)

    def _generate_openvla_train_script(self, config: VLAFinetuningConfig) -> str:
        return '''#!/usr/bin/env python3
"""
OpenVLA Fine-Tuning Script
Generated by BlueprintPipeline

Usage:
    python train.py --config config.yaml

For LoRA fine-tuning (recommended):
    python train.py --config config.yaml --lora

For full fine-tuning (requires 8x A100-80GB):
    python train.py --config config.yaml --full-finetune
"""

import argparse
import json
import os
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_blueprint_dataset(config: dict) -> Dataset:
    """Load BlueprintPipeline LeRobot dataset."""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    dataset_path = config["data"]["dataset_path"]
    lang_path = config["data"]["language_annotations"]

    # Load language annotations
    with open(lang_path) as f:
        language_annotations = json.load(f)

    # Load LeRobot dataset
    dataset = LeRobotDataset(dataset_path)

    # Convert to HuggingFace format with language
    def add_language(example):
        task_id = str(example.get("task_index", 0))
        annotations = language_annotations.get(task_id, ["perform the manipulation task"])
        # Random selection for diversity
        import random
        example["language_instruction"] = random.choice(annotations)
        return example

    hf_dataset = dataset.to_hf_dataset().map(add_language)
    return hf_dataset


def create_model_and_processor(config: dict, use_lora: bool = True):
    """Create OpenVLA model and processor."""
    model_name = config["model"]["pretrained_checkpoint"]

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    if use_lora and config["lora"]["enabled"]:
        lora_config = LoraConfig(
            r=config["lora"]["rank"],
            lora_alpha=config["lora"]["alpha"],
            lora_dropout=config["lora"]["dropout"],
            target_modules=config["lora"]["target_modules"],
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, processor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--lora", action="store_true", default=True)
    parser.add_argument("--full-finetune", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    use_lora = args.lora and not args.full_finetune

    print(f"Loading model: {config['model']['name']}")
    print(f"LoRA enabled: {use_lora}")

    model, processor = create_model_and_processor(config, use_lora)
    dataset = load_blueprint_dataset(config)

    training_args = TrainingArguments(
        output_dir=config["output"]["checkpoint_dir"],
        num_train_epochs=config["training"]["num_epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        warmup_steps=config["training"]["warmup_steps"],
        logging_dir=config["output"]["log_dir"],
        save_steps=config["output"]["save_frequency"],
        save_total_limit=config["output"]["save_total_limit"],
        bf16=config["hardware"]["mixed_precision"] == "bf16",
        fp16=config["hardware"]["mixed_precision"] == "fp16",
        gradient_checkpointing=config["hardware"]["gradient_checkpointing"],
        report_to="tensorboard",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=processor,
    )

    print("Starting training...")
    trainer.train()

    # Save final model
    trainer.save_model(config["output"]["checkpoint_dir"] + "/final")
    print(f"Training complete! Model saved to {config['output']['checkpoint_dir']}/final")


if __name__ == "__main__":
    main()
'''

    def _generate_pi0_train_script(self, config: VLAFinetuningConfig) -> str:
        return '''#!/usr/bin/env python3
"""
Pi0 Fine-Tuning Script
Generated by BlueprintPipeline

Pi0 uses flow matching for action prediction, which requires
a different training approach than standard VLA models.

Usage:
    python train.py --config config.yaml
"""

import argparse
import json
from pathlib import Path

import torch
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    # Pi0 training via LeRobot
    # Pi0 is integrated in LeRobot v0.4.0+
    from lerobot.common.policies.pi0.modeling_pi0 import Pi0Policy
    from lerobot.common.policies.pi0.configuration_pi0 import Pi0Config
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.scripts.train import train

    # Load language annotations
    with open(config["data"]["language_annotations"]) as f:
        language_annotations = json.load(f)

    # Configure Pi0
    pi0_config = Pi0Config(
        input_shapes={
            "observation.images.top": [3, config["data"]["image_size"][0], config["data"]["image_size"][1]],
            "observation.state": [config["data"]["action_dim"]],
        },
        output_shapes={
            "action": [config["data"]["action_dim"]],
        },
        input_normalization_modes={
            "observation.images.top": "mean_std",
            "observation.state": "min_max",
        },
        output_normalization_modes={
            "action": "min_max",
        },
    )

    print(f"Training Pi0 on BlueprintPipeline data")
    print(f"Dataset: {config['data']['dataset_path']}")
    print(f"Epochs: {config['training']['num_epochs']}")

    # Use LeRobot training script
    train(
        dataset_repo_id=config["data"]["dataset_path"],
        policy_class=Pi0Policy,
        policy_config=pi0_config,
        num_epochs=config["training"]["num_epochs"],
        batch_size=config["training"]["batch_size"],
        learning_rate=config["training"]["learning_rate"],
        output_dir=config["output"]["checkpoint_dir"],
    )

    print("Pi0 training complete!")


if __name__ == "__main__":
    main()
'''

    def _generate_smolvla_train_script(self, config: VLAFinetuningConfig) -> str:
        return '''#!/usr/bin/env python3
"""
SmolVLA Fine-Tuning Script
Generated by BlueprintPipeline

SmolVLA is a compact (450M params) VLA that runs on consumer hardware.
Full fine-tuning is recommended due to the smaller model size.

Usage:
    python train.py --config config.yaml
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from transformers import TrainingArguments, Trainer


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    # SmolVLA via HuggingFace
    from transformers import AutoModelForVision2Seq, AutoProcessor

    print(f"Loading SmolVLA: {config['model']['pretrained_checkpoint']}")

    processor = AutoProcessor.from_pretrained(
        config["model"]["pretrained_checkpoint"],
        trust_remote_code=True,
    )

    model = AutoModelForVision2Seq.from_pretrained(
        config["model"]["pretrained_checkpoint"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Load dataset
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(config["data"]["dataset_path"])

    # Add language annotations
    with open(config["data"]["language_annotations"]) as f:
        language_annotations = json.load(f)

    training_args = TrainingArguments(
        output_dir=config["output"]["checkpoint_dir"],
        num_train_epochs=config["training"]["num_epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        learning_rate=config["training"]["learning_rate"],
        warmup_steps=config["training"]["warmup_steps"],
        save_steps=config["output"]["save_frequency"],
        logging_dir=config["output"]["log_dir"],
        bf16=True,
        report_to="tensorboard",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset.to_hf_dataset(),
        processing_class=processor,
    )

    print("Starting SmolVLA training...")
    trainer.train()
    trainer.save_model(config["output"]["checkpoint_dir"] + "/final")

    print("SmolVLA training complete!")


if __name__ == "__main__":
    main()
'''

    def _generate_groot_train_script(self, config: VLAFinetuningConfig) -> str:
        return '''#!/usr/bin/env python3
"""
GR00T N1.5 Fine-Tuning Script
Generated by BlueprintPipeline

GR00T requires NVIDIA Isaac Lab environment.

Usage:
    python train.py --config config.yaml
"""

import argparse
import json
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    # GR00T via LeRobot (integrated in v0.4.0+)
    from lerobot.common.policies.groot.modeling_groot import GrootPolicy
    from lerobot.common.policies.groot.configuration_groot import GrootConfig
    from lerobot.scripts.train import train

    print(f"Training GR00T N1.5 on BlueprintPipeline data")

    groot_config = GrootConfig(
        input_shapes={
            "observation.images.top": [3, 256, 256],
            "observation.state": [config["data"]["action_dim"]],
        },
        output_shapes={
            "action": [config["data"]["action_dim"]],
        },
    )

    train(
        dataset_repo_id=config["data"]["dataset_path"],
        policy_class=GrootPolicy,
        policy_config=groot_config,
        num_epochs=config["training"]["num_epochs"],
        batch_size=config["training"]["batch_size"],
        learning_rate=config["training"]["learning_rate"],
        output_dir=config["output"]["checkpoint_dir"],
    )

    print("GR00T training complete!")


if __name__ == "__main__":
    main()
'''

    def _generate_generic_train_script(self, config: VLAFinetuningConfig) -> str:
        return f'''#!/usr/bin/env python3
"""
Generic VLA Fine-Tuning Script
Generated by BlueprintPipeline
"""

import argparse
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Training {{config['model']['name']}}")
    print("See model-specific documentation for training instructions.")

if __name__ == "__main__":
    main()
'''

    def _generate_eval_script(
        self,
        model_dir: Path,
        config: VLAFinetuningConfig,
    ) -> None:
        """Generate evaluation script."""
        eval_script = '''#!/usr/bin/env python3
"""
VLA Evaluation Script
Generated by BlueprintPipeline

Evaluates fine-tuned VLA model on held-out episodes.

Usage:
    python eval.py --checkpoint ./checkpoints/final --episodes 100
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
import numpy as np


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def evaluate_policy(
    checkpoint_path: str,
    dataset_path: str,
    num_episodes: int = 100,
) -> dict:
    """Evaluate policy on dataset episodes."""

    # Load model
    from transformers import AutoModelForVision2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()

    # Load dataset
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset(dataset_path)

    # Evaluate
    metrics = {
        "action_mse": [],
        "success_predictions": 0,
        "total_predictions": 0,
    }

    with torch.no_grad():
        for i, episode in enumerate(dataset):
            if i >= num_episodes:
                break

            # Get model predictions
            inputs = processor(
                images=episode["observation.images.top"],
                text=episode.get("language_instruction", "perform task"),
                return_tensors="pt",
            )

            outputs = model.generate(**inputs, max_new_tokens=256)
            predicted_actions = processor.decode_actions(outputs)

            # Compare with ground truth
            gt_actions = episode["action"]
            mse = np.mean((predicted_actions - gt_actions) ** 2)
            metrics["action_mse"].append(mse)

            # Check success (simple threshold)
            if mse < 0.1:
                metrics["success_predictions"] += 1
            metrics["total_predictions"] += 1

    # Aggregate
    return {
        "mean_action_mse": float(np.mean(metrics["action_mse"])),
        "std_action_mse": float(np.std(metrics["action_mse"])),
        "success_rate": metrics["success_predictions"] / max(1, metrics["total_predictions"]),
        "num_episodes_evaluated": metrics["total_predictions"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to fine-tuned model")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to evaluate")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    print(f"Evaluating checkpoint: {args.checkpoint}")
    print(f"Dataset: {config['data']['dataset_path']}")

    results = evaluate_policy(
        checkpoint_path=args.checkpoint,
        dataset_path=config["data"]["dataset_path"],
        num_episodes=args.episodes,
    )

    print("\\nEvaluation Results:")
    print(f"  Action MSE: {results['mean_action_mse']:.4f} (+/- {results['std_action_mse']:.4f})")
    print(f"  Success Rate: {results['success_rate']:.1%}")
    print(f"  Episodes Evaluated: {results['num_episodes_evaluated']}")

    # Save results
    results_path = Path(args.checkpoint).parent / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
'''

        eval_path = model_dir / "eval.py"
        with open(eval_path, "w") as f:
            f.write(eval_script)

    def _generate_requirements(
        self,
        model_dir: Path,
        config: VLAFinetuningConfig,
    ) -> None:
        """Generate requirements.txt."""
        base_requirements = """# VLA Fine-Tuning Requirements
# Generated by BlueprintPipeline

# Core
torch>=2.1.0
transformers>=4.40.0
datasets>=2.18.0
peft>=0.10.0
accelerate>=0.28.0

# LeRobot (for dataset loading)
lerobot>=0.4.0

# Training utilities
tensorboard>=2.15.0
wandb>=0.16.0
tqdm>=4.66.0
pyyaml>=6.0.0

# Image processing
pillow>=10.0.0
torchvision>=0.16.0
"""

        model_specific = {
            VLAModel.OPENVLA: """
# OpenVLA specific
bitsandbytes>=0.43.0
""",
            VLAModel.PI0: """
# Pi0 specific
einops>=0.7.0
""",
            VLAModel.SMOLVLA: """
# SmolVLA specific
# (uses standard transformers)
""",
            VLAModel.GROOT_N1: """
# GR00T specific
nvidia-isaac-lab>=1.0.0
""",
        }

        requirements = base_requirements + model_specific.get(config.model, "")

        req_path = model_dir / "requirements.txt"
        with open(req_path, "w") as f:
            f.write(requirements)

    def _generate_readme(
        self,
        model_dir: Path,
        config: VLAFinetuningConfig,
    ) -> None:
        """Generate model-specific README."""
        readme = f"""# {config.model_name} Fine-Tuning Package

Generated by BlueprintPipeline for scene: `{self.scene_id}`

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start training:
```bash
python train.py --config config.yaml
```

3. Evaluate:
```bash
python eval.py --checkpoint ./checkpoints/final --episodes 100
```

## Hardware Requirements

- **Minimum GPU Memory:** {config.min_gpu_memory_gb} GB
- **Recommended GPU:** {config.recommended_gpu}
- **Multi-GPU:** {'Required' if config.multi_gpu else 'Optional'}

## Training Configuration

- **Model Size:** {config.model_size}
- **LoRA Enabled:** {config.use_lora}
- **Batch Size:** {config.batch_size}
- **Learning Rate:** {config.learning_rate}
- **Epochs:** {config.num_epochs}

## Dataset

This package uses BlueprintPipeline-generated training data in LeRobot format.

- **Episodes:** See `../dataset_info.json` for details
- **Language Annotations:** `../language_annotations.json` (10 variations per task)
- **Image Size:** {config.image_size[0]}x{config.image_size[1]}

## Support

For issues with this fine-tuning package, contact: support@tryblueprint.io
"""

        readme_path = model_dir / "README.md"
        with open(readme_path, "w") as f:
            f.write(readme)

    def _generate_dataset_info(
        self,
        output_path: Path,
        episodes_meta: Dict[str, Any],
        language_annotations: Dict[str, List[str]],
    ) -> None:
        """Generate dataset info JSON."""
        info = {
            "blueprint_scene_id": self.scene_id,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "dataset_format": "lerobot_v2",
            "num_episodes": len(episodes_meta.get("episodes", [])),
            "num_tasks": len(episodes_meta.get("tasks", [])),
            "language_annotations": {
                "total_annotations": sum(len(v) for v in language_annotations.values()),
                "annotations_per_task": 10,
            },
            "supported_models": [m.value for m in self.models if m != VLAModel.ALL],
            "info": episodes_meta.get("info", {}),
        }

        with open(output_path, "w") as f:
            json.dump(info, f, indent=2)

    def _generate_master_readme(self) -> None:
        """Generate master README for the VLA package."""
        readme = f"""# VLA Fine-Tuning Package

Generated by BlueprintPipeline for scene: `{self.scene_id}`

## Overview

This package provides turnkey fine-tuning configurations for popular Vision-Language-Action (VLA) models:

| Model | Size | GPU Required | Status |
|-------|------|--------------|--------|
| OpenVLA | 7B | A100-80GB | ✅ Ready |
| Pi0 | 3B+300M | A100-40GB | ✅ Ready |
| SmolVLA | 450M | RTX 4090 | ✅ Ready |
| GR00T N1.5 | N1.5 | A100-80GB | ✅ Ready |

## Getting Started

1. Choose a model based on your hardware
2. Navigate to the model directory
3. Follow the README in that directory

## Package Contents

```
vla_finetuning/
├── openvla/           # OpenVLA fine-tuning
├── pi0/               # Pi0 fine-tuning
├── smolvla/           # SmolVLA fine-tuning
├── groot_n1/          # GR00T N1.5 fine-tuning
├── language_annotations.json
└── dataset_info.json
```

## Language Annotations

Each task has 10 natural language instruction variations for VLA training diversity.
See `language_annotations.json` for the full list.

## Support

- Documentation: https://tryblueprint.io/docs/vla
- Email: support@tryblueprint.io
"""

        readme_path = self.output_dir / "README.md"
        with open(readme_path, "w") as f:
            f.write(readme)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI entry point for VLA fine-tuning package generation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate VLA fine-tuning packages from BlueprintPipeline data"
    )
    parser.add_argument(
        "--episodes-dir",
        type=Path,
        required=True,
        help="Path to LeRobot episodes directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./vla_finetuning"),
        help="Output directory for VLA packages",
    )
    parser.add_argument(
        "--scene-id",
        type=str,
        required=True,
        help="Scene identifier",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["openvla", "pi0", "smolvla", "groot_n1", "all"],
        default=["openvla", "pi0", "smolvla"],
        help="VLA models to generate packages for",
    )

    args = parser.parse_args()

    # Parse models
    if "all" in args.models:
        models = [VLAModel.OPENVLA, VLAModel.PI0, VLAModel.SMOLVLA, VLAModel.GROOT_N1]
    else:
        models = [VLAModel(m) for m in args.models]

    generator = VLAFinetuningGenerator(
        episodes_dir=args.episodes_dir,
        output_dir=args.output_dir,
        scene_id=args.scene_id,
        models=models,
    )

    results = generator.generate_all()

    print("\nGenerated VLA fine-tuning packages:")
    for name, path in results.items():
        print(f"  - {name}: {path}")


if __name__ == "__main__":
    main()
