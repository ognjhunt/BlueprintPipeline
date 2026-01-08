# Blueprint RL Training Service - Architecture Specification

> **Purpose**: This document provides a complete specification for implementing a new `blueprint-rl-training` service that consumes BlueprintPipeline outputs and delivers trained RL policies. Another AI agent should be able to implement this service fully from this specification.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [What BlueprintPipeline Currently Provides](#2-what-blueprintpipeline-currently-provides)
3. [What the RL Training Service Must Do](#3-what-the-rl-training-service-must-do)
4. [System Architecture](#4-system-architecture)
5. [Input Schemas (From BlueprintPipeline)](#5-input-schemas-from-blueprintpipeline)
6. [Output Schemas (What We Deliver)](#6-output-schemas-what-we-deliver)
7. [Core Components](#7-core-components)
8. [Infrastructure Specification](#8-infrastructure-specification)
9. [API Specification](#9-api-specification)
10. [Training Pipeline](#10-training-pipeline)
11. [Evaluation & Benchmarking](#11-evaluation--benchmarking)
12. [Implementation Guide](#12-implementation-guide)
13. [Repository Structure](#13-repository-structure)
14. [Configuration Files](#14-configuration-files)
15. [Deployment](#15-deployment)

---

## 1. Executive Summary

### 1.1 The Gap

BlueprintPipeline generates **RL-ready Isaac Lab task packages** but does NOT:
- Actually run RL training at scale (1000s of parallel environments)
- Orchestrate distributed multi-GPU training
- Deliver trained policy checkpoints
- Provide benchmark evaluation reports

### 1.2 What We're Building

A new service `blueprint-rl-training` that:

| Capability | Description |
|------------|-------------|
| **Consumes** | Isaac Lab task packages from BlueprintPipeline (`scenes/{id}/isaac_lab/`) |
| **Runs** | Distributed PPO training via Isaac Lab + Ray |
| **Scales** | 1024-4096+ parallel environments across multiple GPUs |
| **Delivers** | Policy checkpoints, learning curves, evaluation reports |

### 1.3 Service Comparison

| Factor | BlueprintPipeline | RL Training Service |
|--------|-------------------|---------------------|
| **Runtime** | Minutes (scene generation) | Hours/Days (training) |
| **Compute** | CPU + light GPU (rendering) | Heavy GPU (training) |
| **Scaling** | Job-based (Cloud Run) | Distributed (multi-GPU, Ray) |
| **Lifecycle** | One-shot per scene | Long-running, resumable |
| **Delivery** | USD scenes + configs | Trained models + reports |

---

## 2. What BlueprintPipeline Currently Provides

### 2.1 Isaac Lab Output Directory Structure

BlueprintPipeline generates these files in `gs://{BUCKET}/scenes/{scene_id}/isaac_lab/`:

```
scenes/{scene_id}/isaac_lab/
├── env_cfg.py                    # ManagerBasedEnvCfg configuration
├── task_{policy_id}.py           # Task implementation (obs, actions, rewards)
├── train_cfg.yaml                # PPO training hyperparameters
├── reward_functions.py           # Reward component implementations
├── randomizations.py             # Domain randomization hooks (EventManager)
├── __init__.py                   # Package initialization
├── generation_metadata.json      # Generation info and validation status
└── .isaac_lab_complete           # Completion marker
```

### 2.2 Generated `env_cfg.py` Structure

The generated environment config follows Isaac Lab's `ManagerBasedEnvCfg` architecture:

```python
# Key imports
from omni.isaac.lab.envs import ManagerBasedEnvCfg
from omni.isaac.lab.managers import EventTermCfg, ObservationGroupCfg, RewardTermCfg
from omni.isaac.lab.scene import InteractiveSceneCfg

# Scene entity mapping (object IDs → USD prim paths)
SCENE_ENTITY_MAP = {
    "robot": "/World/Robot",
    "mug_001": "/World/Kitchen/Objects/mug_001",
    "plate_002": "/World/Kitchen/Objects/plate_002",
    # ... more objects from scene manifest
}

@configclass
class SceneCfg(InteractiveSceneCfg):
    num_envs: int = 1024          # Parallel environments
    env_spacing: float = 2.0       # Meters between envs
    # Robot, scene USD, lighting configs...

@configclass
class {TaskName}EnvCfg(ManagerBasedEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=1024, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    episode_length_s: float = 8.33  # ~500 steps at 60Hz
```

### 2.3 Generated `train_cfg.yaml` Structure

```yaml
# Task settings
task_name: "kitchen_dish_loading"
experiment_name: "kitchen_dish_loading_training"

# Environment settings
env:
  num_envs: 1024
  episode_length: 500

# PPO Algorithm settings
algo:
  name: "PPO"
  policy:
    class_name: "ActorCritic"
    init_noise_std: 1.0
    actor_hidden_dims: [256, 256, 128]
    critic_hidden_dims: [256, 256, 128]
    activation: "elu"

  # PPO specific
  clip_param: 0.2
  entropy_coef: 0.01
  value_loss_coef: 1.0
  max_grad_norm: 1.0

  # Learning rate
  learning_rate: 3.0e-4
  lr_schedule: "adaptive"

  # Batch settings
  num_learning_epochs: 5
  num_mini_batches: 4

  # Discount
  gamma: 0.99
  lam: 0.95

# Training settings
runner:
  max_iterations: 1500
  save_interval: 100
  log_interval: 10
  checkpoint_path: null
  resume: false

# Logging
logging:
  wandb:
    enabled: false
    project: "blueprint_recipe"
  tensorboard:
    enabled: true

# Reward weights
rewards:
  grasp_success: 10.0
  placement_accuracy: 5.0
  collision_penalty: -1.0
  action_jerk_penalty: 0.01
  action_magnitude_penalty: 0.001

# Domain randomization
randomization:
  enabled: true
  on_reset:
    - name: object_poses
      params:
        position_range: [-0.2, 0.2, -0.2, 0.2, 0.0, 0.1]
    - name: action_delay
      params:
        delay_range: [0, 3]
  on_step: []
```

### 2.4 Policy Configurations Available

BlueprintPipeline supports 14 policy types (from `policy_configs/environment_policies.json`):

| Policy ID | Description | Default Environments |
|-----------|-------------|---------------------|
| `dexterous_pick_place` | General pick & place | All |
| `articulated_access` | Door/drawer opening | Kitchen, Office |
| `dish_loading` | Dishwasher loading | Kitchen |
| `table_clearing` | Table clearing | Kitchen, Dining |
| `drawer_manipulation` | Drawer operations | Kitchen, Office, Bedroom |
| `door_manipulation` | Door operations | All |
| `laundry_sorting` | Cloth handling | Laundry |
| `mixed_sku_logistics` | Warehouse logistics | Warehouse |
| `grocery_stocking` | Shelf stocking | Grocery |
| `knob_manipulation` | Rotational control | Kitchen, Lab |
| `precision_insertion` | Fine manipulation | Lab, Industrial |
| `general_manipulation` | Generic manipulation | All |
| `panel_interaction` | Button/switch operation | Industrial |
| `pallet_handling` | Pallet manipulation | Warehouse |

### 2.5 Physics Profiles

BlueprintPipeline selects physics profiles based on task type (from `policy_configs/physics_profiles.json`):

| Profile | dt | Substeps | Solver Iters | Use Cases |
|---------|-----|----------|--------------|-----------|
| `manipulation_contact_rich` | 0.004 | 4 | 32 | Grasping, insertion, assembly |
| `manipulation_standard` | 0.008 | 2 | 16 | Pick-place, drawer, door (DEFAULT) |
| `navigation` | 0.016 | 1 | 8 | Mobile robots, locomotion |
| `vision_only` | 0.033 | 1 | 4 | Perception training |
| `deformable` | 0.002 | 8 | 48 | Cloth, cables, soft objects |

### 2.6 Robot Configurations

Three robots are supported:

**Franka Panda** (7 DOF + 2 gripper):
```python
{
    "num_dofs": 7,
    "gripper_dofs": 2,
    "ee_frame": "panda_hand",
    "default_joint_pos": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
    "joint_limits_lower": [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
    "joint_limits_upper": [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
}
```

**UR10** (6 DOF, no gripper by default):
```python
{
    "num_dofs": 6,
    "gripper_dofs": 0,
    "ee_frame": "tool0",
    "default_joint_pos": [0.0, -1.571, 1.571, -1.571, -1.571, 0.0]
}
```

**Fetch** (7 DOF + 2 gripper, mobile base):
```python
{
    "num_dofs": 7,
    "gripper_dofs": 2,
    "ee_frame": "gripper_link",
    "default_joint_pos": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}
```

### 2.7 Reward Functions

BlueprintPipeline generates these reward components (from `tools/isaac_lab_tasks/reward_functions.py`):

**Task Rewards:**
- `grasp_success` - Binary grasp detection
- `placement_accuracy` - Distance-based placement reward
- `joint_progress` - Articulation progress
- `handle_grasp` - Handle grasping for doors/drawers
- `task_completion` - Large bonus for success
- `dish_placed` - Dishes in dishwasher rack
- `sorting_accuracy` - Correct bin placement
- `rotation_accuracy` - Knob/dial rotation

**Sim2Real Transfer Rewards (CRITICAL):**
- `action_jerk_penalty` - Penalizes high-frequency action changes
- `action_rate_penalty` - Penalizes exceeding rate limits
- `action_magnitude_penalty` - Penalizes large actions
- `joint_acceleration_penalty` - Penalizes jerky joint motion
- `smooth_ee_velocity` - Encourages smooth EE motion

**Safety Rewards:**
- `collision_penalty` - Contact force penalty
- `smooth_motion` - Joint acceleration penalty
- `efficiency_bonus` - Time penalty

### 2.8 Scene Manifest Schema

The scene manifest (`scenes/{id}/assets/scene_manifest.json`) provides object information:

```json
{
  "version": "1.0.0",
  "scene_id": "kitchen_001",
  "scene": {
    "coordinate_frame": "y_up",
    "meters_per_unit": 1.0,
    "environment_type": "kitchen",
    "room": {
      "bounds": {"width": 4.0, "depth": 5.0, "height": 2.7},
      "origin": [0.0, 0.0, 0.0]
    }
  },
  "objects": [
    {
      "id": "mug_001",
      "name": "Coffee Mug",
      "category": "kitchenware",
      "sim_role": "manipulable_object",
      "transform": {
        "position": {"x": 0.5, "y": 0.0, "z": 0.9},
        "rotation_quaternion": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
        "scale": {"x": 1.0, "y": 1.0, "z": 1.0}
      },
      "dimensions_est": {"width": 0.08, "depth": 0.08, "height": 0.1},
      "semantics": {
        "class": "mug",
        "affordances": ["Graspable", "Fillable"],
        "affordance_params": {
          "confidence": 0.95,
          "source": "detected"
        }
      },
      "physics": {
        "mass": 0.3,
        "friction": 0.6,
        "restitution": 0.1
      }
    }
  ]
}
```

### 2.9 GCS Storage Layout

```
gs://{BUCKET}/
└── scenes/
    └── {scene_id}/
        ├── input/
        │   └── room.jpg
        ├── assets/
        │   ├── scene_manifest.json      # Scene metadata
        │   └── {object_id}/             # Per-object assets
        ├── usd/
        │   ├── scene.usda               # Assembled USD scene
        │   └── .usd_complete            # Completion marker
        ├── replicator/
        │   ├── placement_regions.usda
        │   └── bundle_metadata.json
        ├── isaac_lab/                   # <-- RL Training Service reads from here
        │   ├── env_cfg.py
        │   ├── task_{policy}.py
        │   ├── train_cfg.yaml
        │   ├── reward_functions.py
        │   ├── randomizations.py
        │   ├── __init__.py
        │   └── .isaac_lab_complete      # Trigger for RL training
        ├── episodes/                    # Episode data (imitation learning)
        │   └── ...
        └── rl_training/                 # <-- RL Training Service writes here (NEW)
            └── ...
```

---

## 3. What the RL Training Service Must Do

### 3.1 Core Functionality

1. **Trigger** on `.isaac_lab_complete` marker in GCS
2. **Download** Isaac Lab task package from `scenes/{id}/isaac_lab/`
3. **Validate** the task package (syntax, imports, sanity rollout)
4. **Train** using distributed PPO with Ray + Isaac Lab
5. **Evaluate** trained policy on fixed benchmark seeds
6. **Upload** results to `scenes/{id}/rl_training/`

### 3.2 Training Capabilities

| Capability | Specification |
|------------|---------------|
| **Algorithm** | PPO (Proximal Policy Optimization) via RL-Games or RSL-RL |
| **Parallelism** | 1024-4096 environments per GPU |
| **Distribution** | Multi-GPU via PyTorch DDP, multi-node via Ray |
| **Checkpointing** | Every N iterations, resumable |
| **Logging** | TensorBoard, Weights & Biases (optional) |

### 3.3 Deliverables

For each training run, deliver:

1. **Policy Checkpoint** (`policy.pt`) - Trained PyTorch model
2. **Training Curves** - Loss, reward, success rate over time
3. **Evaluation Report** - Performance on fixed benchmark seeds
4. **Config Archive** - All configs used for reproducibility

---

## 4. System Architecture

### 4.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BlueprintPipeline                                  │
│  (Existing - generates Isaac Lab task packages)                             │
│                                                                              │
│  Image → 3D-RE-GEN → SimReady → USD → Replicator → Isaac Lab Task          │
│                                                                              │
│  Output: gs://{BUCKET}/scenes/{id}/isaac_lab/                               │
│          - env_cfg.py, train_cfg.yaml, reward_functions.py, etc.            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ Triggers on .isaac_lab_complete
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RL Training Service (NEW)                             │
│                                                                              │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                   │
│  │  Trigger    │     │  Training   │     │  Evaluation │                   │
│  │  Service    │────▶│  Orchestrator│────▶│  Service    │                   │
│  │  (EventArc) │     │  (Ray)      │     │             │                   │
│  └─────────────┘     └─────────────┘     └─────────────┘                   │
│         │                   │                   │                            │
│         │                   ▼                   │                            │
│         │           ┌─────────────┐             │                            │
│         │           │  Training   │             │                            │
│         │           │  Workers    │             │                            │
│         │           │  (GPU Pods) │             │                            │
│         │           └─────────────┘             │                            │
│         │                   │                   │                            │
│         └───────────────────┴───────────────────┘                            │
│                             │                                                │
│                             ▼                                                │
│  Output: gs://{BUCKET}/scenes/{id}/rl_training/                             │
│          - policy.pt, training_curves.json, eval_report.json                │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Component Breakdown

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RL Training Service                                  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        API Layer (FastAPI)                            │   │
│  │  POST /training/start     - Start training job                        │   │
│  │  GET  /training/{id}      - Get training status                       │   │
│  │  POST /training/{id}/stop - Stop training job                         │   │
│  │  GET  /training/{id}/metrics - Get live metrics                       │   │
│  │  POST /evaluation/run     - Run evaluation on trained policy          │   │
│  │  GET  /evaluation/{id}    - Get evaluation results                    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                      │                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     Training Orchestrator (Ray)                       │   │
│  │  - Job scheduling and lifecycle management                            │   │
│  │  - Multi-GPU coordination                                             │   │
│  │  - Checkpointing and resumption                                       │   │
│  │  - Hyperparameter tuning (Ray Tune)                                   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                      │                                       │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌────────────────────┐   │
│  │   Task Loader       │  │   Training Engine   │  │  Evaluation Engine │   │
│  │   - Download from   │  │   - Isaac Lab env   │  │  - Fixed seeds     │   │
│  │     GCS             │  │   - PPO algorithm   │  │  - Success rate    │   │
│  │   - Validate task   │  │   - Distributed     │  │  - Reward metrics  │   │
│  │   - Setup env       │  │     training        │  │  - Benchmark       │   │
│  └─────────────────────┘  └─────────────────────┘  └────────────────────┘   │
│                                      │                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      Storage Layer (GCS)                              │   │
│  │  Read:  scenes/{id}/isaac_lab/*                                      │   │
│  │  Write: scenes/{id}/rl_training/*                                    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Input Schemas (From BlueprintPipeline)

### 5.1 Training Trigger Event

When BlueprintPipeline completes Isaac Lab task generation, it writes a marker file that triggers training:

**Trigger File**: `gs://{BUCKET}/scenes/{scene_id}/isaac_lab/.isaac_lab_complete`

**Content**:
```json
{
  "scene_id": "kitchen_001",
  "timestamp": "2026-01-08T12:00:00Z",
  "policy_id": "dish_loading",
  "robot_type": "franka",
  "environment_type": "kitchen",
  "validation_status": "passed",
  "syntax_valid": true,
  "runtime_validated": true,
  "sanity_rollout_passed": true
}
```

### 5.2 Task Package Files

The RL Training Service must be able to parse and use these files:

#### 5.2.1 `env_cfg.py` - Environment Configuration

```python
# Import structure expected
from omni.isaac.lab.envs import ManagerBasedEnvCfg

# Key class to instantiate
@configclass
class KitchenDishLoadingEnvCfg(ManagerBasedEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=1024, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
```

#### 5.2.2 `train_cfg.yaml` - Training Hyperparameters

See Section 2.3 for full schema.

#### 5.2.3 `generation_metadata.json` - Validation Info

```json
{
  "generated_at": "2026-01-08T12:00:00Z",
  "generator_version": "2.0.0",
  "policy_id": "dish_loading",
  "robot_type": "franka",
  "physics_profile": "manipulation_standard",
  "validation": {
    "syntax_valid": true,
    "imports_valid": true,
    "runtime_validated": true,
    "sanity_rollout": {
      "num_envs": 4,
      "num_steps": 10,
      "passed": true
    }
  },
  "scene_entities": {
    "mug_001": "/World/Kitchen/Objects/mug_001",
    "plate_002": "/World/Kitchen/Objects/plate_002"
  }
}
```

---

## 6. Output Schemas (What We Deliver)

### 6.1 Output Directory Structure

```
gs://{BUCKET}/scenes/{scene_id}/rl_training/
├── jobs/
│   └── {job_id}/
│       ├── config/
│       │   ├── train_cfg.yaml          # Training config used
│       │   ├── env_cfg.py              # Env config used
│       │   └── job_config.json         # Job-specific overrides
│       ├── checkpoints/
│       │   ├── checkpoint_000100.pt    # Intermediate checkpoints
│       │   ├── checkpoint_000500.pt
│       │   └── checkpoint_final.pt     # Final checkpoint
│       ├── logs/
│       │   ├── tensorboard/            # TensorBoard logs
│       │   └── training.log            # Text logs
│       ├── metrics/
│       │   ├── training_curves.json    # Training metrics over time
│       │   └── final_metrics.json      # Final training metrics
│       └── job_status.json             # Job status and metadata
├── policies/
│   └── {policy_name}/
│       ├── policy.pt                   # Best trained policy
│       ├── policy_config.json          # Policy architecture info
│       └── training_summary.json       # Training summary
├── evaluations/
│   └── {eval_id}/
│       ├── eval_config.json            # Evaluation configuration
│       ├── eval_results.json           # Detailed results
│       └── benchmark_report.json       # Standardized benchmark report
└── .rl_training_complete               # Completion marker
```

### 6.2 Training Curves Schema (`training_curves.json`)

```json
{
  "job_id": "job_abc123",
  "scene_id": "kitchen_001",
  "policy_id": "dish_loading",
  "iterations": [
    {
      "iteration": 100,
      "timestamp": "2026-01-08T12:30:00Z",
      "metrics": {
        "mean_reward": 15.2,
        "mean_episode_length": 245,
        "success_rate": 0.45,
        "policy_loss": 0.023,
        "value_loss": 0.15,
        "entropy": 1.2,
        "learning_rate": 0.0003,
        "fps": 45000
      }
    }
  ],
  "reward_components": {
    "grasp_success": [0.0, 2.1, 5.3, ...],
    "placement_accuracy": [0.0, 1.2, 3.4, ...],
    "collision_penalty": [-0.5, -0.3, -0.1, ...]
  }
}
```

### 6.3 Evaluation Report Schema (`benchmark_report.json`)

```json
{
  "eval_id": "eval_xyz789",
  "job_id": "job_abc123",
  "scene_id": "kitchen_001",
  "policy_id": "dish_loading",
  "policy_checkpoint": "checkpoint_final.pt",
  "evaluation_config": {
    "num_episodes": 100,
    "num_envs": 64,
    "fixed_seeds": [42, 123, 456, 789, 1024],
    "deterministic": true
  },
  "results": {
    "success_rate": 0.87,
    "success_rate_std": 0.05,
    "mean_reward": 85.3,
    "reward_std": 12.1,
    "mean_episode_length": 312,
    "episode_length_std": 45,
    "completion_time_mean": 5.2,
    "completion_time_std": 0.8
  },
  "per_seed_results": [
    {
      "seed": 42,
      "success_rate": 0.88,
      "mean_reward": 86.1,
      "episodes": 20
    }
  ],
  "task_specific_metrics": {
    "dishes_placed": 4.2,
    "grasp_attempts": 5.1,
    "collision_count": 0.3
  },
  "sim2real_indicators": {
    "action_smoothness": 0.92,
    "mean_action_jerk": 0.015,
    "max_joint_acceleration": 12.3
  },
  "benchmark_version": "1.0.0",
  "timestamp": "2026-01-08T14:00:00Z"
}
```

### 6.4 Policy Package Schema (`policy_config.json`)

```json
{
  "policy_name": "kitchen_dish_loading_v1",
  "scene_id": "kitchen_001",
  "training_job_id": "job_abc123",
  "architecture": {
    "type": "ActorCritic",
    "actor_hidden_dims": [256, 256, 128],
    "critic_hidden_dims": [256, 256, 128],
    "activation": "elu",
    "init_noise_std": 1.0
  },
  "observation_space": {
    "robot_joint_pos": 7,
    "robot_joint_vel": 7,
    "robot_ee_pos": 3,
    "robot_gripper_pos": 2,
    "target_rel_pos": 3,
    "total_dim": 22
  },
  "action_space": {
    "type": "continuous",
    "dim": 9,
    "control_mode": "joint_velocity"
  },
  "training_info": {
    "algorithm": "PPO",
    "total_iterations": 1500,
    "total_timesteps": 768000000,
    "final_success_rate": 0.87,
    "training_time_hours": 4.5
  },
  "checkpoint_path": "policies/kitchen_dish_loading_v1/policy.pt",
  "compatible_robots": ["franka"],
  "compatible_isaac_lab_version": "1.0.0"
}
```

### 6.5 Completion Marker (`.rl_training_complete`)

```json
{
  "scene_id": "kitchen_001",
  "job_id": "job_abc123",
  "status": "completed",
  "timestamp": "2026-01-08T16:30:00Z",
  "training_iterations": 1500,
  "final_success_rate": 0.87,
  "policy_path": "policies/kitchen_dish_loading_v1/policy.pt",
  "eval_path": "evaluations/eval_xyz789/benchmark_report.json"
}
```

---

## 7. Core Components

### 7.1 Task Loader

**Purpose**: Download and validate Isaac Lab task packages from GCS.

```python
# src/task_loader/loader.py

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import importlib.util
import yaml

@dataclass
class TaskPackage:
    """Loaded Isaac Lab task package."""
    scene_id: str
    policy_id: str
    robot_type: str
    env_cfg_path: Path
    train_cfg_path: Path
    reward_functions_path: Path
    randomizations_path: Path
    task_file_path: Path
    env_cfg_class: type  # Loaded class
    train_cfg: dict      # Parsed YAML
    metadata: dict       # generation_metadata.json

class TaskLoader:
    """Load and validate Isaac Lab task packages from GCS."""

    def __init__(self, gcs_client, local_cache_dir: Path):
        self.gcs_client = gcs_client
        self.cache_dir = local_cache_dir

    def load(self, bucket: str, scene_id: str) -> TaskPackage:
        """
        Download task package from GCS and load into memory.

        Args:
            bucket: GCS bucket name
            scene_id: Scene identifier

        Returns:
            TaskPackage with loaded configs and classes
        """
        # 1. Download files from GCS
        isaac_lab_prefix = f"scenes/{scene_id}/isaac_lab"
        local_dir = self.cache_dir / scene_id / "isaac_lab"
        self._download_directory(bucket, isaac_lab_prefix, local_dir)

        # 2. Load metadata
        metadata = self._load_json(local_dir / "generation_metadata.json")

        # 3. Load train config
        train_cfg = self._load_yaml(local_dir / "train_cfg.yaml")

        # 4. Dynamically import env_cfg module
        env_cfg_class = self._load_env_cfg_class(local_dir / "env_cfg.py")

        # 5. Validate
        self._validate(env_cfg_class, train_cfg, metadata)

        return TaskPackage(
            scene_id=scene_id,
            policy_id=metadata.get("policy_id"),
            robot_type=metadata.get("robot_type"),
            env_cfg_path=local_dir / "env_cfg.py",
            train_cfg_path=local_dir / "train_cfg.yaml",
            reward_functions_path=local_dir / "reward_functions.py",
            randomizations_path=local_dir / "randomizations.py",
            task_file_path=local_dir / f"task_{metadata['policy_id']}.py",
            env_cfg_class=env_cfg_class,
            train_cfg=train_cfg,
            metadata=metadata
        )

    def _load_env_cfg_class(self, env_cfg_path: Path) -> type:
        """Dynamically load the environment config class."""
        spec = importlib.util.spec_from_file_location("env_cfg", env_cfg_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find the *EnvCfg class
        for name in dir(module):
            if name.endswith("EnvCfg") and name != "ManagerBasedEnvCfg":
                return getattr(module, name)
        raise ValueError(f"No *EnvCfg class found in {env_cfg_path}")

    def _validate(self, env_cfg_class, train_cfg, metadata):
        """Validate the task package is usable."""
        # Check metadata validation passed
        if not metadata.get("validation", {}).get("syntax_valid"):
            raise ValueError("Task package failed syntax validation")

        # Check train_cfg has required fields
        required_fields = ["env", "algo", "runner"]
        for field in required_fields:
            if field not in train_cfg:
                raise ValueError(f"train_cfg missing required field: {field}")
```

### 7.2 Training Engine

**Purpose**: Run distributed PPO training using Isaac Lab.

```python
# src/training/engine.py

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable
import torch
import torch.distributed as dist

@dataclass
class TrainingConfig:
    """Training configuration."""
    num_envs: int = 1024
    max_iterations: int = 1500
    save_interval: int = 100
    log_interval: int = 10

    # PPO hyperparameters
    learning_rate: float = 3e-4
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    value_loss_coef: float = 1.0
    gamma: float = 0.99
    lam: float = 0.95
    num_learning_epochs: int = 5
    num_mini_batches: int = 4

    # Distributed training
    num_gpus: int = 1
    use_ray: bool = True

    # Checkpointing
    checkpoint_dir: Path = None
    resume_from: Optional[Path] = None

@dataclass
class TrainingResult:
    """Result of a training run."""
    job_id: str
    final_checkpoint: Path
    total_iterations: int
    final_metrics: dict
    training_curves: list
    training_time_seconds: float

class TrainingEngine:
    """
    Distributed PPO training engine using Isaac Lab.

    Supports:
    - Single GPU training
    - Multi-GPU training via PyTorch DDP
    - Multi-node training via Ray
    """

    def __init__(
        self,
        task_package: "TaskPackage",
        config: TrainingConfig,
        callbacks: Optional[list[Callable]] = None
    ):
        self.task_package = task_package
        self.config = config
        self.callbacks = callbacks or []

        # Will be initialized in setup()
        self.env = None
        self.agent = None
        self.optimizer = None

    def setup(self):
        """Initialize environment, agent, and optimizer."""
        # Import Isaac Lab components
        from omni.isaac.lab.envs import ManagerBasedEnv

        # Create environment from loaded config
        env_cfg = self.task_package.env_cfg_class()
        env_cfg.scene.num_envs = self.config.num_envs

        self.env = ManagerBasedEnv(cfg=env_cfg)

        # Setup PPO agent
        self.agent = self._create_ppo_agent()

        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            self.agent.parameters(),
            lr=self.config.learning_rate
        )

        # Resume from checkpoint if specified
        if self.config.resume_from:
            self._load_checkpoint(self.config.resume_from)

    def train(self) -> TrainingResult:
        """
        Run training loop.

        Returns:
            TrainingResult with final checkpoint and metrics
        """
        import time

        start_time = time.time()
        training_curves = []

        for iteration in range(self.config.max_iterations):
            # Collect rollouts
            rollouts = self._collect_rollouts()

            # Update policy
            metrics = self._update_policy(rollouts)

            # Log metrics
            if iteration % self.config.log_interval == 0:
                metrics["iteration"] = iteration
                training_curves.append(metrics)
                self._log_metrics(metrics)

            # Save checkpoint
            if iteration % self.config.save_interval == 0:
                self._save_checkpoint(iteration)

            # Run callbacks
            for callback in self.callbacks:
                callback(iteration, metrics)

        # Save final checkpoint
        final_checkpoint = self._save_checkpoint("final")

        training_time = time.time() - start_time

        return TrainingResult(
            job_id=self.job_id,
            final_checkpoint=final_checkpoint,
            total_iterations=self.config.max_iterations,
            final_metrics=training_curves[-1] if training_curves else {},
            training_curves=training_curves,
            training_time_seconds=training_time
        )

    def _create_ppo_agent(self):
        """Create PPO agent with ActorCritic network."""
        # Use RL-Games or RSL-RL implementation
        # This is a simplified version
        from .networks import ActorCritic

        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]

        return ActorCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dims=self.task_package.train_cfg["algo"]["policy"]["actor_hidden_dims"],
            activation=self.task_package.train_cfg["algo"]["policy"]["activation"]
        )

    def _collect_rollouts(self):
        """Collect rollout data from environments."""
        # Implementation depends on RL library (RL-Games, RSL-RL, etc.)
        pass

    def _update_policy(self, rollouts) -> dict:
        """Update policy using PPO."""
        # Implementation depends on RL library
        pass

    def _save_checkpoint(self, iteration) -> Path:
        """Save training checkpoint."""
        checkpoint_path = self.config.checkpoint_dir / f"checkpoint_{iteration}.pt"
        torch.save({
            "iteration": iteration,
            "agent_state_dict": self.agent.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.__dict__
        }, checkpoint_path)
        return checkpoint_path

    def _load_checkpoint(self, checkpoint_path: Path):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        self.agent.load_state_dict(checkpoint["agent_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
```

### 7.3 Evaluation Engine

**Purpose**: Evaluate trained policies on fixed benchmark seeds.

```python
# src/evaluation/engine.py

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import torch
import numpy as np

@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    num_episodes: int = 100
    num_envs: int = 64
    fixed_seeds: list[int] = None  # Default: [42, 123, 456, 789, 1024]
    deterministic: bool = True
    record_videos: bool = False

    def __post_init__(self):
        if self.fixed_seeds is None:
            self.fixed_seeds = [42, 123, 456, 789, 1024]

@dataclass
class EvaluationResult:
    """Evaluation results."""
    eval_id: str
    success_rate: float
    success_rate_std: float
    mean_reward: float
    reward_std: float
    mean_episode_length: float
    episode_length_std: float
    per_seed_results: list[dict]
    task_specific_metrics: dict
    sim2real_indicators: dict

class EvaluationEngine:
    """
    Evaluate trained policies using fixed benchmark seeds.

    Provides standardized evaluation metrics for comparing
    policies across different training runs and scenes.
    """

    def __init__(
        self,
        task_package: "TaskPackage",
        policy_path: Path,
        config: EvaluationConfig
    ):
        self.task_package = task_package
        self.policy_path = policy_path
        self.config = config

    def evaluate(self) -> EvaluationResult:
        """
        Run evaluation across all fixed seeds.

        Returns:
            EvaluationResult with comprehensive metrics
        """
        all_results = []

        for seed in self.config.fixed_seeds:
            seed_results = self._evaluate_seed(seed)
            all_results.append(seed_results)

        # Aggregate results
        return self._aggregate_results(all_results)

    def _evaluate_seed(self, seed: int) -> dict:
        """Evaluate policy with a specific seed."""
        # Set seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create environment
        env_cfg = self.task_package.env_cfg_class()
        env_cfg.scene.num_envs = self.config.num_envs
        env = self._create_env(env_cfg)

        # Load policy
        policy = self._load_policy()

        # Run episodes
        episode_rewards = []
        episode_lengths = []
        episode_successes = []
        action_jerks = []

        episodes_per_seed = self.config.num_episodes // len(self.config.fixed_seeds)

        for _ in range(episodes_per_seed):
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            prev_action = None
            episode_jerk = []

            while not done:
                # Get action from policy
                with torch.no_grad():
                    if self.config.deterministic:
                        action = policy.act_deterministic(obs)
                    else:
                        action = policy.act(obs)

                # Track action jerk for sim2real metrics
                if prev_action is not None:
                    jerk = torch.norm(action - prev_action).item()
                    episode_jerk.append(jerk)
                prev_action = action.clone()

                # Step environment
                obs, reward, done, info = env.step(action)
                episode_reward += reward.mean().item()
                episode_length += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_successes.append(info.get("success", False))
            action_jerks.extend(episode_jerk)

        return {
            "seed": seed,
            "success_rate": np.mean(episode_successes),
            "mean_reward": np.mean(episode_rewards),
            "reward_std": np.std(episode_rewards),
            "mean_episode_length": np.mean(episode_lengths),
            "mean_action_jerk": np.mean(action_jerks) if action_jerks else 0.0,
            "episodes": episodes_per_seed
        }

    def _aggregate_results(self, seed_results: list[dict]) -> EvaluationResult:
        """Aggregate results across all seeds."""
        success_rates = [r["success_rate"] for r in seed_results]
        rewards = [r["mean_reward"] for r in seed_results]
        lengths = [r["mean_episode_length"] for r in seed_results]
        jerks = [r["mean_action_jerk"] for r in seed_results]

        return EvaluationResult(
            eval_id=self._generate_eval_id(),
            success_rate=np.mean(success_rates),
            success_rate_std=np.std(success_rates),
            mean_reward=np.mean(rewards),
            reward_std=np.std(rewards),
            mean_episode_length=np.mean(lengths),
            episode_length_std=np.std(lengths),
            per_seed_results=seed_results,
            task_specific_metrics=self._compute_task_metrics(),
            sim2real_indicators={
                "action_smoothness": 1.0 - min(1.0, np.mean(jerks) * 10),
                "mean_action_jerk": np.mean(jerks)
            }
        )
```

### 7.4 Ray Training Orchestrator

**Purpose**: Manage distributed training jobs using Ray.

```python
# src/orchestrator/ray_orchestrator.py

import ray
from ray import tune
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import uuid

@dataclass
class JobConfig:
    """Training job configuration."""
    scene_id: str
    bucket: str
    num_gpus: int = 1
    num_workers: int = 1  # For multi-node
    priority: int = 0
    max_runtime_hours: float = 24.0

    # Hyperparameter overrides
    num_envs: Optional[int] = None
    max_iterations: Optional[int] = None
    learning_rate: Optional[float] = None

@dataclass
class JobStatus:
    """Training job status."""
    job_id: str
    scene_id: str
    status: str  # pending, running, completed, failed
    progress: float  # 0.0 - 1.0
    current_iteration: int
    metrics: Optional[dict]
    error: Optional[str]

class RayOrchestrator:
    """
    Orchestrate distributed training jobs using Ray.

    Features:
    - Multi-GPU training coordination
    - Job queueing and prioritization
    - Checkpointing and resumption
    - Hyperparameter tuning (Ray Tune)
    """

    def __init__(self, ray_address: Optional[str] = None):
        if not ray.is_initialized():
            ray.init(address=ray_address)

        self.active_jobs: dict[str, ray.ObjectRef] = {}

    def submit_job(self, config: JobConfig) -> str:
        """
        Submit a new training job.

        Args:
            config: Job configuration

        Returns:
            job_id: Unique identifier for tracking
        """
        job_id = f"rl_train_{config.scene_id}_{uuid.uuid4().hex[:8]}"

        # Create Ray training config
        scaling_config = ScalingConfig(
            num_workers=config.num_workers,
            use_gpu=True,
            resources_per_worker={"GPU": config.num_gpus}
        )

        run_config = RunConfig(
            name=job_id,
            storage_path=f"gs://{config.bucket}/scenes/{config.scene_id}/rl_training/jobs/{job_id}",
            checkpoint_config=ray.train.CheckpointConfig(
                num_to_keep=3,
                checkpoint_frequency=100
            )
        )

        # Create trainer
        trainer = TorchTrainer(
            train_loop_per_worker=self._train_loop,
            train_loop_config={
                "scene_id": config.scene_id,
                "bucket": config.bucket,
                "num_envs": config.num_envs,
                "max_iterations": config.max_iterations,
                "learning_rate": config.learning_rate
            },
            scaling_config=scaling_config,
            run_config=run_config
        )

        # Submit job
        result_ref = trainer.fit()
        self.active_jobs[job_id] = result_ref

        return job_id

    def get_status(self, job_id: str) -> JobStatus:
        """Get status of a training job."""
        if job_id not in self.active_jobs:
            return JobStatus(
                job_id=job_id,
                scene_id="unknown",
                status="not_found",
                progress=0.0,
                current_iteration=0,
                metrics=None,
                error="Job not found"
            )

        # Query Ray for job status
        # Implementation depends on Ray version
        pass

    def stop_job(self, job_id: str) -> bool:
        """Stop a running training job."""
        if job_id in self.active_jobs:
            ray.cancel(self.active_jobs[job_id])
            del self.active_jobs[job_id]
            return True
        return False

    def submit_hyperparameter_sweep(
        self,
        config: JobConfig,
        param_space: dict,
        num_samples: int = 10
    ) -> str:
        """
        Submit a hyperparameter tuning job using Ray Tune.

        Args:
            config: Base job configuration
            param_space: Parameter search space
            num_samples: Number of trials to run

        Returns:
            sweep_id: Unique identifier for the sweep
        """
        sweep_id = f"rl_sweep_{config.scene_id}_{uuid.uuid4().hex[:8]}"

        tuner = tune.Tuner(
            self._train_loop,
            param_space=param_space,
            tune_config=tune.TuneConfig(
                num_samples=num_samples,
                scheduler=tune.schedulers.ASHAScheduler(
                    metric="success_rate",
                    mode="max",
                    max_t=config.max_iterations or 1500,
                    grace_period=100
                )
            ),
            run_config=RunConfig(
                name=sweep_id,
                storage_path=f"gs://{config.bucket}/scenes/{config.scene_id}/rl_training/sweeps/{sweep_id}"
            )
        )

        results = tuner.fit()
        return sweep_id

    @staticmethod
    def _train_loop(config: dict):
        """Training loop executed by Ray workers."""
        import ray.train as train
        from ..task_loader import TaskLoader
        from ..training import TrainingEngine, TrainingConfig

        # Load task package
        loader = TaskLoader(gcs_client=None, local_cache_dir=Path("/tmp/tasks"))
        task_package = loader.load(config["bucket"], config["scene_id"])

        # Create training config
        train_config = TrainingConfig(
            num_envs=config.get("num_envs", 1024),
            max_iterations=config.get("max_iterations", 1500),
            learning_rate=config.get("learning_rate", 3e-4)
        )

        # Run training
        engine = TrainingEngine(task_package, train_config)
        engine.setup()

        # Training loop with Ray reporting
        for iteration in range(train_config.max_iterations):
            metrics = engine.train_step()

            # Report to Ray
            train.report(metrics)

            # Save checkpoint
            if iteration % 100 == 0:
                with train.checkpoint_dir() as checkpoint_dir:
                    engine.save_checkpoint(checkpoint_dir)
```

---

## 8. Infrastructure Specification

### 8.1 GKE Cluster Requirements

```yaml
# Cluster specification for RL training
cluster:
  name: blueprint-rl-cluster
  location: us-central1-a

  # Node pools
  node_pools:
    # CPU pool for orchestration
    - name: cpu-pool
      machine_type: n2-standard-8
      num_nodes: 2
      autoscaling:
        enabled: true
        min_nodes: 1
        max_nodes: 5

    # GPU pool for training
    - name: gpu-pool
      machine_type: n1-standard-16
      accelerator:
        type: nvidia-tesla-v100  # or a100-40gb for larger jobs
        count: 1
      num_nodes: 0  # Scale to zero when idle
      autoscaling:
        enabled: true
        min_nodes: 0
        max_nodes: 16
      taints:
        - key: nvidia.com/gpu
          value: present
          effect: NoSchedule

    # Multi-GPU pool for large-scale training
    - name: multi-gpu-pool
      machine_type: a2-highgpu-4g  # 4x A100-40GB
      num_nodes: 0
      autoscaling:
        enabled: true
        min_nodes: 0
        max_nodes: 4
      taints:
        - key: nvidia.com/gpu
          value: present
          effect: NoSchedule
```

### 8.2 Kubernetes Resources

#### 8.2.1 Namespace Setup

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: rl-training
  labels:
    app: blueprint-rl-training
---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: rl-training-quota
  namespace: rl-training
spec:
  hard:
    requests.nvidia.com/gpu: "32"
    limits.nvidia.com/gpu: "32"
    requests.memory: "512Gi"
    limits.memory: "1Ti"
```

#### 8.2.2 Training Job Template

```yaml
# k8s/training-job-template.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: rl-training-${JOB_ID}
  namespace: rl-training
  labels:
    app: rl-training
    scene-id: ${SCENE_ID}
    job-id: ${JOB_ID}
spec:
  backoffLimit: 2
  activeDeadlineSeconds: 86400  # 24 hours max
  ttlSecondsAfterFinished: 3600

  template:
    metadata:
      labels:
        app: rl-training
        job-id: ${JOB_ID}
      annotations:
        cluster-autoscaler.kubernetes.io/safe-to-evict: "false"
    spec:
      restartPolicy: OnFailure
      serviceAccountName: rl-training-sa

      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-tesla-v100

      tolerations:
        - key: nvidia.com/gpu
          operator: Equal
          value: present
          effect: NoSchedule

      containers:
        - name: trainer
          image: ${REGION}-docker.pkg.dev/${PROJECT_ID}/rl-training/trainer:latest

          command: ["python", "-m", "rl_training.cli", "train"]
          args:
            - "--scene-id=${SCENE_ID}"
            - "--bucket=${BUCKET}"
            - "--job-id=${JOB_ID}"
            - "--num-envs=${NUM_ENVS}"
            - "--max-iterations=${MAX_ITERATIONS}"

          env:
            - name: GOOGLE_APPLICATION_CREDENTIALS
              value: /secrets/gcs/key.json
            - name: CUDA_VISIBLE_DEVICES
              value: "0"
            - name: NVIDIA_VISIBLE_DEVICES
              value: "all"
            - name: WANDB_API_KEY
              valueFrom:
                secretKeyRef:
                  name: rl-training-secrets
                  key: wandb-api-key
                  optional: true

          resources:
            requests:
              cpu: "8"
              memory: "32Gi"
              nvidia.com/gpu: "1"
            limits:
              cpu: "16"
              memory: "64Gi"
              nvidia.com/gpu: "1"

          volumeMounts:
            - name: gcs-credentials
              mountPath: /secrets/gcs
              readOnly: true
            - name: dshm
              mountPath: /dev/shm
            - name: workspace
              mountPath: /workspace

      volumes:
        - name: gcs-credentials
          secret:
            secretName: gcs-service-account
        - name: dshm
          emptyDir:
            medium: Memory
            sizeLimit: 32Gi
        - name: workspace
          emptyDir:
            sizeLimit: 100Gi
```

#### 8.2.3 Ray Cluster

```yaml
# k8s/ray-cluster.yaml
apiVersion: ray.io/v1alpha1
kind: RayCluster
metadata:
  name: rl-training-cluster
  namespace: rl-training
spec:
  rayVersion: '2.9.0'

  headGroupSpec:
    serviceType: ClusterIP
    rayStartParams:
      dashboard-host: '0.0.0.0'
      num-cpus: '0'  # Head doesn't run training
    template:
      spec:
        containers:
          - name: ray-head
            image: ${REGION}-docker.pkg.dev/${PROJECT_ID}/rl-training/ray:latest
            resources:
              requests:
                cpu: "4"
                memory: "8Gi"
              limits:
                cpu: "4"
                memory: "8Gi"
            ports:
              - containerPort: 6379
                name: gcs-server
              - containerPort: 8265
                name: dashboard
              - containerPort: 10001
                name: client

  workerGroupSpecs:
    - groupName: gpu-workers
      replicas: 0
      minReplicas: 0
      maxReplicas: 8
      rayStartParams:
        num-gpus: '1'
      template:
        spec:
          nodeSelector:
            cloud.google.com/gke-accelerator: nvidia-tesla-v100
          tolerations:
            - key: nvidia.com/gpu
              operator: Equal
              value: present
              effect: NoSchedule
          containers:
            - name: ray-worker
              image: ${REGION}-docker.pkg.dev/${PROJECT_ID}/rl-training/ray:latest
              resources:
                requests:
                  cpu: "8"
                  memory: "32Gi"
                  nvidia.com/gpu: "1"
                limits:
                  cpu: "16"
                  memory: "64Gi"
                  nvidia.com/gpu: "1"
              volumeMounts:
                - name: dshm
                  mountPath: /dev/shm
          volumes:
            - name: dshm
              emptyDir:
                medium: Memory
                sizeLimit: 32Gi
```

### 8.3 Cloud Workflow

```yaml
# workflows/rl-training-pipeline.yaml
main:
  params: [event]
  steps:
    - extract:
        assign:
          - bucket: ${event.data.bucket}
          - object: ${event.data.name}
          - projectId: ${sys.get_env("GOOGLE_CLOUD_PROJECT_ID")}
          - region: "us-central1"
        next: filter_trigger

    # Trigger on .isaac_lab_complete marker
    - filter_trigger:
        switch:
          - condition: '${text.match_regex(object, "^scenes/.+/isaac_lab/\\.isaac_lab_complete$")}'
            next: derive
        next: skip

    - derive:
        assign:
          - parts: ${text.split(object, "/")}
          - sceneId: ${parts[1]}
          - jobId: '${"rl_train_" + sceneId + "_" + string(int(sys.now()))}'
        next: check_already_trained

    # Idempotence check
    - check_already_trained:
        try:
          call: googleapis.storage.v1.objects.get
          args:
            bucket: ${bucket}
            object: '${"scenes/" + sceneId + "/rl_training/.rl_training_complete"}'
          result: existingMarker
        except:
          as: e
          steps:
            - check_not_found:
                switch:
                  - condition: '${e.code == 404}'
                    next: load_training_config
                next: raise_error
        next: skip_already_trained

    - skip_already_trained:
        return:
          status: "SKIPPED"
          message: "RL training already completed"

    - raise_error:
        raise: ${e}

    # Load training configuration from scene
    - load_training_config:
        call: googleapis.storage.v1.objects.get
        args:
          bucket: ${bucket}
          object: '${"scenes/" + sceneId + "/isaac_lab/train_cfg.yaml"}'
          alt: "media"
        result: trainConfigRaw
        next: parse_config

    - parse_config:
        assign:
          - numEnvs: 1024  # Default
          - maxIterations: 1500  # Default
        next: submit_training_job

    # Submit to GKE
    - submit_training_job:
        call: googleapis.cloudbuild.v1.projects.builds.create
        args:
          projectId: ${projectId}
          body:
            steps:
              - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
                entrypoint: 'bash'
                args:
                  - '-c'
                  - |
                    gcloud container clusters get-credentials rl-training-cluster \
                      --zone=${region}-a --project=${projectId}

                    # Create training job from template
                    export JOB_ID="${jobId}"
                    export SCENE_ID="${sceneId}"
                    export BUCKET="${bucket}"
                    export NUM_ENVS="${numEnvs}"
                    export MAX_ITERATIONS="${maxIterations}"

                    envsubst < /workspace/k8s/training-job-template.yaml | kubectl apply -f -

                    # Wait for completion
                    kubectl wait --for=condition=complete job/${jobId} \
                      --namespace=rl-training --timeout=24h
            timeout: '86400s'
        result: buildResult
        next: wait_for_completion

    - wait_for_completion:
        # Poll for job completion
        call: sys.sleep
        args:
          seconds: 300
        next: check_job_status

    - check_job_status:
        # Check if training completed
        try:
          call: googleapis.storage.v1.objects.get
          args:
            bucket: ${bucket}
            object: '${"scenes/" + sceneId + "/rl_training/.rl_training_complete"}'
        except:
          as: e
          steps:
            - still_running:
                switch:
                  - condition: '${e.code == 404}'
                    next: wait_for_completion
        next: done

    - done:
        return:
          status: "SUCCESS"
          scene_id: ${sceneId}
          job_id: ${jobId}

    - skip:
        return:
          status: "SKIPPED"
          message: "Not an Isaac Lab completion marker"
```

---

## 9. API Specification

### 9.1 REST API (FastAPI)

```python
# src/api/main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

app = FastAPI(title="Blueprint RL Training Service", version="1.0.0")

# Request/Response Models

class TrainingRequest(BaseModel):
    scene_id: str
    bucket: str
    num_envs: Optional[int] = 1024
    max_iterations: Optional[int] = 1500
    num_gpus: Optional[int] = 1
    priority: Optional[int] = 0

class TrainingResponse(BaseModel):
    job_id: str
    scene_id: str
    status: str
    created_at: datetime

class TrainingStatus(BaseModel):
    job_id: str
    scene_id: str
    status: str  # pending, running, completed, failed
    progress: float
    current_iteration: int
    metrics: Optional[dict]
    error: Optional[str]
    created_at: datetime
    updated_at: datetime

class EvaluationRequest(BaseModel):
    scene_id: str
    bucket: str
    policy_path: str
    num_episodes: Optional[int] = 100
    fixed_seeds: Optional[list[int]] = None

class EvaluationResponse(BaseModel):
    eval_id: str
    scene_id: str
    status: str
    created_at: datetime

class BenchmarkReport(BaseModel):
    eval_id: str
    scene_id: str
    success_rate: float
    success_rate_std: float
    mean_reward: float
    reward_std: float
    per_seed_results: list[dict]
    sim2real_indicators: dict

# Endpoints

@app.post("/training/start", response_model=TrainingResponse)
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start a new RL training job."""
    from .orchestrator import RayOrchestrator, JobConfig

    orchestrator = RayOrchestrator()
    job_config = JobConfig(
        scene_id=request.scene_id,
        bucket=request.bucket,
        num_gpus=request.num_gpus,
        num_envs=request.num_envs,
        max_iterations=request.max_iterations,
        priority=request.priority
    )

    job_id = orchestrator.submit_job(job_config)

    return TrainingResponse(
        job_id=job_id,
        scene_id=request.scene_id,
        status="pending",
        created_at=datetime.utcnow()
    )

@app.get("/training/{job_id}", response_model=TrainingStatus)
async def get_training_status(job_id: str):
    """Get status of a training job."""
    from .orchestrator import RayOrchestrator

    orchestrator = RayOrchestrator()
    status = orchestrator.get_status(job_id)

    if status.status == "not_found":
        raise HTTPException(status_code=404, detail="Job not found")

    return status

@app.post("/training/{job_id}/stop")
async def stop_training(job_id: str):
    """Stop a running training job."""
    from .orchestrator import RayOrchestrator

    orchestrator = RayOrchestrator()
    success = orchestrator.stop_job(job_id)

    if not success:
        raise HTTPException(status_code=404, detail="Job not found or already stopped")

    return {"status": "stopped", "job_id": job_id}

@app.get("/training/{job_id}/metrics")
async def get_training_metrics(job_id: str):
    """Get live training metrics."""
    # Return training curves, loss, reward, etc.
    pass

@app.post("/evaluation/run", response_model=EvaluationResponse)
async def run_evaluation(request: EvaluationRequest, background_tasks: BackgroundTasks):
    """Run evaluation on a trained policy."""
    from .evaluation import EvaluationEngine, EvaluationConfig

    eval_id = f"eval_{request.scene_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

    # Run evaluation in background
    background_tasks.add_task(
        _run_evaluation_task,
        eval_id,
        request.scene_id,
        request.bucket,
        request.policy_path,
        request.num_episodes,
        request.fixed_seeds
    )

    return EvaluationResponse(
        eval_id=eval_id,
        scene_id=request.scene_id,
        status="running",
        created_at=datetime.utcnow()
    )

@app.get("/evaluation/{eval_id}", response_model=BenchmarkReport)
async def get_evaluation_results(eval_id: str):
    """Get evaluation results."""
    # Load and return evaluation report from GCS
    pass

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
```

### 9.2 gRPC API (Optional, for internal use)

```protobuf
// proto/rl_training.proto
syntax = "proto3";

package rl_training;

service RLTrainingService {
  rpc StartTraining(TrainingRequest) returns (TrainingResponse);
  rpc GetTrainingStatus(StatusRequest) returns (TrainingStatus);
  rpc StopTraining(StopRequest) returns (StopResponse);
  rpc StreamMetrics(StatusRequest) returns (stream TrainingMetrics);
  rpc RunEvaluation(EvaluationRequest) returns (EvaluationResponse);
  rpc GetEvaluationResults(EvaluationResultsRequest) returns (BenchmarkReport);
}

message TrainingRequest {
  string scene_id = 1;
  string bucket = 2;
  int32 num_envs = 3;
  int32 max_iterations = 4;
  int32 num_gpus = 5;
}

message TrainingResponse {
  string job_id = 1;
  string status = 2;
}

message TrainingStatus {
  string job_id = 1;
  string status = 2;
  float progress = 3;
  int32 current_iteration = 4;
  map<string, float> metrics = 5;
}

message TrainingMetrics {
  int32 iteration = 1;
  float mean_reward = 2;
  float success_rate = 3;
  float policy_loss = 4;
  float value_loss = 5;
  float fps = 6;
}
```

---

## 10. Training Pipeline

### 10.1 Training Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           Training Pipeline                                   │
│                                                                               │
│  1. TRIGGER                                                                   │
│     └─► EventArc detects .isaac_lab_complete in GCS                          │
│                                                                               │
│  2. LOAD                                                                      │
│     ├─► Download task package from GCS                                       │
│     ├─► Validate env_cfg.py, train_cfg.yaml                                  │
│     └─► Setup Isaac Lab environment                                          │
│                                                                               │
│  3. TRAIN                                                                     │
│     ├─► Initialize PPO agent (ActorCritic network)                           │
│     ├─► Training loop:                                                        │
│     │   ├─► Collect rollouts (parallel envs)                                 │
│     │   ├─► Compute advantages (GAE)                                         │
│     │   ├─► Update policy (PPO clipped objective)                            │
│     │   ├─► Log metrics (TensorBoard, W&B)                                   │
│     │   └─► Checkpoint (every N iterations)                                  │
│     └─► Save final checkpoint                                                 │
│                                                                               │
│  4. EVALUATE                                                                  │
│     ├─► Load best checkpoint                                                  │
│     ├─► Run fixed-seed evaluation                                            │
│     └─► Generate benchmark report                                             │
│                                                                               │
│  5. UPLOAD                                                                    │
│     ├─► Upload policy checkpoint to GCS                                       │
│     ├─► Upload training curves                                                │
│     ├─► Upload evaluation report                                              │
│     └─► Write .rl_training_complete marker                                   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 10.2 PPO Implementation Details

Use **RL-Games** or **RSL-RL** as the PPO implementation. Here's the expected interface:

```python
# src/training/ppo/agent.py

import torch
import torch.nn as nn
from typing import Tuple

class ActorCritic(nn.Module):
    """Actor-Critic network for PPO."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        actor_hidden_dims: list[int] = [256, 256, 128],
        critic_hidden_dims: list[int] = [256, 256, 128],
        activation: str = "elu",
        init_noise_std: float = 1.0
    ):
        super().__init__()

        # Build actor network
        actor_layers = []
        prev_dim = obs_dim
        for hidden_dim in actor_hidden_dims:
            actor_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(activation)
            ])
            prev_dim = hidden_dim
        actor_layers.append(nn.Linear(prev_dim, act_dim))
        self.actor = nn.Sequential(*actor_layers)

        # Build critic network
        critic_layers = []
        prev_dim = obs_dim
        for hidden_dim in critic_hidden_dims:
            critic_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(activation)
            ])
            prev_dim = hidden_dim
        critic_layers.append(nn.Linear(prev_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

        # Action distribution parameters
        self.log_std = nn.Parameter(torch.ones(act_dim) * torch.log(torch.tensor(init_noise_std)))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action mean and value."""
        action_mean = self.actor(obs)
        value = self.critic(obs)
        return action_mean, value

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        action_mean, value = self.forward(obs)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(action_mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value

    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        """Return deterministic action (mean)."""
        action_mean, _ = self.forward(obs)
        return action_mean

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update."""
        action_mean, value = self.forward(obs)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(action_mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, value.squeeze(-1), entropy

    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "elu": nn.ELU(),
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "selu": nn.SELU()
        }
        return activations.get(name, nn.ELU())


class PPOAlgorithm:
    """PPO algorithm implementation."""

    def __init__(
        self,
        actor_critic: ActorCritic,
        clip_param: float = 0.2,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 1.0,
        max_grad_norm: float = 1.0,
        learning_rate: float = 3e-4,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        gamma: float = 0.99,
        lam: float = 0.95
    ):
        self.actor_critic = actor_critic
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.gamma = gamma
        self.lam = lam

        self.optimizer = torch.optim.Adam(actor_critic.parameters(), lr=learning_rate)

    def update(self, rollouts: "RolloutBuffer") -> dict:
        """Update policy using collected rollouts."""
        # Compute advantages using GAE
        advantages = self._compute_gae(rollouts)
        returns = advantages + rollouts.values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        for _ in range(self.num_learning_epochs):
            for batch in rollouts.get_batches(self.num_mini_batches):
                obs, actions, old_log_probs, old_values, batch_advantages, batch_returns = batch

                # Evaluate actions
                new_log_probs, new_values, entropy = self.actor_critic.evaluate(obs, actions)

                # Policy loss (clipped)
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
                value_loss = 0.5 * (batch_returns - new_values).pow(2).mean()

                # Total loss
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates
        }

    def _compute_gae(self, rollouts: "RolloutBuffer") -> torch.Tensor:
        """Compute Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rollouts.rewards)
        last_gae = 0

        for t in reversed(range(rollouts.num_steps)):
            if t == rollouts.num_steps - 1:
                next_value = rollouts.last_values
            else:
                next_value = rollouts.values[t + 1]

            delta = rollouts.rewards[t] + self.gamma * next_value * (1 - rollouts.dones[t]) - rollouts.values[t]
            advantages[t] = last_gae = delta + self.gamma * self.lam * (1 - rollouts.dones[t]) * last_gae

        return advantages
```

---

## 11. Evaluation & Benchmarking

### 11.1 Benchmark Protocol

Every trained policy is evaluated using a standardized protocol:

1. **Fixed Seeds**: `[42, 123, 456, 789, 1024]` (configurable)
2. **Episodes per Seed**: 20 (100 total by default)
3. **Deterministic Evaluation**: No action noise
4. **Metrics Collected**:
   - Success rate
   - Mean reward
   - Episode length
   - Task-specific metrics
   - Sim2Real indicators

### 11.2 Sim2Real Indicators

Critical metrics for predicting real-world performance:

| Metric | Description | Target |
|--------|-------------|--------|
| `action_smoothness` | 1 - normalized jerk | > 0.9 |
| `mean_action_jerk` | Average action change | < 0.02 |
| `max_joint_acceleration` | Peak acceleration | < 50 rad/s² |
| `contact_force_variance` | Grasp stability | < 10 N |

### 11.3 Benchmark Report Format

```json
{
  "benchmark_version": "1.0.0",
  "eval_id": "eval_kitchen_001_20260108",
  "scene_id": "kitchen_001",
  "policy_id": "dish_loading",
  "robot_type": "franka",

  "evaluation_config": {
    "num_episodes": 100,
    "fixed_seeds": [42, 123, 456, 789, 1024],
    "deterministic": true,
    "num_envs": 64
  },

  "summary": {
    "success_rate": 0.87,
    "success_rate_95ci": [0.82, 0.92],
    "mean_reward": 85.3,
    "reward_95ci": [78.1, 92.5],
    "mean_episode_length": 312
  },

  "per_seed_results": [
    {"seed": 42, "success_rate": 0.88, "mean_reward": 86.1},
    {"seed": 123, "success_rate": 0.85, "mean_reward": 82.3},
    {"seed": 456, "success_rate": 0.90, "mean_reward": 89.2},
    {"seed": 789, "success_rate": 0.84, "mean_reward": 81.5},
    {"seed": 1024, "success_rate": 0.88, "mean_reward": 87.4}
  ],

  "sim2real_indicators": {
    "action_smoothness": 0.92,
    "mean_action_jerk": 0.015,
    "max_joint_acceleration": 12.3,
    "gripper_force_consistency": 0.88
  },

  "task_specific_metrics": {
    "dishes_placed_mean": 4.2,
    "grasp_attempts_mean": 5.1,
    "collision_count_mean": 0.3
  },

  "comparison_to_baseline": {
    "baseline_success_rate": 0.65,
    "improvement": "+33.8%"
  }
}
```

---

## 12. Implementation Guide

### 12.1 Development Environment Setup

```bash
# Clone repository
git clone https://github.com/yourorg/blueprint-rl-training.git
cd blueprint-rl-training

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Install Isaac Lab (requires NVIDIA GPU)
# Follow: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html

# Setup pre-commit hooks
pre-commit install
```

### 12.2 Local Development

```bash
# Run unit tests
pytest tests/unit -v

# Run integration tests (requires GPU)
pytest tests/integration -v

# Run local training (single GPU)
python -m rl_training.cli train \
  --scene-id kitchen_001 \
  --bucket my-bucket \
  --local \
  --num-envs 64 \
  --max-iterations 100

# Run evaluation
python -m rl_training.cli evaluate \
  --scene-id kitchen_001 \
  --bucket my-bucket \
  --policy-path policies/kitchen_dish_loading/policy.pt
```

### 12.3 Docker Build

```dockerfile
# Dockerfile
FROM nvcr.io/nvidia/isaac-sim:4.0.0-isaaclab

# Install Python dependencies
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

# Copy source code
COPY src/ /app/src/
COPY pyproject.toml /app/

# Install package
WORKDIR /app
RUN pip install -e .

# Set entrypoint
ENTRYPOINT ["python", "-m", "rl_training.cli"]
```

### 12.4 Deployment

```bash
# Build and push Docker image
gcloud builds submit --config cloudbuild.yaml

# Deploy Ray cluster to GKE
kubectl apply -f k8s/ray-cluster.yaml

# Deploy API service
kubectl apply -f k8s/api-deployment.yaml

# Setup EventArc trigger
gcloud eventarc triggers create rl-training-trigger \
  --location=us-central1 \
  --service-account="${WORKFLOW_SA}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --destination-workflow=rl-training-pipeline \
  --destination-workflow-location=us-central1 \
  --event-filters="type=google.cloud.storage.object.v1.finalized" \
  --event-filters="bucket=${BUCKET}"
```

---

## 13. Repository Structure

```
blueprint-rl-training/
├── README.md
├── pyproject.toml
├── requirements.txt
├── Dockerfile
├── cloudbuild.yaml
│
├── src/
│   └── rl_training/
│       ├── __init__.py
│       ├── cli.py                      # CLI entrypoint
│       │
│       ├── api/                        # REST API
│       │   ├── __init__.py
│       │   ├── main.py                 # FastAPI app
│       │   ├── models.py               # Pydantic models
│       │   └── routes/
│       │       ├── training.py
│       │       └── evaluation.py
│       │
│       ├── task_loader/               # Load from BlueprintPipeline
│       │   ├── __init__.py
│       │   ├── loader.py              # GCS download & validation
│       │   ├── validator.py           # Task package validation
│       │   └── schemas.py             # Input schemas
│       │
│       ├── training/                  # Training engine
│       │   ├── __init__.py
│       │   ├── engine.py              # Main training loop
│       │   ├── config.py              # Training configuration
│       │   ├── callbacks.py           # Training callbacks
│       │   └── ppo/
│       │       ├── __init__.py
│       │       ├── agent.py           # ActorCritic network
│       │       ├── algorithm.py       # PPO algorithm
│       │       └── rollout_buffer.py  # Experience storage
│       │
│       ├── evaluation/                # Evaluation engine
│       │   ├── __init__.py
│       │   ├── engine.py              # Evaluation runner
│       │   ├── benchmarks.py          # Benchmark protocols
│       │   └── metrics.py             # Metric computation
│       │
│       ├── orchestrator/              # Job orchestration
│       │   ├── __init__.py
│       │   ├── ray_orchestrator.py    # Ray-based orchestration
│       │   ├── job_manager.py         # Job lifecycle
│       │   └── scheduler.py           # Job scheduling
│       │
│       ├── storage/                   # GCS integration
│       │   ├── __init__.py
│       │   ├── gcs_client.py          # GCS operations
│       │   └── paths.py               # Path conventions
│       │
│       └── utils/
│           ├── __init__.py
│           ├── logging.py             # Logging setup
│           └── metrics.py             # Metric utilities
│
├── k8s/                               # Kubernetes manifests
│   ├── namespace.yaml
│   ├── ray-cluster.yaml
│   ├── training-job-template.yaml
│   ├── api-deployment.yaml
│   └── secrets/
│
├── workflows/                         # Cloud Workflows
│   └── rl-training-pipeline.yaml
│
├── tests/
│   ├── unit/
│   │   ├── test_task_loader.py
│   │   ├── test_training_engine.py
│   │   └── test_evaluation.py
│   ├── integration/
│   │   ├── test_full_pipeline.py
│   │   └── test_gcs_integration.py
│   └── fixtures/
│       └── sample_task_package/
│
├── docs/
│   ├── API.md
│   ├── DEPLOYMENT.md
│   └── BENCHMARKS.md
│
└── scripts/
    ├── setup_gke.sh
    ├── deploy.sh
    └── run_local.sh
```

---

## 14. Configuration Files

### 14.1 `pyproject.toml`

```toml
[project]
name = "blueprint-rl-training"
version = "1.0.0"
description = "RL Training-as-a-Service for BlueprintPipeline"
requires-python = ">=3.10"
dependencies = [
    "fastapi>=0.104.0",
    "uvicorn>=0.24.0",
    "pydantic>=2.5.0",
    "ray[default]>=2.9.0",
    "torch>=2.1.0",
    "numpy>=1.24.0",
    "google-cloud-storage>=2.13.0",
    "google-cloud-workflows>=1.12.0",
    "pyyaml>=6.0",
    "tensorboard>=2.15.0",
    "wandb>=0.16.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
    "black>=23.11.0",
    "ruff>=0.1.6",
    "mypy>=1.7.0",
    "pre-commit>=3.6.0",
]

[project.scripts]
rl-training = "rl_training.cli:main"

[tool.black]
line-length = 100
target-version = ["py310"]

[tool.ruff]
line-length = 100
select = ["E", "F", "I", "N", "W"]

[tool.mypy]
python_version = "3.10"
strict = true

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

### 14.2 `cloudbuild.yaml`

```yaml
substitutions:
  _REGION: us-central1
  _REPO: rl-training
  _IMAGE_NAME: trainer

steps:
  # Build Docker image
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'build'
      - '-t'
      - '${_REGION}-docker.pkg.dev/${PROJECT_ID}/${_REPO}/${_IMAGE_NAME}:${SHORT_SHA}'
      - '-t'
      - '${_REGION}-docker.pkg.dev/${PROJECT_ID}/${_REPO}/${_IMAGE_NAME}:latest'
      - '-f'
      - 'Dockerfile'
      - '.'
    timeout: '3600s'

  # Push to Artifact Registry
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'push'
      - '--all-tags'
      - '${_REGION}-docker.pkg.dev/${PROJECT_ID}/${_REPO}/${_IMAGE_NAME}'

  # Update GKE deployment
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        gcloud container clusters get-credentials rl-training-cluster \
          --zone=${_REGION}-a --project=${PROJECT_ID}
        kubectl set image deployment/rl-training-api \
          api=${_REGION}-docker.pkg.dev/${PROJECT_ID}/${_REPO}/${_IMAGE_NAME}:${SHORT_SHA} \
          --namespace=rl-training

  # Deploy workflow
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    args:
      - 'gcloud'
      - 'workflows'
      - 'deploy'
      - 'rl-training-pipeline'
      - '--location=${_REGION}'
      - '--source=workflows/rl-training-pipeline.yaml'

timeout: '7200s'

options:
  logging: CLOUD_LOGGING_ONLY
  machineType: 'E2_HIGHCPU_8'
```

### 14.3 Environment Variables

```bash
# .env.example

# GCP Configuration
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
GCS_BUCKET=blueprint-scenes

# Training Configuration
DEFAULT_NUM_ENVS=1024
DEFAULT_MAX_ITERATIONS=1500
DEFAULT_LEARNING_RATE=0.0003

# Ray Configuration
RAY_ADDRESS=auto
RAY_NUM_GPUS=1

# Logging
WANDB_API_KEY=your-wandb-key
WANDB_PROJECT=blueprint-rl-training
TENSORBOARD_LOG_DIR=/logs/tensorboard

# API Configuration
API_HOST=0.0.0.0
API_PORT=8080
```

---

## 15. Deployment

### 15.1 Prerequisites

1. GCP Project with:
   - GKE cluster with GPU node pool
   - Artifact Registry repository
   - Cloud Workflows API enabled
   - EventArc API enabled
   - Cloud Storage bucket

2. Service accounts:
   - `rl-training-sa` with Storage Admin, GKE access
   - `workflow-sa` with Workflows Invoker

### 15.2 Initial Setup

```bash
# 1. Create GKE cluster
gcloud container clusters create rl-training-cluster \
  --zone=us-central1-a \
  --machine-type=n2-standard-8 \
  --num-nodes=2 \
  --enable-autoscaling --min-nodes=1 --max-nodes=5

# 2. Add GPU node pool
gcloud container node-pools create gpu-pool \
  --cluster=rl-training-cluster \
  --zone=us-central1-a \
  --machine-type=n1-standard-16 \
  --accelerator=type=nvidia-tesla-v100,count=1 \
  --num-nodes=0 \
  --enable-autoscaling --min-nodes=0 --max-nodes=8

# 3. Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.3/nvidia-device-plugin.yml

# 4. Create namespace and secrets
kubectl apply -f k8s/namespace.yaml
kubectl create secret generic gcs-service-account \
  --from-file=key.json=/path/to/sa-key.json \
  --namespace=rl-training

# 5. Deploy Ray cluster
kubectl apply -f k8s/ray-cluster.yaml

# 6. Build and deploy
gcloud builds submit --config cloudbuild.yaml

# 7. Setup EventArc trigger
gcloud eventarc triggers create rl-training-trigger \
  --location=us-central1 \
  --service-account=workflow-sa@${PROJECT_ID}.iam.gserviceaccount.com \
  --destination-workflow=rl-training-pipeline \
  --destination-workflow-location=us-central1 \
  --event-filters="type=google.cloud.storage.object.v1.finalized" \
  --event-filters="bucket=${BUCKET}"
```

### 15.3 Verification

```bash
# Check Ray cluster
kubectl get rayclusters -n rl-training
kubectl port-forward svc/rl-training-cluster-head-svc 8265:8265 -n rl-training
# Visit http://localhost:8265 for Ray dashboard

# Check API
kubectl port-forward svc/rl-training-api 8080:8080 -n rl-training
curl http://localhost:8080/health

# Test training trigger
gsutil cp test_marker.json gs://${BUCKET}/scenes/test_001/isaac_lab/.isaac_lab_complete
# Watch workflow execution in GCP Console
```

---

## 16. Summary

This document specifies a complete **RL Training-as-a-Service** that:

1. **Triggers** automatically when BlueprintPipeline generates Isaac Lab task packages
2. **Loads** and validates task packages from GCS
3. **Trains** using distributed PPO with 1024+ parallel environments
4. **Evaluates** policies on fixed benchmark seeds
5. **Delivers** policy checkpoints, training curves, and evaluation reports

### Key Integration Points with BlueprintPipeline

| BlueprintPipeline Output | RL Training Service Input |
|--------------------------|---------------------------|
| `scenes/{id}/isaac_lab/env_cfg.py` | Environment configuration |
| `scenes/{id}/isaac_lab/train_cfg.yaml` | Training hyperparameters |
| `scenes/{id}/isaac_lab/reward_functions.py` | Reward implementations |
| `scenes/{id}/isaac_lab/randomizations.py` | Domain randomization |
| `scenes/{id}/isaac_lab/.isaac_lab_complete` | **Trigger file** |

### Key Outputs

| Output | Path | Description |
|--------|------|-------------|
| Policy checkpoint | `scenes/{id}/rl_training/policies/{name}/policy.pt` | Trained PyTorch model |
| Training curves | `scenes/{id}/rl_training/jobs/{job_id}/metrics/training_curves.json` | Loss, reward over time |
| Benchmark report | `scenes/{id}/rl_training/evaluations/{eval_id}/benchmark_report.json` | Standardized evaluation |
| Completion marker | `scenes/{id}/rl_training/.rl_training_complete` | Signals completion |

### Implementation Priority

1. **Phase 1**: Task Loader + Single-GPU Training Engine
2. **Phase 2**: Evaluation Engine + Benchmark Protocol
3. **Phase 3**: Ray Orchestrator + Multi-GPU Support
4. **Phase 4**: API + Cloud Workflow Integration
5. **Phase 5**: Hyperparameter Tuning + Advanced Features

---

*This architecture document is designed to be self-contained. Another AI agent should be able to implement the full service by following this specification.*
