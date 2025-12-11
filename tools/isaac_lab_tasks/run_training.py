"""Minimal training entry point for generated Isaac Lab tasks.

This script is intentionally lightweight so it can run inside an Isaac Lab
workstation without additional tooling. It wires the generated environment
configuration, builds the ManagerBasedEnv, and executes a brief rollout with
zero actions to validate wiring before attaching a full RL trainer.
"""

from __future__ import annotations

import argparse
import importlib
from typing import Any

import torch
from omni.isaac.lab.app import AppLauncher
from omni.isaac.lab.envs import ManagerBasedEnv
from omni.isaac.lab_tasks.utils import parse_env_cfg


def _load_env_cfg(task_package: str, env_cfg_class: str, num_envs: int) -> Any:
    """Dynamically import an env_cfg class and override num_envs."""

    module = importlib.import_module(task_package)
    cfg_cls = getattr(module, env_cfg_class)
    cfg = cfg_cls()
    cfg.scene.num_envs = num_envs
    return cfg


def rollout_sanity_check(env: ManagerBasedEnv, steps: int = 4):
    """Run a short rollout with zero actions to verify managers are wired."""

    env.reset()
    action_shape = env.action_manager.action_spec.shape
    actions = torch.zeros((env.num_envs, action_shape[0]), device=env.device)

    for _ in range(steps):
        obs, rewards, dones, info = env.step(actions)
        if torch.any(dones):
            env.reset(torch.nonzero(dones).squeeze(-1))

    return obs, rewards


def main():
    parser = argparse.ArgumentParser(description="Run a generated Isaac Lab task")
    parser.add_argument(
        "--task-package",
        default="examples.kitchen_recipe_pack.isaac_lab",
        help="Python package containing env_cfg.py",
    )
    parser.add_argument("--env-cfg", default="KitchenPickPlaceEnvCfg", help="EnvCfg class name to instantiate")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of vectorized environments")
    parser.add_argument("--rollout-steps", type=int, default=8, help="Steps for the sanity rollout")
    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    app_launcher.start()

    env_cfg = parse_env_cfg(_load_env_cfg(args.task_package, args.env_cfg, args.num_envs))
    env = ManagerBasedEnv(env_cfg)

    obs, rewards = rollout_sanity_check(env, steps=args.rollout_steps)
    print("Sanity rollout complete", {"obs_keys": list(obs.keys()), "reward_terms": list(rewards.keys())})


if __name__ == "__main__":
    main()
