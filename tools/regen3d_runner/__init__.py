"""3D-RE-GEN runner — execute 3D scene reconstruction on a remote GPU VM.

This module provides the integration between BlueprintPipeline and the
3D-RE-GEN reconstruction system (arXiv:2512.17459).

Usage:
    from tools.regen3d_runner import Regen3DRunner, Regen3DConfig

    config = Regen3DConfig.from_env()
    runner = Regen3DRunner(config)
    result = runner.run_reconstruction(
        input_image=Path("scenes/my_scene/input/room.jpg"),
        scene_id="my_scene",
        output_dir=Path("scenes/my_scene/regen3d"),
    )

Reference:
    - Paper: https://arxiv.org/abs/2512.17459
    - Project: https://3dregen.jdihlmann.com/
    - GitHub: https://github.com/cgtuebingen/3D-RE-GEN
"""

from tools.regen3d_runner.runner import (
    Regen3DConfig,
    Regen3DRunner,
    ReconstructionResult,
)
from tools.regen3d_runner.output_harvester import (
    harvest_regen3d_native_output,
    HarvestResult,
)
from tools.regen3d_runner.vm_executor import (
    VMConfig,
    VMExecutor,
    VMExecutorError,
    SSHConnectionError,
    CommandTimeoutError,
    GPUNotAvailableError,
)

__all__ = [
    "Regen3DConfig",
    "Regen3DRunner",
    "ReconstructionResult",
    "harvest_regen3d_native_output",
    "HarvestResult",
    "VMConfig",
    "VMExecutor",
    "VMExecutorError",
    "SSHConnectionError",
    "CommandTimeoutError",
    "GPUNotAvailableError",
]
