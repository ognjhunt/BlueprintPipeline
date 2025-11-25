# SAM 3D job

This job reconstructs SAM 3D objects from asset crops using the upstream
[`facebookresearch/sam-3d-objects`](https://github.com/facebookresearch/sam-3d-objects)
implementation.

## Model assets

* Inference code is baked into the image from the SAM 3D Objects repository under
  `/app/sam3d-objects`.
* Checkpoints are expected under `/app/sam3d-assets/checkpoints/hf/pipeline.yaml` and
  associated weights (`*.ckpt`/`*.safetensors`).
* At runtime, `run_sam3d_from_assets.py` will download the
  `facebook/sam-3d-objects` model from Hugging Face (revision controlled via
  `SAM3D_HF_REVISION`, default branch uses the `hf` pipeline) if no local
  `pipeline.yaml` is found.

Authentication for Hugging Face is provided via `HUGGINGFACE_TOKEN` or
`HF_TOKEN`. The download step only requests files under `checkpoints/hf/` to keep
network usage minimal.

## Configuration

* `SAM3D_CONFIG_PATH`: Override to point at a specific `pipeline.yaml`.
* `SAM3D_REPO_ROOT`/`SAM3D_CHECKPOINT_ROOT`: Override baked-in inference code or
  checkpoint locations if you prefer external volumes.
* `SAM3D_HF_REPO` and `SAM3D_HF_REVISION`: Control which model repository and
  revision are used when downloading checkpoints. Use a release tag that matches
  your desired SAM 3D Objects version.

The runner validates that `pipeline.yaml` exists and that weights are co-located
before starting inference, emitting actionable logs if anything is missing.
