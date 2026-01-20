# Genie Sim version pinning

The Genie Sim GPU job builds against a pinned Genie Sim ref. The canonical value
lives in `genie-sim-gpu-job/GENIESIM_REF` and is used in two places:

1. **Docker build default**: the Dockerfile reads `GENIESIM_REF` to default the
   `ARG GENIESIM_REF` value when no build arg is provided.
2. **CI build arg**: `.github/workflows/docker-build.yml` reads the same file and
   passes `GENIESIM_REF` into the Docker build.

## Updating the pinned ref

1. Pick a full git SHA or release tag from `AgibotTech/genie_sim`.
2. Update `genie-sim-gpu-job/GENIESIM_REF` with the new value.
3. (Optional) Override during a local build:
   ```bash
   docker build --build-arg GENIESIM_REF=<sha-or-tag> -f genie-sim-gpu-job/Dockerfile genie-sim-gpu-job
   ```

CI validates that the ref is either a full 40-character SHA or a tag-like value
(before building the image).

## Isaac Sim base image version

The Isaac Sim base image tag is controlled by `ARG ISAAC_SIM_VERSION` in
`genie-sim-gpu-job/Dockerfile`. Update the default value in the Dockerfile to
pin a new version, or override it at build time with
`--build-arg ISAAC_SIM_VERSION=<version>`. CI validates that the build arg
default is present and non-empty.
