# blueprint_sim

## Purpose / scope
Core simulation package for blueprint assembly and scene generation.

## Primary entrypoints
- `assembly.py` assembly logic.
- `replicator.py` replication utilities.
- `simready.py` sim-ready processing.
- `manifest.py` manifest helpers.
- `recipe_compiler/` compiler package.

## Required inputs / outputs
- **Inputs:** scene manifests, asset metadata, and recipe definitions consumed by the modules above.
- **Outputs:** assembled simulation-ready artifacts and manifests.

## Key environment variables
- Environment variables controlling input/output locations and pipeline configuration.

## How to run locally
- Use the modules as a package: `python -m blueprint_sim.<module>` or import them from your scripts.

