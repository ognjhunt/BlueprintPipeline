# policy_configs

## Purpose / scope
Configuration files that define policy behaviors, environments, and defaults used across the pipeline.

## Index
- `adaptive_timeouts.yaml` adaptive timeout policies.
- `environment_policies.json` environment policy definitions.
- `physics_profiles.json` physics profile definitions.
- `planning_defaults.yaml` planning defaults.
- `retry_policy.yaml` retry/backoff defaults for workflows and local tools.
- `robot_embodiments.json` robot embodiment definitions.

## Primary entrypoints
- YAML/JSON configuration files listed above.

## Required inputs / outputs
- **Inputs:** consumed by pipeline components that read policy configuration.
- **Outputs:** configuration data loaded into runtime.

## Key environment variables
- Environment variables used by consuming services to locate these configs (if applicable).

## How to run locally
- These files are static configs; update them and run the consuming job or tests to validate changes.
