# Genie Sim Robot Catalog

The Genie Sim robot allowlist lives in `tools/geniesim_adapter/geniesim_robot_catalog.json` so it can
be updated independently of `tools/config/pipeline_config.json`.

## When to update

Update the catalog whenever Genie Sim 3.0 adds, removes, or renames a supported robot. The catalog
is treated as the source of truth for validation in the submit/import jobs and pipeline config.

## How to update

1. **Edit the catalog JSON**
   - Open `tools/geniesim_adapter/geniesim_robot_catalog.json`.
   - Add or remove entries under `robots`.
   - Keep `name` values lowercase and snake_case (matching Genie Sim identifiers).
   - Update `geniesim_version` and `updated_at` as needed.

2. **Keep specs in sync**
   - If you add/remove a robot, update `tools/geniesim_adapter/multi_robot_config.py` so
     `RobotType`/`ROBOT_SPECS` match the catalog.

3. **Verify pipeline config**
   - Ensure `tools/config/pipeline_config.json` continues to reference the catalog via
     `robot_config.geniesim_robot_catalog_path`.
   - The `robot_config.supported_robots` list should stay aligned with the catalog.

4. **Run tests**
   - Run the unit tests that validate the catalog allowlist (for example, `pytest tests/test_geniesim_robot_catalog.py`).

## Validation behavior

- In production, the submit and import jobs fail fast if an unsupported robot is configured.
- Outside production, the jobs log a warning but continue.
