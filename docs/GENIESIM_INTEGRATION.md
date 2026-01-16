# Genie Sim Integration Checklist

Use this checklist when wiring Genie Sim into a new environment (local, staging, or CI). It captures the required ports, credentials, and health checks needed before running export/submit/import flows.

## Required ports

- [ ] **gRPC service reachable** on `GENIESIM_HOST:GENIESIM_PORT` (default `localhost:50051`).
  - If running via Docker Compose, verify the port mapping in `docker-compose.geniesim-server.yaml` and confirm that the host is reachable from the caller.
- [ ] **Isaac Sim runtime** is reachable locally (or inside the container) via `ISAAC_SIM_PATH` pointing to `python.sh`.
- [ ] **Firewall/host rules** allow inbound gRPC traffic to the Genie Sim server port from the job runner, test container, or CI agent.

## Credentials & environment variables

- [ ] **NGC API key** available when pulling NVIDIA Isaac Sim images (`NGC_API_KEY`).
- [ ] **Staging secrets** in CI for the staging workflow:
  - [ ] `STAGING_DATA_ROOT` for scene data root.
  - [ ] `STAGING_SCENE_ID` (or a workflow input override).
  - [ ] `STAGING_ENVIRONMENT_TYPE` (defaults to `kitchen`).
- [ ] **Genie Sim repo path** set locally via `GENIESIM_ROOT` when running the pipeline outside CI.
- [ ] **gRPC Python dependency** (`grpcio`) installed in the Python environment used by the adapter.

## Server health checks

- [ ] **Run the health check module** before submit/import steps:
  - `python -m tools.geniesim_adapter.geniesim_healthcheck`
  - This uses the Genie Sim gRPC client generated in `tools/geniesim_adapter/geniesim_grpc_pb2.py` and `tools/geniesim_adapter/geniesim_grpc_pb2_grpc.py`.
- [ ] **Verify the server bootstrap** behavior in `tools/geniesim_adapter/geniesim_server.py` when running local or containerized deployments.
- [ ] **Confirm adapter wiring** for export/submit/import flows in:
  - `tools/geniesim_adapter/exporter.py`
  - `tools/geniesim_adapter/local_framework.py`
  - `tools/geniesim_adapter/task_config.py`
  - `tools/geniesim_adapter/scene_graph.py`
  - `tools/geniesim_adapter/asset_index.py`

## Staging workflow reference

- [ ] **Review the staging E2E workflow** for expected env vars, Docker Compose usage, and health checks:
  - `.github/workflows/geniesim-staging-e2e.yml`
- [ ] **Match the staging run sequence**:
  - Start gRPC server via `docker compose -f docker-compose.geniesim-server.yaml up -d`.
  - Wait for readiness with `python -m tools.geniesim_adapter.geniesim_healthcheck`.
  - Execute `pytest tests/test_geniesim_staging_e2e.py` inside Isaac Sim.

## Preflight recap (quick checklist)

- [ ] `GENIESIM_HOST` and `GENIESIM_PORT` accessible from the runner.
- [ ] `ISAAC_SIM_PATH` points to `python.sh`.
- [ ] `grpcio` installed.
- [ ] `GENIESIM_ROOT` set for local runs.
- [ ] Health check passes: `python -m tools.geniesim_adapter.geniesim_healthcheck`.
