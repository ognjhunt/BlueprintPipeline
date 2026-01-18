# Genie Sim Integration Checklist

Use this checklist when wiring Genie Sim into a local environment. It captures the required ports, credentials, and health checks needed before running export/submit/import flows.

## Required ports

- [ ] **gRPC service reachable** on `GENIESIM_HOST:GENIESIM_PORT` (default `localhost:50051`).
  - If running via Docker Compose, verify the port mapping in `docker-compose.geniesim-server.yaml` and confirm the host is reachable from the local runner.
- [ ] **Isaac Sim runtime** is reachable locally (or inside the container) via `ISAAC_SIM_PATH` pointing to `python.sh`.

## Credentials & environment variables

- [ ] **NGC key** available when pulling NVIDIA Isaac Sim images (`NGC_API_KEY`).
- [ ] **Genie Sim repo path** set locally via `GENIESIM_ROOT` when running the pipeline outside CI.
- [ ] **gRPC Python dependency** (`grpcio`) installed in the Python environment used by the adapter.
- [ ] **Production deployments** explicitly set `GENIESIM_ENV=production` and enable `ISAACSIM_REQUIRED=true` + `CUROBO_REQUIRED=true` to enforce runtime checks.

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

## Preflight recap (quick checklist)

- [ ] `GENIESIM_HOST` and `GENIESIM_PORT` accessible from the runner.
- [ ] `ISAAC_SIM_PATH` points to `python.sh`.
- [ ] `grpcio` installed.
- [ ] `GENIESIM_ROOT` set for local runs.
- [ ] Health check passes: `python -m tools.geniesim_adapter.geniesim_healthcheck`.

## Production toggle

Use `GENIESIM_ENV=production` as the canonical production toggle for Genie Sim integrations, along with required runtime flags:

```bash
export GENIESIM_ENV=production
export ISAACSIM_REQUIRED=true
export CUROBO_REQUIRED=true
```
