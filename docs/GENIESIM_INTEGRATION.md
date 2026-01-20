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
- [ ] **Optional runtime cap** configured via `GENIESIM_COLLECTION_TIMEOUT_S` to bound local data collection duration.
- [ ] **Firebase uploads (submit/import)** have credentials available in production or service mode:
  - `FIREBASE_STORAGE_BUCKET`
  - `FIREBASE_SERVICE_ACCOUNT_JSON` or `FIREBASE_SERVICE_ACCOUNT_PATH`
  - Optional prefix: `FIREBASE_UPLOAD_PREFIX` (used by submit/import).
  - `ENABLE_FIREBASE_UPLOAD=true` to force uploads outside production/service mode (import defaults to enabled in production/service mode).

### Migration note: Firebase upload prefix rename

The Firebase upload prefix is now standardized on `FIREBASE_UPLOAD_PREFIX` for both submit and import. If you previously set `FIREBASE_EPISODE_PREFIX`, update your environment, workflow, and secret/configmap values to avoid mismatched dataset paths in downstream consumers.

## Secure gRPC configuration

Use these environment variables to enable TLS, auth metadata, and circuit breaker behavior in the adapter:

- **TLS channel configuration**
  - `GENIESIM_TLS_CERT`: path to client certificate (PEM).
  - `GENIESIM_TLS_KEY`: path to client private key (PEM).
  - `GENIESIM_TLS_CA`: path to CA bundle (PEM).
- **Auth metadata injection**
  - `GENIESIM_AUTH_TOKEN`: inline JWT or bearer token.
  - `GENIESIM_AUTH_TOKEN_PATH`: path to a file containing the JWT/bearer token.
  - `GENIESIM_AUTH_CERT`: path to a client cert to forward in metadata (`x-geniesim-client-cert`).
  - `GENIESIM_AUTH_KEY`: path to a client key to forward in metadata (`x-geniesim-client-key`).
- **Circuit breaker controls**
  - `GENIESIM_CIRCUIT_BREAKER_FAILURE_THRESHOLD`: number of consecutive failures before tripping (default: 3).
  - `GENIESIM_CIRCUIT_BREAKER_BACKOFF_SECONDS`: backoff window before retry (default: 2.0).

Example secure configuration:

```bash
export GENIESIM_HOST=geniesim.internal
export GENIESIM_PORT=50051
export GENIESIM_TLS_CA=/etc/ssl/certs/geniesim-ca.pem
export GENIESIM_TLS_CERT=/etc/ssl/certs/geniesim-client.pem
export GENIESIM_TLS_KEY=/etc/ssl/private/geniesim-client-key.pem
export GENIESIM_AUTH_TOKEN_PATH=/etc/secrets/geniesim.jwt
```

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

## Local mock server behavior

When using the lightweight local mock server (`tools/geniesim_adapter/geniesim_server.py`) for development or CI, the servicer provides deterministic, non-error responses for core telemetry endpoints:

- **GetIKStatus** returns `success=true`, `ik_solvable=true`, and a mock solution based on the current joint state (or seed positions when supplied).
- **GetTaskStatus** returns `success=true` with a basic `status` and `progress` value for quick availability checks.
- **StreamObservations** yields a short sequence of static `GetObservationResponse` frames to emulate streaming updates.

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
export PIPELINE_ENV=production
export ISAACSIM_REQUIRED=true
export CUROBO_REQUIRED=true
```

Production runs that generate asset indexes must also configure a valid embedding provider
(`OPENAI_API_KEY` or `QWEN_API_KEY`/`DASHSCOPE_API_KEY` plus the matching embedding model).
