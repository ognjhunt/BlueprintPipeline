# genie-sim-import-webhook

## Purpose / scope
Webhook service for ingesting GenieSim import events.

## Primary entrypoints
- `main.py` service entrypoint.
- `Dockerfile` container image definition.

## Required inputs / outputs
- **Inputs:** HTTP webhook payloads from GenieSim import events.
- **Outputs:** downstream triggers or stored payloads as handled by `main.py`.

## Key environment variables
- Service configuration such as listening port, credentials, and downstream endpoint URLs.
- `HEALTHCHECK_ALLOWED_HOSTS`: optional comma-separated list of hostnames permitted for
  `LLM_HEALTH_URL` and `ISAAC_SIM_HEALTH_URL` checks. When set, the health probes only run
  against hosts in this allowlist and still require HTTPS.

## How to run locally
- Build the container: `docker build -t genie-sim-import-webhook .`
- Run the service: `python main.py` (set required env vars before running).
