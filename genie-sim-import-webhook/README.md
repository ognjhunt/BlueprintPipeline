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
- Webhook authentication (required in production):
  - **HMAC**: set `WEBHOOK_HMAC_SECRET` and send `X-Webhook-Signature` (hex SHA-256 HMAC) with
    the raw request body.
  - **OIDC**: set `WEBHOOK_OIDC_AUDIENCE` and send a bearer token in the `Authorization` header.
- `HEALTHCHECK_ALLOWED_HOSTS`: optional comma-separated list of hostnames permitted for
  `LLM_HEALTH_URL` and `ISAAC_SIM_HEALTH_URL` checks. When set, the health probes only run
  against hosts in this allowlist and still require HTTPS.

## Webhook auth setup checklist
1. Decide on authentication method (HMAC or OIDC).
2. Set `WEBHOOK_HMAC_SECRET` or `WEBHOOK_OIDC_AUDIENCE` in the deployment environment.
3. Update deployment manifests or Cloud Run settings to include these env vars.
4. Validate a signed webhook request reaches `POST /webhooks/geniesim/job-complete`
   successfully (example: send a payload with the appropriate signature or bearer token).

## How to run locally
- Build the container: `docker build -t genie-sim-import-webhook .`
- Run the service: `python main.py` (set required env vars before running).
