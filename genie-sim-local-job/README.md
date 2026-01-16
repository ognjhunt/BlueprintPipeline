# Genie Sim Local Job

This job runs the Genie Sim submit/export tooling locally. Python dependencies are pinned in
`requirements.txt` and constrained via `requirements.lock` for deterministic installs in the
Docker image.

## Updating dependencies

1. Update versions in `tools/requirements-pins.txt` (the single source of truth), then
   run `python tools/sync_requirements_pins.py` to refresh all `requirements.txt` files.
2. Regenerate the lock file using Python 3.11 (to match the Docker base image):

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip pip-tools
   python -m piptools compile --output-file genie-sim-local-job/requirements.lock \
     genie-sim-local-job/requirements.txt
   ```

3. Commit both `requirements.txt` and `requirements.lock` so the Docker build stays
   deterministic.

The Dockerfile installs with:

```bash
pip install -r requirements.txt -c requirements.lock
```
