# VM Scope Remediation (2026-02-10)

## Target
- Instance: `isaac-sim-ubuntu`
- Zone: `us-east1-c`
- Service account: `744608654760-compute@developer.gserviceaccount.com`

## Before
Command:
```bash
gcloud compute instances describe isaac-sim-ubuntu \
  --zone=us-east1-c \
  --format='json(status,serviceAccounts,machineType,guestAccelerators)'
```

Result (excerpt):
- `status`: `TERMINATED`
- scopes:
  - `https://www.googleapis.com/auth/devstorage.read_only`
  - `https://www.googleapis.com/auth/logging.write`
  - `https://www.googleapis.com/auth/monitoring.write`
  - `https://www.googleapis.com/auth/service.management.readonly`
  - `https://www.googleapis.com/auth/servicecontrol`
  - `https://www.googleapis.com/auth/trace.append`

## Remediation Commands
```bash
gcloud compute instances stop isaac-sim-ubuntu --zone=us-east1-c

gcloud compute instances set-service-account isaac-sim-ubuntu \
  --zone=us-east1-c \
  --service-account=744608654760-compute@developer.gserviceaccount.com \
  --scopes=cloud-platform

gcloud compute instances start isaac-sim-ubuntu --zone=us-east1-c
```

## After
Command:
```bash
gcloud compute instances describe isaac-sim-ubuntu \
  --zone=us-east1-c \
  --format='json(status,serviceAccounts)'
```

Result (excerpt):
- `status`: `RUNNING`
- scopes:
  - `https://www.googleapis.com/auth/cloud-platform`

## Outcome
- VM scopes successfully remediated from `devstorage.read_only` to `cloud-platform`.
