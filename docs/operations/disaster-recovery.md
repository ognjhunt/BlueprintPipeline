# Disaster recovery (RTO/RPO)

## Targets

| Component | RTO | RPO | Notes |
| --- | --- | --- | --- |
| Workflow orchestration | 30 minutes | 15 minutes | Regional failover via Workflows + Cloud Run jobs. |
| GKE GPU workloads | 2 hours | 30 minutes | Secondary GKE cluster must be pre-provisioned. |
| GCS pipeline artifacts | 1 hour | 15 minutes | Multi/dual-region buckets reduce data loss. |
| Monitoring/alerting | 1 hour | 30 minutes | Managed services; relies on regional availability. |

## Assumptions

- Primary and secondary regions are pre-provisioned and tested.
- Multi-region or dual-region GCS buckets are used for pipeline data.
- Critical GKE jobs are deployed to the secondary cluster.
- Workflow region routing is configured with primary/secondary regions.

## Recovery procedure

1. **Detect and confirm regional impact**
   - Review workflow/job errors for the primary region.
   - Check GKE cluster status and GPU node availability.

2. **Failover orchestration**
   - Update `PRIMARY_WORKFLOW_REGION` / `SECONDARY_WORKFLOW_REGION` if the primary is impaired.
   - Ensure Eventarc triggers are pointing at the active workflow region.

3. **Failover GKE workloads**
   - Switch `PRIMARY_GKE_CLUSTER` / `PRIMARY_GKE_ZONE` to the secondary cluster values.
   - Deploy the secondary-region kustomize overlay (see `k8s/secondary-region/`).

4. **Validate data integrity**
   - Confirm object markers exist for in-flight scenes.
   - Re-run failed workflows if needed (idempotent markers should prevent duplication).

5. **Return to primary**
   - Restore primary regions once healthy.
   - Backfill any missing outputs and reconcile job logs.

## Testing cadence

- Quarterly failover drill covering workflows, GKE jobs, and GCS access.
- Monthly validation of Eventarc trigger routing and bucket replication settings.
