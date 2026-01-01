# Workflow Trigger Verification

Use `tools/verify_workflow_triggers_and_dryrun.py` to double-check that the
Eventarc / Workflows plumbing is wired correctly without needing GCP access.

```bash
python tools/verify_workflow_triggers_and_dryrun.py
```

The script performs checks on:

1. **USD assembly trigger** – confirms `workflows/usd-assembly-pipeline.yaml`
   filters on finalized `scenes/*/assets/.regen3d_complete` objects and invokes
   the simready-job, followed by the final USD assembly run.
2. **3D-RE-GEN trigger** – confirms the regen3d workflow listens for
   3D-RE-GEN outputs and writes the `.regen3d_complete` marker.
3. **Dry run** – simulates a `.regen3d_complete` finalize event, showing the
   expected sequence of actions (simready-job → USD assembly → replicator-job)
   that should occur automatically after the completion marker appears.

This lightweight validation replaces manual Eventarc trigger inspection when you
just need to assert the wiring is present and the orchestration order is
correct.
