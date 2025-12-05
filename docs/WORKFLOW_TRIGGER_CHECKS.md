# Workflow trigger verification

Use `tools/verify_workflow_triggers_and_dryrun.py` to double-check that the
Eventarc / Workflows plumbing is wired correctly without needing GCP access.

```bash
python tools/verify_workflow_triggers_and_dryrun.py
```

The script performs three checks:

1. **USD assembly trigger** – confirms `workflows/usd-assembly-pipeline.yaml`
   filters on finalized `scenes/*/assets/.hunyuan_complete` objects and invokes
   the convert-only USD assembly job, then `simready-job`, followed by the final
   USD assembly run.
2. **Hunyuan trigger** – confirms `workflows/hunyuan-pipeline.yaml` listens for
   `scene_assets.json` and writes the `.hunyuan_complete` marker, with parity in
   `hunyuan-job/run_hunyuan_from_assets.py`.
3. **Dry run** – simulates a `.hunyuan_complete` finalize event, showing the
   expected sequence of actions (convert-only USD assembly → `simready-job` →
   final USD assembly) that should occur automatically after the completion
   marker appears.

This lightweight validation replaces manual Eventarc trigger inspection when you
just need to assert the wiring is present and the orchestration order is
correct.
