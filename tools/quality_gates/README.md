# Quality Gates Notifications

## Production requirements
Quality gate approvals must have at least one notification channel configured when
`PRODUCTION_MODE` (or `PIPELINE_ENV=production`) is enabled. The pipeline will
raise an error if `human_approval.notification_channels` is empty in production.

The primary pipeline entrypoints (for example `tools/run_local_pipeline.py`,
`tools/run_scene_batch.py`, and `tools/quality_gates/sli_gate_runner.py`) now
construct a default `NotificationService` from `quality_config.json` and
environment overrides. The service is passed into
`QualityGateRegistry.run_checkpoint(...)` / `run_checkpoint_with_approval(...)`
so `notify_on_fail` and `notify_on_pass` gates emit alerts.

Configure channels in `tools/quality_gates/quality_config.json`, for example:

```json
{
  "human_approval": {
    "notification_channels": ["email", "slack"]
  }
}
```

### Environment variable overrides
You can override notification configuration without editing JSON:

| Purpose | Environment variable | Example |
| --- | --- | --- |
| Channels | `BP_QUALITY_HUMAN_APPROVAL_NOTIFICATION_CHANNELS` | `email,slack` |
| Email recipient | `QA_EMAIL` | `qa-team@example.com` |
| Slack webhook | `BP_QUALITY_NOTIFICATIONS_SLACK_WEBHOOK_URL` | `https://hooks.slack.com/services/...` |
| SMS recipient | `QA_PHONE` | `15551234567` |
| Override email recipients | `BP_QUALITY_NOTIFICATIONS_EMAIL_RECIPIENTS` | `qa-team@example.com,qa-lead@example.com` |
| Override SMS recipients | `BP_QUALITY_NOTIFICATIONS_SMS_RECIPIENTS` | `15551234567,15557654321` |

When multiple recipients are supplied, the notification service uses the first
entry for delivery (configure a shared inbox or distribution list for emails
if you need multiple recipients).

Email delivery relies on standard provider configuration (e.g. `SENDGRID_API_KEY`
or SMTP settings) as described in `notification_service.py`.

## Genie Sim kinematic reachability gate
`geniesim_kinematic_reachability` validates that task target poses in
`task_config.json` are reachable by the configured robot IK solver. Configure
the thresholds in `tools/quality_gates/quality_config.json`:

```json
{
    "thresholds": {
    "geniesim_kinematic_reachability": {
      "enabled": true,
      "min_reachability_rate": 1.0,
      "max_unreachable_targets": 0,
      "check_place_targets": true
    }
  }
}
```

In production, a failed reachability gate blocks submission unless a manual
override is granted via the quality gate approval workflow. Ensure your override
metadata complies with `gate_overrides` in `quality_config.json` (including
`allowed_overriders`) so the audit entry is recorded.

The default policy requires full reachability: no unreachable targets are
allowed, and the minimum reachability rate is set to `1.0`.

Automated override payloads can be provided via `BP_QUALITY_OVERRIDE_METADATA`
as JSON. The payload must include `approver_id`, `category`, `ticket`, and
`justification` (a timestamp is auto-filled when omitted). Example:

```json
{
  "approver_id": "ops@example.com",
  "category": "known_issue",
  "ticket": "OPS-1234",
  "justification": "Known IK modeling gap; reviewed by robotics team."
}
```

## Approval storage (production)
Production deployments should use the Firestore approval store and a persistent
filesystem path for any local fallback or migration data. The loader prefers
Firestore when available and only falls back to filesystem storage for local
runs. The default filesystem path is `/var/lib/blueprintpipeline/approvals`.
Configure a writable directory and set:

```bash
export QUALITY_APPROVAL_PATH=/var/lib/blueprintpipeline/approvals
```

`QUALITY_APPROVAL_PATH` overrides `approval_store.filesystem_path` from
`tools/quality_gates/quality_config.json` and should point to persistent storage
in production.

For local development, override the default with a per-user or temporary path,
for example:

```bash
export QUALITY_APPROVAL_PATH=~/.blueprintpipeline/approvals
```

You can also use `BP_QUALITY_APPROVAL_STORE_FILESYSTEM_PATH` to set the same
value in environments where `QUALITY_APPROVAL_PATH` is not preferred.
