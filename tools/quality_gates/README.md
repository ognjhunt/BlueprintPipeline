# Quality Gates Notifications

## Production requirements
Quality gate approvals must have at least one notification channel configured when
`PRODUCTION_MODE` (or `PIPELINE_ENV=production`) is enabled. The pipeline will
raise an error if `human_approval.notification_channels` is empty in production.

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

Email delivery relies on standard provider configuration (e.g. `SENDGRID_API_KEY`
or SMTP settings) as described in `notification_service.py`.

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
