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
