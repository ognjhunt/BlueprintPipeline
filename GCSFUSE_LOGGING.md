# GCSFuse Logging Configuration

This document explains how to enable detailed GCSFuse logging for Cloud Run Jobs.

## Background

Cloud Run Jobs automatically mount GCS buckets using GCSFuse when volume mounts are configured. By default, GCSFuse runs with minimal logging. The log message:

```
GCSFuse is mounted with bucket blueprint-8c1ca.appspot.com. Enable GCSFuse logging with mount options for more logs
```

indicates that GCSFuse is running but not configured with detailed logging.

## Why Enable GCSFuse Logging?

Detailed GCSFuse logs can help diagnose:
- File access performance issues
- Cache hit/miss patterns
- Network errors during file operations
- Mount/unmount problems
- Permission issues

## How to Enable GCSFuse Logging

### Option 1: Using gcloud CLI (Recommended)

Update the Cloud Run Job to add GCSFuse mount options with logging enabled:

```bash
# For multiview-job
gcloud run jobs update multiview-job \
  --region=us-central1 \
  --update-volume-mounts name=gcs-volume,mount-path=/mnt/gcs \
  --set-volume-flags name=gcs-volume,log-format=text,log-severity=info,file-mode=755

# For seg-job
gcloud run jobs update seg-job \
  --region=us-central1 \
  --update-volume-mounts name=gcs-volume,mount-path=/mnt/gcs \
  --set-volume-flags name=gcs-volume,log-format=text,log-severity=info,file-mode=755

# For sam3d-job
gcloud run jobs update sam3d-job \
  --region=us-central1 \
  --update-volume-mounts name=gcs-volume,mount-path=/mnt/gcs \
  --set-volume-flags name=gcs-volume,log-format=text,log-severity=info,file-mode=755
```

### GCSFuse Mount Options

Common GCSFuse mount options for logging:

- `log-format`: Format of logs (`text` or `json`)
- `log-severity`: Logging level (`info`, `debug`, `warning`, `error`)
- `log-file`: Optional path to write logs to a file
- `debug_fuse`: Enable FUSE-level debug logging (very verbose)
- `debug_gcs`: Enable GCS API debug logging (very verbose)

**Warning**: Using `debug_fuse` or `debug_gcs` generates extremely verbose logs and may impact performance.

### Recommended Configuration for Production

For production use, enable INFO-level logging without debug flags:

```bash
--set-volume-flags name=gcs-volume,log-format=text,log-severity=info
```

### Recommended Configuration for Debugging

For debugging specific issues, use DEBUG-level logging temporarily:

```bash
--set-volume-flags name=gcs-volume,log-format=text,log-severity=debug,debug_gcs=true
```

**Remember to revert to INFO level after debugging to avoid log volume issues.**

### Option 2: Using Terraform

If you're using Terraform to manage Cloud Run Jobs, add the volume mount options:

```hcl
resource "google_cloud_run_v2_job" "multiview_job" {
  name     = "multiview-job"
  location = "us-central1"

  template {
    template {
      volumes {
        name = "gcs-volume"
        gcs {
          bucket    = var.bucket_name
          read_only = false
          mount_options = [
            "log-format=text",
            "log-severity=info",
            "file-mode=755"
          ]
        }
      }

      containers {
        image = "gcr.io/${var.project_id}/multiview-job:latest"

        volume_mounts {
          name       = "gcs-volume"
          mount_path = "/mnt/gcs"
        }
      }
    }
  }
}
```

### Option 3: Using Cloud Console

1. Go to Cloud Run in the Google Cloud Console
2. Select the job (e.g., `multiview-job`)
3. Click "Edit & Deploy New Revision"
4. Navigate to "Volumes" section
5. Edit the GCS volume mount
6. Add mount options:
   - `log-format=text`
   - `log-severity=info`
7. Save and deploy

## Verifying GCSFuse Logging

After enabling logging, check the Cloud Run Job logs:

```bash
# View logs for a specific job execution
gcloud logging read "resource.type=cloud_run_job AND resource.labels.job_name=multiview-job" \
  --limit=100 \
  --format=json

# Filter for GCSFuse-specific logs
gcloud logging read "resource.type=cloud_run_job AND jsonPayload.message=~'GCSFuse'" \
  --limit=50
```

You should see additional GCSFuse log entries like:

```
time="2025-11-30 01:17:48" severity=INFO message="Creating Storage handle..."
time="2025-11-30 01:17:48" severity=INFO message="UserAgent = gcsfuse/3.4.4..."
time="2025-11-30 01:17:48" severity=INFO message="Set up root directory for bucket..."
```

## Performance Considerations

- **INFO level**: Minimal performance impact, recommended for production
- **DEBUG level**: Moderate performance impact, use only for troubleshooting
- **debug_fuse/debug_gcs**: Significant performance impact and log volume, use sparingly

## Troubleshooting

### Logs still not showing up

1. Verify the job configuration:
   ```bash
   gcloud run jobs describe multiview-job --region=us-central1 --format=yaml
   ```

2. Check that volume mount options are applied in the output

3. Trigger a new job execution to see updated logs

### Too many logs

If logs are overwhelming:

1. Reduce log severity to `warning` or `error`
2. Disable debug flags
3. Use log filters in Cloud Logging to focus on specific issues

## Additional Resources

- [GCSFuse Documentation](https://github.com/GoogleCloudPlatform/gcsfuse/blob/master/docs/mounting.md)
- [Cloud Run Volume Mounts](https://cloud.google.com/run/docs/configuring/services/cloud-storage-volume-mounts)
- [GCSFuse Mount Options](https://github.com/GoogleCloudPlatform/gcsfuse/blob/master/docs/mounting.md#options)
