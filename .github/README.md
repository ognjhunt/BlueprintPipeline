# .github

## Purpose / scope
Houses GitHub-specific configuration for this repository, primarily GitHub Actions workflows that automate CI/CD tasks.

## Primary entrypoints
- `workflows/*.yaml` GitHub Actions workflows executed by GitHub Actions.

## Required inputs / outputs
- **Inputs:** repository events (push, pull request, schedule, workflow dispatch), workflow secrets, and repository files.
- **Outputs:** workflow artifacts, build/test status checks, and logs in the Actions UI.

## Key environment variables
- Standard GitHub Actions variables such as `GITHUB_SHA`, `GITHUB_REF`, `GITHUB_WORKSPACE`, and any repo secrets referenced by workflows.

## How to run locally
- Use a GitHub Actions runner emulator (e.g., `act`) if needed. Otherwise, run the scripts referenced by workflows directly.

