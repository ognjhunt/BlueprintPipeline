# tests

## Purpose / scope
Automated tests covering pipeline integrations, contracts, and job behavior.

## Primary entrypoints
- `pytest` test suite rooted in this directory.
- `conftest.py` shared fixtures.
- `tools/` test helpers.

## Required inputs / outputs
- **Inputs:** test fixtures, mock contracts, and environment configuration.
- **Outputs:** pytest results and reports.

## Key environment variables
- Environment flags and credentials required by integration tests.

## How to run locally
- Run all tests: `pytest`
- Run a subset: `pytest tests/test_pipeline_e2e.py`

