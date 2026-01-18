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
- Run dataset regression checks: `pytest tests/test_dataset_regression.py` (uses thresholds in `tests/fixtures/golden/dataset_regression/thresholds.json` to detect metric drift).

## Test dependencies
- Install test requirements: `pip install -r tests/requirements.txt`.
- The staging and Isaac Sim test suites import `pxr`, so install the OpenUSD
  Python bindings via `usd-core` (see the OpenUSD dependency note in the repo
  `README.md`) before running those tests locally.
