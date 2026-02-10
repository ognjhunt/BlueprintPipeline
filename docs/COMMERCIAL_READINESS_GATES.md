# Commercial Readiness Gates (Frozen Baseline)

## Purpose

This document freezes the release gate thresholds and phase exit criteria for
the "sellable, diverse, sim-ready dataset" objective.

## Effective Date

- Baseline frozen on 2026-02-10.
- Owner approval status: pending explicit sign-off in release review.

## Release Gate (100%)

1. Certification
- `certification_pass_rate >= 0.98` using production profile.
- No recurring dominant critical gate code in run histogram.

2. Import Viability
- `episodes.passed_validation > 0` on release-path runs.
- `quality.average_score >= 0.85`.
- Zero null/unknown `scene_id` in new import manifests.

3. Quality Gate Enforcement
- No `quality_gate_skipped=true` in production mode runs.
- Reports include required/executed/skipped checkpoint coverage.

4. Diversity
- At least 12 canonical scenes.
- At least 4 scene families.
- At least 3 robots (`franka`, `ur5e`, `ur10`) with balanced contribution.
- No single scene contributes more than 25% of accepted episodes.

5. Production Evidence
- `production_mode=true` in production validation artifacts.
- No mock fallback indicators in release-path runs.
- Runtime distributions present (P50/P90/P95/P99) with timeout usage.

6. CI Reliability
- Core data-flow entrypoint tests green, including
  `tests/test_pipeline_data_flow.py::test_pipeline_data_flow_entrypoints`.
- Nightly canary per robot stable for 7 consecutive days before release cut.

## Phase Exit Gates

### Phase 0
- `analysis_outputs/readiness_scorecard.json` generated.
- Scorecard generation wired into CI.

### Phase 1
- Non-null `scene_id` in new manifests (single + combined).
- JSON-backed runs do not fail due missing parquet assumptions.
- Healthy run shows `episodes.passed_validation > 0`.

### Phase 2
- Production runs do not skip quality gates.
- Resume/checkpoint runs still emit complete gate reports.

### Phase 3
- Preprod certification pass rate `>= 0.95`.
- Burn-down evidence for critical certification gate codes.

### Phase 4
- Release matrix summary published with per-scene/per-robot metrics.
- Balance and per-robot success constraints met.

### Phase 5
- Production E2E harness run in strict profile.
- Production validation artifacts and runtime SLO evidence populated.

### Phase 6
- Strict-cert regression suite present and green.
- Nightly canary per robot active.
- 7-day stability gate enforced pre-release.
