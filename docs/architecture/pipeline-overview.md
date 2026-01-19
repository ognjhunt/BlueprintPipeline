# Pipeline Overview (Mermaid)

This overview diagram aligns with the job naming and flow described in:
- `docs/PIPELINE_ANALYSIS.md`
- `docs/GENIESIM_INTEGRATION.md`
- `docs/PRODUCTION_E2E_VALIDATION.md`

```mermaid
flowchart LR
    %% External inputs
    source["Source Images / Capture"] --> regen3d["regen3d-job\n(scene manifest + layout)"]

    %% Core asset preparation
    regen3d --> interactive["interactive-job\n(articulation + URDF)"]
    regen3d --> simready["simready-job\n(physics + USD prep)"]
    interactive --> simready

    %% Scene assembly
    simready --> usdassembly["usd-assembly-job\n(scene.usda)"]
    usdassembly --> replicator["replicator-job\n(replicator bundle)"]

    %% Variation assets pipeline
    subgraph Variations
        variationgen["variation-gen-job\n(asset prompts)"]
        variationassets["variation-asset-pipeline-job\n(variation assets)"]
        variationgen --> variationassets
    end

    %% Episode generation and downstream consumers
    usdassembly --> episodegen["episode-generation-job\n(episodes + quality reports)"]
    variationassets --> episodegen

    %% Genie Sim integration
    variationassets --> geniesimexport["genie-sim-export-job\n(export bundle)"]
    geniesimexport --> geniesimsubmit["genie-sim-submit-job\n(local gRPC run)"]
    geniesimsubmit --> geniesimimport["genie-sim-import-job\n(validation + import manifest)"]

    %% Training and validation
    replicator --> isaaclab["isaac-lab-job\n(RL training package)"]
    usdassembly --> isaaclab

    %% Optional import results
    geniesimimport --> episodegen
```

Notes:
- Episode generation consumes the USD scene output plus variation assets, consistent with the E2E validation gates in `docs/PRODUCTION_E2E_VALIDATION.md`.
- Genie Sim jobs follow the local health-check and gRPC integration flow in `docs/GENIESIM_INTEGRATION.md`.
- The upstream asset preparation flow mirrors the component breakdown in `docs/PIPELINE_ANALYSIS.md`.

## Experimental / Optional Add-ons (Disabled by Default)

When explicitly enabled, the pipeline can branch to experimental add-ons:
- **DWM preparation** (`dwm-preparation-job`) for egocentric interaction bundles.
- **Dream2Flow preparation** (`dream2flow-preparation-job`) for video/flow bundles.

These steps are intentionally omitted from the core diagram because they are
disabled unless explicitly enabled.
