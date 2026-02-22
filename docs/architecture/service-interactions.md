# Service Interaction Diagram (Mermaid)

This diagram highlights the runtime services and external dependencies referenced in:
- `docs/PIPELINE_ANALYSIS.md`
- `docs/GENIESIM_INTEGRATION.md`
- `docs/PRODUCTION_E2E_VALIDATION.md`

```mermaid
flowchart TB
    %% Job runners
    subgraph Jobs[Pipeline Jobs]
        stage1job["text-scene-adapter-job"]
        interactivejob["interactive-job"]
        simreadyjob["simready-job"]
        usdasmbjob["usd-assembly-job"]
        episodejob["episode-generation-job"]
        variationjob["variation-gen-job"]
        variationassetjob["variation-asset-pipeline-job"]
        geniesimexportjob["genie-sim-export-job"]
        geniesimsubmitjob["genie-sim-submit-job"]
        geniesimimportjob["genie-sim-import-job"]
    end

    %% Internal services
    particulate["particulate-service\n(articulation)" ]

    %% External systems
    gcs["GCS / Artifact Bucket"]
    geniesimgrpc["Genie Sim gRPC Server\n(local-only)"]
    isaacsim["Isaac Sim Runtime"]
    llm["LLM Providers\n(Gemini/OpenAI)"]
    stage1ext["Stage 1 text generation\n(external capture)"]

    %% Data flow
    stage1ext --> stage1job
    stage1job --> gcs
    stage1job --> interactivejob
    interactivejob --> particulate
    interactivejob --> gcs

    simreadyjob --> llm
    simreadyjob --> gcs
    simreadyjob --> usdasmbjob
    usdasmbjob --> gcs

    variationjob --> llm
    variationjob --> variationassetjob
    variationassetjob --> gcs

    episodejob --> isaacsim
    episodejob --> gcs

    geniesimexportjob --> gcs
    geniesimexportjob --> geniesimsubmitjob
    geniesimsubmitjob --> geniesimgrpc
    geniesimgrpc --> geniesimimportjob
    geniesimimportjob --> gcs
```

Notes:
- `interactive-job` depends on `particulate-service` for articulation detection.
- `genie-sim-submit-job` talks to the local Genie Sim gRPC server; the ports and health checks are tracked in `docs/GENIESIM_INTEGRATION.md`.
- GCS (or the configured artifact bucket) is the shared persistence layer across stages, which underpins the marker-driven workflow checks described in `docs/PRODUCTION_E2E_VALIDATION.md`.
