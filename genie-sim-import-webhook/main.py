import json
import os

from flask import Flask, jsonify, request
from google.cloud.workflows import executions_v1

app = Flask(__name__)


def _get_project_id() -> str:
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("PROJECT_ID")
    if not project_id:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT must be set")
    return project_id


@app.get("/healthz")
def health_check():
    return jsonify({"status": "ok"})


@app.post("/webhooks/geniesim/job-complete")
def handle_job_complete():
    payload = request.get_json(silent=True) or {}
    job_id = payload.get("job_id")
    if not job_id:
        return jsonify({"error": "job_id is required"}), 400

    payload.setdefault("status", "completed")

    workflow_name = os.getenv("WORKFLOW_NAME", "genie-sim-import-pipeline")
    region = os.getenv("WORKFLOW_REGION", "us-central1")

    client = executions_v1.ExecutionsClient()
    parent = f"projects/{_get_project_id()}/locations/{region}/workflows/{workflow_name}"
    execution = executions_v1.Execution(argument=json.dumps(payload))
    response = client.create_execution(request={"parent": parent, "execution": execution})

    return jsonify({"status": "triggered", "execution": response.name}), 202


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
