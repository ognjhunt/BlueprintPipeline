#!/usr/bin/env python3
"""
RunPod SAGE baseline runner.

Creates two RunPod pods:
- Pod A: TRELLIS HTTP server (port 8080)
- Pod B: Isaac Sim (NGC container) + SAGE client/server scripts

Then optionally runs:
- Text-only room generation
- Image-conditioned room generation

This script is intentionally stdlib-only (curl/requests not required).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNPOD_CREDS_FILE = REPO_ROOT / "configs" / "runpod_credentials.env"


def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"[runpod-sage {ts}] {msg}", file=sys.stderr, flush=True)


def _read_env_file(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    out: Dict[str, str] = {}
    for raw in path.read_text("utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            continue
        # Strip simple wrapping quotes.
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]
        out[k] = v
    return out


def _ensure_env_var(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        raise RuntimeError(f"Missing required env var: {name}")
    return val


def _run(cmd: list[str], *, check: bool = True, capture: bool = False, text: bool = True) -> subprocess.CompletedProcess:
    _log(f"exec: {shlex.join(cmd)}")
    return subprocess.run(
        cmd,
        check=check,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        text=text,
    )


def _graphql(api_url: str, query: str) -> dict:
    # RunPod's GraphQL endpoint is behind Cloudflare, and in some environments
    # Python's default TLS fingerprint can trigger 403 / error code 1010.
    # Prefer invoking curl (which tends to be whitelisted) and fall back to urllib.
    user_agent = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    if shutil.which("curl"):
        payload = json.dumps({"query": query})
        # Avoid logging the RunPod API key.
        safe_url = re.sub(r"(api_key=)[^&]+", r"\\1***", api_url)
        _log(f"exec: curl POST {safe_url}")
        p = subprocess.run(
            [
                "curl",
                "-sS",
                "--fail-with-body",
                "--request",
                "POST",
                "--header",
                "content-type: application/json",
                "--header",
                "accept: application/json",
                "--user-agent",
                user_agent,
                "--data",
                payload,
                api_url,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if p.returncode != 0:
            out = (p.stdout or "").strip()
            raise RuntimeError(f"RunPod API curl failed (rc={p.returncode}): {out[:500]}")
        raw = (p.stdout or "").encode("utf-8")
    else:
        payload = json.dumps({"query": query}).encode("utf-8")
        req = urllib.request.Request(
            api_url,
            data=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json", "User-Agent": user_agent},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
            raise RuntimeError(f"RunPod API HTTPError: {e.code} {e.reason}: {body[:500]}") from e
        except Exception as e:
            raise RuntimeError(f"RunPod API request failed: {e}") from e

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"RunPod API returned invalid JSON: {raw[:500]!r}") from e

    if isinstance(data, dict) and data.get("errors"):
        raise RuntimeError(f"RunPod API returned errors: {data['errors']}")
    return data.get("data") or {}


@dataclasses.dataclass(frozen=True)
class PodEndpoint:
    ip: str
    port: int


@dataclasses.dataclass
class PodInfo:
    pod_id: str
    ssh: Optional[PodEndpoint] = None
    trellis_http: Optional[PodEndpoint] = None


def _escape_graphql_string(value: str) -> str:
    # GraphQL string literal escaping.
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def _launch_pod(
    *,
    api_url: str,
    cloud_type: str,
    gpu_type: str,
    image: str,
    name: str,
    volume_gb: int,
    disk_gb: int,
    min_vcpu: int,
    min_mem_gb: int,
    ports: str,
    env: Dict[str, str],
) -> PodInfo:
    env_items = ", ".join(
        f'{{ key: "{_escape_graphql_string(k)}", value: "{_escape_graphql_string(v)}" }}' for k, v in env.items()
    )
    query = (
        "mutation { podFindAndDeployOnDemand(input: { "
        f"cloudType: {cloud_type}, gpuCount: 1, volumeInGb: {volume_gb}, "
        f"containerDiskInGb: {disk_gb}, minVcpuCount: {min_vcpu}, minMemoryInGb: {min_mem_gb}, "
        f'gpuTypeId: "{_escape_graphql_string(gpu_type)}", '
        f'name: "{_escape_graphql_string(name)}", '
        f'imageName: "{_escape_graphql_string(image)}", '
        f'dockerArgs: "", ports: "{_escape_graphql_string(ports)}", '
        'volumeMountPath: "/workspace", '
        f"env: [{env_items}] "
        "}) { id desiredStatus costPerHr } }"
    )
    data = _graphql(api_url, query)
    launch = data.get("podFindAndDeployOnDemand") or {}
    pod_id = str(launch.get("id") or "").strip()
    if not pod_id:
        raise RuntimeError("RunPod returned no pod id")
    return PodInfo(pod_id=pod_id)


def _poll_pod_runtime(
    *,
    api_url: str,
    pod_id: str,
    timeout_s: int,
    want_trellis: bool,
) -> PodInfo:
    info = PodInfo(pod_id=pod_id)
    deadline = time.time() + max(120, timeout_s)
    while time.time() < deadline:
        query = (
            'query { pod(input: {podId: "' + _escape_graphql_string(pod_id) + '"}) { '
            "desiredStatus runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } } } }"
        )
        data = _graphql(api_url, query)
        pod = data.get("pod") or {}
        runtime = pod.get("runtime") or {}
        ports = runtime.get("ports") or []
        for port in ports:
            if not (port.get("isIpPublic") and port.get("ip") and port.get("publicPort")):
                continue
            priv = port.get("privatePort")
            pub = int(port.get("publicPort"))
            ip = str(port.get("ip"))
            if priv == 22:
                info.ssh = PodEndpoint(ip=ip, port=pub)
            if want_trellis and priv == 8080:
                info.trellis_http = PodEndpoint(ip=ip, port=pub)
        if info.ssh and (not want_trellis or info.trellis_http):
            return info
        time.sleep(15)
    raise RuntimeError(f"Timed out waiting for pod runtime ports (pod={pod_id})")


def _ssh_base_args(*, key_path: Path, ssh: PodEndpoint) -> list[str]:
    return [
        "ssh",
        "-i",
        str(key_path),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "ConnectTimeout=30",
        "-p",
        str(ssh.port),
        f"root@{ssh.ip}",
    ]


def _scp_base_args(*, key_path: Path, ssh: PodEndpoint) -> list[str]:
    return [
        "scp",
        "-i",
        str(key_path),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-P",
        str(ssh.port),
    ]


def _wait_for_ssh(*, key_path: Path, ssh: PodEndpoint, timeout_s: int) -> None:
    base = _ssh_base_args(key_path=key_path, ssh=ssh)
    deadline = time.time() + max(60, timeout_s)
    while True:
        probe = subprocess.run(base + ["echo ready"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if probe.returncode == 0:
            return
        if time.time() >= deadline:
            raise RuntimeError(f"Timed out waiting for SSH readiness at {ssh.ip}:{ssh.port}")
        time.sleep(10)


def _scp_to_pod(*, key_path: Path, ssh: PodEndpoint, local_path: Path, remote_path: str) -> None:
    base = _scp_base_args(key_path=key_path, ssh=ssh)
    _run(base + [str(local_path), f"root@{ssh.ip}:{remote_path}"])


def _ssh_run(*, key_path: Path, ssh: PodEndpoint, remote_cmd: str, capture: bool = False) -> subprocess.CompletedProcess:
    base = _ssh_base_args(key_path=key_path, ssh=ssh)
    return _run(base + [remote_cmd], capture=capture)


def _terminate_pod(*, api_url: str, pod_id: str) -> None:
    try:
        query = 'mutation { podTerminate(input: {podId: "' + _escape_graphql_string(pod_id) + '"}) }'
        _graphql(api_url, query)
    except Exception as e:
        _log(f"WARNING: failed to terminate pod {pod_id}: {e}")


def _extract_layout_id(stdout: str) -> Optional[str]:
    m = re.search(r"Layout ID:\\s*([A-Za-z0-9_\\-]+)", stdout)
    return m.group(1) if m else None


def _default_room_desc_for_archetype(archetype: str) -> str:
    normalized = archetype.strip().lower()
    presets = {
        "kitchen": "A medium-sized commercial kitchen prep line with stainless counters, drawers, and mugs.",
        "grocery": "A grocery / retail aisle with stocked shelves, a refrigerated case, baskets, and price labels.",
        "warehouse": "A warehouse tote-picking zone with racking, carts, boxes, and a packing table.",
        "loading_dock": "A loading dock bay with pallets, a roll-up door, staging area, and shipping supplies.",
        "lab": "A laboratory bench area with workbenches, cabinets, sample racks, and lab containers.",
        "hospital": "A hospital patient room with a bed, medical cart, supply cabinets, and bright clinical lighting.",
        "office": "An office workspace with desks, drawers, shelves, and small objects on surfaces.",
        "utility_room": "A utility / mechanical room with electrical panels, tool shelves, cabinets, and maintenance clutter.",
        "home_laundry": "A home laundry room with washer, dryer, hamper, cabinets, and clothing items.",
    }
    if normalized in presets:
        return presets[normalized]
    humanized = normalized.replace("_", " ")
    return f"A medium-sized {humanized}."


def main() -> int:
    parser = argparse.ArgumentParser(description="RunPod SAGE baseline (TRELLIS + Isaac Sim + SAGE).")
    parser.add_argument("--secrets-env-file", type=str, default="", help="Optional env file with HF_TOKEN/OPENAI_API_KEY/NGC_API_KEY.")
    parser.add_argument("--keep-pods", action="store_true", help="Do not terminate pods at the end.")
    parser.add_argument("--skip-runs", action="store_true", help="Only bootstrap pods, do not run generation.")
    parser.add_argument(
        "--archetype",
        type=str,
        default="kitchen",
        choices=[
            "kitchen",
            "grocery",
            "warehouse",
            "loading_dock",
            "lab",
            "hospital",
            "office",
            "utility_room",
            "home_laundry",
        ],
        help="Optional archetype preset that selects a default room description (used when --room-desc is empty).",
    )
    parser.add_argument(
        "--room-desc",
        type=str,
        default="",
        help="Text-only baseline room description (overrides --archetype when set).",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="",
        help="Optional local image path for image-conditioned run (will be uploaded to Pod B).",
    )
    parser.add_argument(
        "--image-room-desc",
        type=str,
        default="Reconstruct a semantically coherent version of this room.",
        help="Room description to pair with --image.",
    )
    parser.add_argument("--openai-model", type=str, default="gpt-4o")
    parser.add_argument("--openai-base-url", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--slurm-job-id", type=str, default="runpod_sage_001")
    parser.add_argument("--runpod-cloud-type", type=str, default="COMMUNITY", choices=["COMMUNITY", "SECURE"])
    parser.add_argument("--runpod-gpu-type", type=str, default="NVIDIA L40S")
    parser.add_argument("--runpod-image", type=str, default="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04")
    parser.add_argument("--pod-a-volume-gb", type=int, default=120)
    parser.add_argument("--pod-a-disk-gb", type=int, default=40)
    parser.add_argument("--pod-b-volume-gb", type=int, default=250)
    parser.add_argument("--pod-b-disk-gb", type=int, default=60)
    parser.add_argument("--runpod-min-vcpu", type=int, default=8)
    parser.add_argument("--runpod-min-mem-gb", type=int, default=30)
    parser.add_argument("--boot-timeout-s", type=int, default=900)
    parser.add_argument("--ssh-ready-timeout-s", type=int, default=240)
    args = parser.parse_args()
    room_desc = (args.room_desc or "").strip() or _default_room_desc_for_archetype(args.archetype)

    # Load optional secrets env file.
    if args.secrets_env_file:
        secrets = _read_env_file(Path(args.secrets_env_file))
        for k, v in secrets.items():
            os.environ.setdefault(k, v)

    # Load RunPod API key.
    if not os.environ.get("RUNPOD_API_KEY"):
        creds = _read_env_file(RUNPOD_CREDS_FILE)
        if "RUNPOD_API_KEY" in creds:
            os.environ["RUNPOD_API_KEY"] = creds["RUNPOD_API_KEY"]

    runpod_api_key = _ensure_env_var("RUNPOD_API_KEY")
    hf_token = _ensure_env_var("HF_TOKEN")
    openai_api_key = _ensure_env_var("OPENAI_API_KEY")
    ngc_api_key = _ensure_env_var("NGC_API_KEY")

    api_url = f"https://api.runpod.io/graphql?api_key={runpod_api_key}"

    bootstrap_trellis = REPO_ROOT / "scripts" / "runpod_sage" / "bootstrap_trellis_pod.sh"
    bootstrap_isaac = REPO_ROOT / "scripts" / "runpod_sage" / "bootstrap_sage_isaacsim_pod.sh"
    run_room_desc = REPO_ROOT / "scripts" / "runpod_sage" / "run_room_desc.sh"

    for p in [bootstrap_trellis, bootstrap_isaac, run_room_desc]:
        if not p.exists():
            raise RuntimeError(f"Missing local script: {p}")

    pods: Dict[str, PodInfo] = {}
    out_state: Dict[str, Any] = {"started_at": time.time(), "pods": {}}

    # Ephemeral SSH key for both pods.
    temp_dir = Path(tempfile.mkdtemp(prefix="runpod-sage-"))
    key_path = temp_dir / "id_ed25519"
    _run(["ssh-keygen", "-t", "ed25519", "-N", "", "-f", str(key_path)])
    public_key = (key_path.with_suffix(".pub")).read_text("utf-8").strip()

    def save_state() -> None:
        out = REPO_ROOT / "analysis_outputs" / f"runpod_sage_baseline_{int(time.time())}.json"
        # keep overwriting the same file name for convenience
        out.write_text(json.dumps(out_state, indent=2) + "\n", encoding="utf-8")
        _log(f"state: {out}")

    try:
        # Pod A
        pod_a_name = f"sage-trellis-{int(time.time())}"
        _log(f"Launching Pod A (TRELLIS): {pod_a_name}")
        pod_a = _launch_pod(
            api_url=api_url,
            cloud_type=args.runpod_cloud_type,
            gpu_type=args.runpod_gpu_type,
            image=args.runpod_image,
            name=pod_a_name,
            volume_gb=args.pod_a_volume_gb,
            disk_gb=args.pod_a_disk_gb,
            min_vcpu=args.runpod_min_vcpu,
            min_mem_gb=args.runpod_min_mem_gb,
            ports="22/tcp,8080/tcp",
            env={"PUBLIC_KEY": public_key},
        )
        pods["pod_a"] = pod_a

        pod_a = _poll_pod_runtime(
            api_url=api_url,
            pod_id=pod_a.pod_id,
            timeout_s=args.boot_timeout_s,
            want_trellis=True,
        )
        if not pod_a.ssh or not pod_a.trellis_http:
            raise RuntimeError("Pod A missing ssh or trellis endpoint")
        _wait_for_ssh(key_path=key_path, ssh=pod_a.ssh, timeout_s=args.ssh_ready_timeout_s)

        out_state["pods"]["pod_a"] = dataclasses.asdict(pod_a)
        save_state()

        # Upload secrets/env + bootstrap
        secrets_a = temp_dir / "pod_a_secrets.env"
        secrets_a.write_text(f"export HF_TOKEN={shlex.quote(hf_token)}\n", encoding="utf-8")
        os.chmod(secrets_a, 0o600)

        _scp_to_pod(key_path=key_path, ssh=pod_a.ssh, local_path=secrets_a, remote_path="/workspace/.sage_runpod_secrets.env")
        _scp_to_pod(key_path=key_path, ssh=pod_a.ssh, local_path=bootstrap_trellis, remote_path="/workspace/bootstrap_trellis_pod.sh")
        _ssh_run(
            key_path=key_path,
            ssh=pod_a.ssh,
            remote_cmd="bash -lc 'chmod 600 /workspace/.sage_runpod_secrets.env; chmod +x /workspace/bootstrap_trellis_pod.sh; /workspace/bootstrap_trellis_pod.sh'",
        )

        trellis_url = f"http://{pod_a.trellis_http.ip}:{pod_a.trellis_http.port}"
        _log(f"TRELLIS public URL: {trellis_url}")

        # Pod B
        pod_b_name = f"sage-isaac-{int(time.time())}"
        _log(f"Launching Pod B (Isaac Sim + SAGE): {pod_b_name}")
        pod_b = _launch_pod(
            api_url=api_url,
            cloud_type=args.runpod_cloud_type,
            gpu_type=args.runpod_gpu_type,
            image=args.runpod_image,
            name=pod_b_name,
            volume_gb=args.pod_b_volume_gb,
            disk_gb=args.pod_b_disk_gb,
            min_vcpu=args.runpod_min_vcpu,
            min_mem_gb=args.runpod_min_mem_gb,
            ports="22/tcp",
            env={"PUBLIC_KEY": public_key},
        )
        pods["pod_b"] = pod_b

        pod_b = _poll_pod_runtime(api_url=api_url, pod_id=pod_b.pod_id, timeout_s=args.boot_timeout_s, want_trellis=False)
        if not pod_b.ssh:
            raise RuntimeError("Pod B missing ssh endpoint")
        _wait_for_ssh(key_path=key_path, ssh=pod_b.ssh, timeout_s=args.ssh_ready_timeout_s)
        out_state["pods"]["pod_b"] = dataclasses.asdict(pod_b)
        save_state()

        secrets_b = temp_dir / "pod_b_secrets.env"
        secrets_b.write_text(
            "\n".join(
                [
                    f"export OPENAI_API_KEY={shlex.quote(openai_api_key)}",
                    f"export NGC_API_KEY={shlex.quote(ngc_api_key)}",
                    f"export SLURM_JOB_ID={shlex.quote(args.slurm_job_id)}",
                    f"export TRELLIS_SERVER_URL={shlex.quote(trellis_url)}",
                    f"export OPENAI_BASE_URL={shlex.quote(args.openai_base_url)}",
                    f"export OPENAI_MODEL={shlex.quote(args.openai_model)}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        os.chmod(secrets_b, 0o600)

        _scp_to_pod(key_path=key_path, ssh=pod_b.ssh, local_path=secrets_b, remote_path="/workspace/.sage_runpod_secrets.env")
        _scp_to_pod(key_path=key_path, ssh=pod_b.ssh, local_path=bootstrap_isaac, remote_path="/workspace/bootstrap_sage_isaacsim_pod.sh")
        _scp_to_pod(key_path=key_path, ssh=pod_b.ssh, local_path=run_room_desc, remote_path="/workspace/run_room_desc.sh")

        _ssh_run(
            key_path=key_path,
            ssh=pod_b.ssh,
            remote_cmd="bash -lc 'chmod 600 /workspace/.sage_runpod_secrets.env; chmod +x /workspace/bootstrap_sage_isaacsim_pod.sh; chmod +x /workspace/run_room_desc.sh; /workspace/bootstrap_sage_isaacsim_pod.sh'",
        )

        if args.skip_runs:
            _log("skip-runs enabled; bootstrap complete.")
            return 0

        # Run 1: text-only
        _log("Running baseline 1 (text-only)")
        run1 = _ssh_run(
            key_path=key_path,
            ssh=pod_b.ssh,
            remote_cmd="bash -lc " + shlex.quote(f'/workspace/run_room_desc.sh {shlex.quote(room_desc)}'),
            capture=True,
        )
        run1_out = run1.stdout or ""
        layout1 = _extract_layout_id(run1_out)
        if layout1:
            _log(f"baseline1 layout_id={layout1}")
        else:
            _log("WARNING: could not parse layout id from baseline 1 output")

        out_state["baseline1"] = {"archetype": args.archetype, "room_desc": room_desc, "layout_id": layout1}
        save_state()

        # Optional: Run 2 with image
        if args.image:
            img_path = Path(args.image).expanduser().resolve()
            if not img_path.exists():
                raise RuntimeError(f"--image not found: {img_path}")
            _log(f"Uploading image to Pod B: {img_path}")
            _ssh_run(key_path=key_path, ssh=pod_b.ssh, remote_cmd="bash -lc 'mkdir -p /workspace/inputs'")
            _scp_to_pod(key_path=key_path, ssh=pod_b.ssh, local_path=img_path, remote_path="/workspace/inputs/ref.png")
            _log("Running baseline 2 (image-conditioned)")
            run2 = _ssh_run(
                key_path=key_path,
                ssh=pod_b.ssh,
                remote_cmd="bash -lc "
                + shlex.quote(
                    "INPUT_IMAGE_PATH=/workspace/inputs/ref.png "
                    + f"/workspace/run_room_desc.sh {shlex.quote(args.image_room_desc)}"
                ),
                capture=True,
            )
            run2_out = run2.stdout or ""
            layout2 = _extract_layout_id(run2_out)
            if layout2:
                _log(f"baseline2 layout_id={layout2}")
            else:
                _log("WARNING: could not parse layout id from baseline 2 output")
            out_state["baseline2"] = {"room_desc": args.image_room_desc, "layout_id": layout2, "image": str(img_path)}
            save_state()

        # Critique (remote, stdlib-only)
        for label, layout_id in [
            ("baseline1", (out_state.get("baseline1") or {}).get("layout_id")),
            ("baseline2", (out_state.get("baseline2") or {}).get("layout_id")),
        ]:
            if not layout_id:
                continue
            _log(f"Critiquing {label} layout_id={layout_id}")
            critique_cmd = r"""
python3 - <<'PY'
import json, os, sys
layout_id = os.environ.get("LAYOUT_ID")
root = f"/workspace/SAGE/server/results/{layout_id}"
layout_path = f"{root}/{layout_id}.json"
report = {
  "layout_id": layout_id,
  "layout_json_exists": os.path.exists(layout_path),
  "rooms": 0,
  "objects": 0,
  "empty_support_surfaces": 0,
  "assets_missing_glb": 0,
  "assets_missing_texture": 0,
}
if not os.path.exists(layout_path):
  print(json.dumps(report))
  sys.exit(0)

layout = json.load(open(layout_path, "r"))
rooms = layout.get("rooms") or []
report["rooms"] = len(rooms)
objs = []
for r in rooms:
  for o in (r.get("objects") or []):
    if isinstance(o, dict) and o.get("id"):
      objs.append(o)
report["objects"] = len(objs)

obj_by_id = {o["id"]: o for o in objs}
children = {}
for o in objs:
  pid = o.get("place_id")
  if pid and pid not in ("floor", "wall"):
    children.setdefault(pid, 0)
    children[pid] += 1

surface_tokens = ("table", "desk", "counter", "countertop", "shelf", "nightstand", "dresser", "cabinet", "island")
empty = 0
for o in objs:
  if o.get("place_id") != "floor":
    continue
  t = str(o.get("type") or "").lower()
  if any(tok in t for tok in surface_tokens):
    if children.get(o["id"], 0) == 0:
      empty += 1
report["empty_support_surfaces"] = empty

missing_glb = 0
missing_tex = 0
for o in objs:
  src = str(o.get("source") or "").strip()
  sid = str(o.get("source_id") or "").strip()
  if not src or not sid:
    continue
  glb = f"{root}/{src}/{sid}.glb"
  tex = f"{root}/{src}/{sid}_texture.png"
  if not os.path.exists(glb):
    missing_glb += 1
  if not os.path.exists(tex):
    missing_tex += 1
report["assets_missing_glb"] = missing_glb
report["assets_missing_texture"] = missing_tex

print(json.dumps(report))
PY
"""
            critique = _ssh_run(
                key_path=key_path,
                ssh=pod_b.ssh,
                remote_cmd="bash -lc "
                + shlex.quote(f"LAYOUT_ID={shlex.quote(str(layout_id))} {critique_cmd}"),
                capture=True,
            )
            critique_line = (critique.stdout or "").strip().splitlines()[-1] if critique.stdout else "{}"
            out_state.setdefault("critiques", {})[label] = json.loads(critique_line)
            save_state()

        _log("All requested runs complete.")
        _log("Reminder: pods keep costing until terminated.")
        return 0
    finally:
        if not args.keep_pods:
            for pod_label, pod in pods.items():
                if pod.pod_id:
                    _log(f"Terminating {pod_label} pod: {pod.pod_id}")
                    _terminate_pod(api_url=api_url, pod_id=pod.pod_id)


if __name__ == "__main__":
    raise SystemExit(main())
