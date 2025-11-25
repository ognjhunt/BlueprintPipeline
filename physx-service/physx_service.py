import base64
import io
import os

from flask import Flask, jsonify, request
import trimesh

app = Flask(__name__)


def make_dummy_assets():
    """
    Build a simple cube mesh and a URDF that references it as 'part.glb'.
    The geometry is tiny but 100% valid, so Isaac / USD tools can load it.

    Later you can replace this with a real PhysX-Anything pipeline that
    generates meshes based on the input image.
    """
    # 1) Create a cube mesh with extents 0.4 m in each dimension.
    #    trimesh supports exporting to GLB directly. 
    mesh = trimesh.creation.box(extents=(0.4, 0.4, 0.4))

    # 2) Export the mesh to GLB in memory.
    buf = io.BytesIO()
    mesh.export(buf, file_type="glb")
    glb_bytes = buf.getvalue()
    mesh_b64 = base64.b64encode(glb_bytes).decode("ascii")

    # 3) Minimal URDF with one link that uses this mesh for visual + collision.
    urdf_text = """
<robot name="physx_dummy">
  <link name="base">
    <visual>
      <geometry>
        <mesh filename="part.glb"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <mesh filename="part.glb"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.001" iyy="0.001" izz="0.001"
               ixy="0.0" ixz="0.0" iyz="0.0"/>
    </inertial>
  </link>
</robot>
""".strip()

    urdf_b64 = base64.b64encode(urdf_text.encode("utf-8")).decode("ascii")

    return mesh_b64, urdf_b64


@app.route("/", methods=["GET"])
def healthcheck():
    """
    Simple health endpoint so you can hit the service in a browser / curl.
    """
    return "ok", 200


@app.route("/", methods=["POST"])
def generate_assets():
    """
    Main endpoint used by interactive-job.

    Expects JSON:
        { "image_base64": "<base64 png/jpg bytes>" }

    Returns JSON with mesh / URDF in base64 so the job can write them to disk.

    This matches what run_interactive_assets.py expects:
      - it POSTs to PHYSX_ENDPOINT
      - with a JSON body containing "image_base64"
      - and reads "mesh_base64" / "urdf_base64" from the response.
    """
    data = request.get_json(force=True, silent=True) or {}
    img_b64 = data.get("image_base64")

    if not img_b64:
        # interactive-job always sends image_base64; this is just defensive.
        return jsonify({"error": "image_base64 is required"}), 400

    # For now we ignore the image; later you can add real processing here.
    mesh_b64, urdf_b64 = make_dummy_assets()

    return jsonify(
        {
            "mesh_base64": mesh_b64,
            "urdf_base64": urdf_b64,
            "placeholder": False,
            "generator": "dummy-physx-service",
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    # Cloud Run requires 0.0.0.0
    app.run(host="0.0.0.0", port=port)
