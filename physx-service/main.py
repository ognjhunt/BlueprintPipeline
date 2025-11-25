# physx-service/main.py
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()


class PhysxRequest(BaseModel):
    image_base64: str


class PhysxResponse(BaseModel):
    mesh_base64: str | None = None
    urdf_base64: str | None = None


# For now, just return placeholder assets generated on the fly.
# This is only to test the end‑to‑end pipeline.
def _placeholder_glb_bytes() -> bytes:
    # Not a real GLB – just something to see bytes move through.
    # Later we'll replace this with PhysX-Anything output.
    return b"Placeholder GLB from physx-service"


def _placeholder_urdf_bytes() -> bytes:
    return b"""
<robot name="placeholder">
  <link name="base">
    <visual>
      <geometry>
        <mesh filename="part.glb" />
      </geometry>
    </visual>
  </link>
  <joint name="fixed" type="fixed">
    <parent link="base"/>
    <child link="base"/>
  </joint>
</robot>
""".strip()


@app.post("/physx", response_model=PhysxResponse)
async def run_physx(req: PhysxRequest) -> PhysxResponse:
    # Minimal validation
    if not req.image_base64:
        raise HTTPException(status_code=400, detail="image_base64 is required")

    # In stub mode we ignore the image, just confirm we can decode it.
    try:
        base64.b64decode(req.image_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="image_base64 is not valid base64")

    mesh_bytes = _placeholder_glb_bytes()
    urdf_bytes = _placeholder_urdf_bytes()

    return PhysxResponse(
        mesh_base64=base64.b64encode(mesh_bytes).encode("ascii").decode("ascii"),
        urdf_base64=base64.b64encode(urdf_bytes).encode("ascii").decode("ascii"),
    )
