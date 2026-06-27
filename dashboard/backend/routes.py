import io, logging
from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

from dashboard.backend.services import (
    run_interpolation, get_system_status, get_triplet_list
)

logger = logging.getLogger("routes")
router = APIRouter()


class InterpolateRequest(BaseModel):
    triplet_index: int = 0
    stride:        int = 10


@router.get("/api/system/status")
def system_status():
    return get_system_status()


@router.get("/api/triplets")
def triplets(stride: int = Query(10), limit: int = Query(200)):
    return get_triplet_list(stride, limit)


@router.post("/api/interpolate")
def interpolate(req: InterpolateRequest):
    return run_interpolation(
        triplet_index=req.triplet_index,
        stride=req.stride,
    )


@router.post("/api/interpolate/upload")
async def interpolate_upload(
    t0_file: UploadFile = File(...),
    t2_file: UploadFile = File(...),
    gt_file: Optional[UploadFile] = File(None),
):
    t0_b = await t0_file.read()
    t2_b = await t2_file.read()
    gt_b = await gt_file.read() if gt_file else None
    return run_interpolation(t0_bytes=t0_b, t2_bytes=t2_b, gt_bytes=gt_b)


@router.get("/api/metrics/latest")
def metrics_latest():
    result = run_interpolation()
    return result.get("metrics", {})


@router.get("/api/visuals/heatmap")
def heatmap_visual():
    result = run_interpolation()
    b64 = result.get("images", {}).get("diff_heatmap", "")
    import base64
    raw = base64.b64decode(b64)
    return StreamingResponse(io.BytesIO(raw), media_type="image/png")


@router.get("/api/visuals/vectors")
def vector_visual():
    result = run_interpolation()
    b64 = result.get("images", {}).get("optical_flow", "")
    import base64
    raw = base64.b64decode(b64)
    return StreamingResponse(io.BytesIO(raw), media_type="image/png")