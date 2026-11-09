"""
FastAPI Edge Web Streaming Server.
Provides real-time multipart MJPEG video streaming, REST telemetry endpoints,
and dynamic pipeline parameter synchronization.
"""

import time
import logging
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from core.pipeline import EdgePipelineManager

logger = logging.getLogger("server")


app = FastAPI(title="Edge Underwater AI Pipeline", version="1.0.0")

pipeline: EdgePipelineManager = None

templates = Jinja2Templates(directory="templates")
app.mount("/web", StaticFiles(directory="web"), name="web")


class ConfigUpdate(BaseModel):
    """Schema for runtime calibration updates from the web dashboard."""
    apply_enhancement: bool | None = None
    side_by_side: bool | None = None
    split_orientation: str | None = None
    conf_thresh: float | None = None
    red_gain: float | None = None
    clip_limit: float | None = None


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Serves project branding icon for browser tabs."""
    return FileResponse("web/img/logo.png")


@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    """Renders the primary monitoring dashboard."""
    telemetry = pipeline.get_telemetry() if pipeline else {}
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"initial_data": telemetry}
    )


def generate_frames():
    """Yields continuous multipart MJPEG frames to browser clients."""
    while True:
        if pipeline:
            frame_bytes = pipeline.get_jpeg_frame()
            if frame_bytes is not None:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        time.sleep(0.033)


@app.get("/video_feed")
async def video_feed():
    """Multipart video feed endpoint."""
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/stats")
async def get_stats():
    """Returns instantaneous throughput, latency breakdown, and object counts."""
    if not pipeline:
        return JSONResponse({"status": "error", "message": "Pipeline uninitialized"}, status_code=500)
    return JSONResponse(pipeline.get_telemetry())


@app.post("/api/config")
async def update_config(config: ConfigUpdate):
    """Synchronizes live filter parameters and confidence thresholds from the UI."""
    if not pipeline:
        return JSONResponse({"status": "error", "message": "Pipeline uninitialized"}, status_code=500)

    pipeline.update_config(
        apply_enh=config.apply_enhancement,
        side_by_side=config.side_by_side,
        split_orientation=config.split_orientation,
        conf=config.conf_thresh,
        red_gain=config.red_gain,
        clip_limit=config.clip_limit
    )
    return JSONResponse({"status": "success", "updated": pipeline.get_telemetry()})
