"""
FastAPI Edge Web Streaming Server.
Provides real-time multipart MJPEG video streaming, REST telemetry endpoints,
dynamic video source switching, video file upload, and model calibration synchronization.
"""

import os
import shutil
import time
import logging
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from core.pipeline import EdgePipelineManager

logger = logging.getLogger("server")

app = FastAPI(title="Edge Underwater AI Pipeline", version="2.2.0")

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
    model_choice: str | None = None


class SourceChangeRequest(BaseModel):
    """Schema for video source switching."""
    source: str | int


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Serves project branding icon for browser tabs."""
    return FileResponse("web/img/logo.png")


@app.get("/", response_class=HTMLResponse)
def index_page(request: Request):
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
def video_feed():
    """Multipart video feed endpoint."""
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/stats")
def get_stats():
    """Returns instantaneous throughput, latency breakdown, and object counts."""
    if not pipeline:
        return JSONResponse({"status": "error", "message": "Pipeline uninitialized"}, status_code=500)
    return JSONResponse(pipeline.get_telemetry())


@app.post("/api/config")
def update_config(config: ConfigUpdate):
    """Synchronizes live filter parameters and confidence thresholds from the UI."""
    if not pipeline:
        return JSONResponse({"status": "error", "message": "Pipeline uninitialized"}, status_code=500)

    pipeline.update_config(
        apply_enh=config.apply_enhancement,
        side_by_side=config.side_by_side,
        split_orientation=config.split_orientation,
        conf=config.conf_thresh,
        red_gain=config.red_gain,
        clip_limit=config.clip_limit,
        model_choice=config.model_choice
    )
    return JSONResponse({"status": "success", "updated": pipeline.get_telemetry()})


@app.get("/api/videos")
def list_videos():
    """Returns a list of local sample video files and current video source."""
    video_dir = "videos"
    if not os.path.exists(video_dir):
        return JSONResponse({"videos": [], "current": ""})

    valid_exts = (".mp4", ".avi", ".mov", ".mkv")
    files = sorted([f for f in os.listdir(video_dir) if f.lower().endswith(valid_exts)])
    current_source = str(pipeline.video_source) if pipeline else ""
    return JSONResponse({
        "videos": files,
        "current": os.path.basename(current_source)
    })


@app.post("/api/change_source")
def change_source(req: SourceChangeRequest):
    """Switches the active video source dynamically."""
    if not pipeline:
        return JSONResponse({"status": "error", "message": "Pipeline uninitialized"}, status_code=500)

    pipeline.change_source(req.source)
    return JSONResponse({
        "status": "success",
        "source": str(req.source),
        "message": f"Switching video source to: {req.source}"
    })


@app.post("/api/upload_video")
def upload_video(file: UploadFile = File(...)):
    """Receives an uploaded video file, saves it, and switches video feed to it."""
    if not pipeline:
        return JSONResponse({"status": "error", "message": "Pipeline uninitialized"}, status_code=500)

    filename = os.path.basename(file.filename)
    if not filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        return JSONResponse({"status": "error", "message": "Invalid video format. Supported: .mp4, .avi, .mov, .mkv"}, status_code=400)

    save_dir = os.path.abspath("videos")
    os.makedirs(save_dir, exist_ok=True)
    dest_path = os.path.join(save_dir, filename)

    try:
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            buffer.flush()

        logger.info(f"Uploaded custom video saved to: {dest_path}")
        pipeline.change_source(dest_path)

        return JSONResponse({
            "status": "success",
            "filename": filename,
            "path": dest_path,
            "message": f"Uploaded {filename} successfully. Switching feed now."
        })
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return JSONResponse({"status": "error", "message": f"Upload failed: {str(e)}"}, status_code=500)
