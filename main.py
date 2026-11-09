"""
main.py
=======
Production launcher for the Edge Underwater AI Pipeline & Web Streaming Server.
Coordinates video ingestion, Rust optical enhancement, YOLOv8 inference, and FastAPI.
"""

import os
import sys
import argparse
import logging
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("main")

import server
from core.pipeline import EdgePipelineManager, resolve_video_source


def parse_args():
    """Parses command-line arguments for video source, model weights, and network options."""
    parser = argparse.ArgumentParser(description="Edge Underwater AI Pipeline & FastAPI Server")
    parser.add_argument("--video", type=str, default="test.mp4", help="Video file path or camera index (e.g., 0)")
    parser.add_argument("--model", type=str, default="edge_model.pt", help="Path to YOLOv8 weights (default: edge_model.pt)")
    parser.add_argument("--conf", type=float, default=0.20, help="Initial confidence threshold")
    parser.add_argument("--max-res", type=int, default=720, help="Downscaling height ceiling for real-time edge processing")
    parser.add_argument("--device", type=str, default=None, help="Device placement ('0', 'cpu', or auto-detect)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Binding host IP (0.0.0.0 enables LAN access)")
    parser.add_argument("--port", type=int, default=8000, help="Web server port (default: 8000)")
    return parser.parse_args()


def main():
    """Initializes the background processing pipeline and serves the dashboard."""
    args = parse_args()
    args.video = resolve_video_source(args.video)

    logger.info("=" * 65)
    logger.info("EDGE UNDERWATER AI PIPELINE & WEB STREAMING DASHBOARD")
    logger.info("=" * 65)
    logger.info(f"Video Source       : {args.video}")
    logger.info(f"YOLO Weights       : {args.model}")
    logger.info(f"Max Resolution     : {args.max_res}p")
    logger.info(f"Local Dashboard    : http://localhost:{args.port} (or http://127.0.0.1:{args.port})")
    logger.info(f"Network Access     : http://<HOST_IP>:{args.port}")
    logger.info("=" * 65)

    pipeline = EdgePipelineManager(
        video_source=args.video,
        model_path=args.model,
        max_res=args.max_res,
        conf_thresh=args.conf,
        device=args.device
    )

    server.pipeline = pipeline
    pipeline.start()

    try:
        uvicorn.run(
            server.app,
            host=args.host,
            port=args.port,
            log_level="info"
        )
    except KeyboardInterrupt:
        logger.info("Graceful shutdown requested.")
    finally:
        pipeline.stop()
        logger.info("Edge pipeline terminated.")


if __name__ == "__main__":
    main()
