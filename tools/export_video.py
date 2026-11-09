"""
Applies Rust optical enhancement and YOLOv8 detection to input video clips,
saving a composite side-by-side or annotated MP4 file for demonstrations.
"""

import argparse
import os
import sys
import time
import logging

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("tools.export_video")

from core.enhancer import UnderwaterEnhancer
from core.detector import DebrisDetector
from core.pipeline import resolve_video_source


def export_annotated_video(input_path: str, output_path: str, model_path: str, conf: float, side_by_side: bool, device: str):
    """Processes an input video file and writes an annotated MP4 to disk."""
    input_path = resolve_video_source(input_path)
    if not os.path.exists(input_path):
        logger.error(f"Input video '{input_path}' not found!")
        sys.exit(1)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error(f"Failed to open input video: {input_path}")
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    out_w = w * 2 if side_by_side else w
    out_h = h

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    actual_model = model_path if os.path.exists(model_path) else "edge_model.pt"
    detector = DebrisDetector(model_path=actual_model, conf_thresh=conf, device=device)
    enhancer = UnderwaterEnhancer()

    logger.info(f"Rendering '{input_path}' -> '{output_path}' ({total_frames} frames at {fps:.1f} FPS)...")
    start_time = time.time()
    frame_idx = 0

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        enhanced = enhancer.enhance(raw_frame)
        annotated, counts, infer_ms = detector.detect_and_annotate(enhanced)

        if side_by_side:
            raw_labeled = raw_frame.copy()
            cv2.putText(raw_labeled, "RAW UNDERWATER FEED", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(annotated, "ENHANCED + YOLOv8 DETECTIONS", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            composed = np.hstack((raw_labeled, annotated))
        else:
            composed = annotated

        writer.write(composed)
        if frame_idx % 30 == 0 or frame_idx == total_frames:
            elapsed = time.time() - start_time
            proc_fps = frame_idx / max(elapsed, 1e-5)
            logger.info(f"Processed {frame_idx}/{total_frames} frames ({frame_idx/total_frames*100:.1f}%) at {proc_fps:.1f} FPS")

    cap.release()
    writer.release()
    logger.info(f"Video export complete: '{output_path}' ({os.path.getsize(output_path) / (1024*1024):.2f} MB)")


def parse_args():
    """Parses command-line arguments for video rendering."""
    parser = argparse.ArgumentParser(description="Export Underwater Video with AI Detections")
    parser.add_argument("--input", type=str, default="test.mp4", help="Input video path")
    parser.add_argument("--output", type=str, default="underwater_rendered_demo.mp4", help="Output video path")
    parser.add_argument("--model", type=str, default="edge_model.pt", help="YOLO weights path")
    parser.add_argument("--conf", type=float, default=0.20, help="Confidence threshold")
    parser.add_argument("--single", action="store_true", help="Export single enhanced view instead of side-by-side")
    parser.add_argument("--device", type=str, default=None, help="Device ('0', 'cpu', or auto-detect)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_annotated_video(
        input_path=args.input,
        output_path=args.output,
        model_path=args.model,
        conf=args.conf,
        side_by_side=not args.single,
        device=args.device
    )
