"""
core/pipeline.py
================
Unified Video Processing Pipeline.
Integrates continuous frame ingestion, Rust optical enhancement, and YOLOv8n inference.
Maintains a thread-safe frame buffer for real-time FastAPI MJPEG streaming.
"""

import os
import time
import logging
import threading
import cv2
import numpy as np

logger = logging.getLogger("core.pipeline")

from core.enhancer import UnderwaterEnhancer
from core.detector import DebrisDetector


def resolve_video_source(source: str | int) -> str | int:
    """Resolves video source by checking root, videos/ folder, and extensions."""
    if str(source).isdigit():
        return int(source)

    v_str = str(source)
    candidates = [
        v_str,
        v_str + ".mp4",
        os.path.join("videos", v_str),
        os.path.join("videos", v_str + ".mp4"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return v_str


class EdgePipelineManager:
    """Coordinates camera capture, optical restoration, object detection, and streaming."""

    def __init__(
        self,
        video_source: str = "test.mp4",
        model_path: str = "edge_model.pt",
        max_res: int = 720,
        conf_thresh: float = 0.20,
        device: str = None
    ):
        self.video_source = resolve_video_source(video_source)
        self.max_res = max_res

        self.enhancer = UnderwaterEnhancer()
        self.detector = DebrisDetector(model_path=model_path, conf_thresh=conf_thresh, device=device)

        self.apply_enhancement = True
        self.side_by_side = True
        self.split_orientation = "horizontal"
        self.running = False

        self.fps = 0.0
        self.t_enh = 0.0
        self.t_yolo = 0.0
        self.counts = {"Trash": 0, "Bio": 0, "Submersible_Gear": 0}

        self.lock = threading.Lock()
        self.latest_jpeg = None
        self.thread = None

    def start(self):
        """Starts the background worker thread."""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.thread.start()
        logger.info("Edge Video Pipeline worker started.")

    def stop(self):
        """Stops the processing loop cleanly."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        logger.info("Edge Video Pipeline worker stopped.")

    def update_config(self, apply_enh=None, side_by_side=None, conf=None, red_gain=None, clip_limit=None, split_orientation=None):
        """Applies dynamic runtime parameter updates from the dashboard."""
        if apply_enh is not None:
            self.apply_enhancement = bool(apply_enh)
        if side_by_side is not None:
            self.side_by_side = bool(side_by_side)
        if split_orientation is not None:
            self.split_orientation = str(split_orientation)
        if conf is not None:
            self.detector.update_confidence(conf)
        self.enhancer.update_parameters(clip_limit=clip_limit, red_gain=red_gain)

    def get_telemetry(self) -> dict:
        """Returns instantaneous pipeline performance and detection metrics."""
        return {
            "fps": round(self.fps, 1),
            "enhancement_ms": round(self.t_enh, 1),
            "yolo_ms": round(self.t_yolo, 1),
            "counts": self.counts,
            "apply_enhancement": self.apply_enhancement,
            "side_by_side": self.side_by_side,
            "split_orientation": self.split_orientation,
            "conf_thresh": self.detector.conf_thresh,
            "red_gain": self.enhancer.red_gain,
            "clip_limit": self.enhancer.clip_limit,
            "device": self.detector.device.upper(),
            "rust_accelerated": self.enhancer.rust_bridge.loaded
        }

    def get_jpeg_frame(self) -> bytes:
        """Thread-safe retrieval of the latest encoded JPEG frame."""
        with self.lock:
            return self.latest_jpeg

    def _processing_loop(self):
        """Core frame ingestion, enhancement, inference, and encoding loop."""
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            logger.error(f"Could not open video source: '{self.video_source}'")
            self.running = False
            return

        alpha = 0.1
        prev_time = time.time()

        while self.running:
            ret, raw_frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, raw_frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

            h, w = raw_frame.shape[:2]
            if self.max_res > 0 and h > self.max_res:
                scale = self.max_res / float(h)
                raw_frame = cv2.resize(raw_frame, (int(w * scale), self.max_res))

            t0 = time.time()
            if self.apply_enhancement:
                enhanced_frame = self.enhancer.enhance(raw_frame)
            else:
                enhanced_frame = raw_frame.copy()
            self.t_enh = (time.time() - t0) * 1000.0

            annotated_frame, counts, t_infer = self.detector.detect_and_annotate(enhanced_frame)
            self.t_yolo = t_infer
            self.counts = counts

            if self.side_by_side:
                raw_labeled = raw_frame.copy()
                cv2.rectangle(raw_labeled, (0, 0), (raw_labeled.shape[1], 36), (15, 20, 25), -1)
                cv2.putText(raw_labeled, "RAW UNDERWATER FEED", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 100, 255), 2, cv2.LINE_AA)

                cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], 36), (15, 20, 25), -1)
                cv2.putText(annotated_frame, "ENHANCED + YOLOv8 DETECTIONS", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 128), 2, cv2.LINE_AA)

                if self.split_orientation == "vertical":
                    if raw_labeled.shape[1] != annotated_frame.shape[1]:
                        raw_labeled = cv2.resize(raw_labeled, (annotated_frame.shape[1], annotated_frame.shape[0]))
                    composite = np.vstack((raw_labeled, annotated_frame))
                else:
                    if raw_labeled.shape[0] != annotated_frame.shape[0]:
                        raw_labeled = cv2.resize(raw_labeled, (annotated_frame.shape[1], annotated_frame.shape[0]))
                    composite = np.hstack((raw_labeled, annotated_frame))
            else:
                composite = annotated_frame

            curr_time = time.time()
            fps_instant = 1.0 / max(curr_time - prev_time, 1e-5)
            self.fps = (alpha * fps_instant) + ((1.0 - alpha) * self.fps)
            prev_time = curr_time

            success, jpeg_buf = cv2.imencode(".jpg", composite, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if success:
                with self.lock:
                    self.latest_jpeg = jpeg_buf.tobytes()

            time.sleep(0.005)

        cap.release()
