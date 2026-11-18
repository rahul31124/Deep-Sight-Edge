"""
core/pipeline.py
================
Unified Video Processing Pipeline.
Integrates continuous frame ingestion, Rust SIMD optical enhancement, and Ultralytics YOLO inference.
Supports live dynamic video source switching, video uploads, and zero-latency hot-swapping between
YOLO26 Nano and YOLOv8 Nano via detector caching.
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
from core.detector import DebrisDetector, resolve_model_path


def resolve_video_source(source: str | int) -> str | int:
    """Resolves video source by checking numeric index, V4L2 device nodes, and local paths."""
    v_str = str(source).strip()
    if v_str.isdigit():
        return int(v_str)

    if v_str.startswith("/dev/video"):
        return v_str

    candidates = [
        v_str,
        v_str + ".mp4",
        os.path.join("videos", v_str),
        os.path.join("videos", v_str + ".mp4"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return v_str


class EdgePipelineManager:
    """Coordinates camera capture, optical restoration, object detection, and streaming."""

    def __init__(
        self,
        video_source: str = "videos/gg.mp4",
        model_path: str = None,
        max_res: int = 720,
        conf_thresh: float = 0.25,
        device: str = None
    ):
        self.video_source = resolve_video_source(video_source)
        self.max_res = max_res
        self.device = device
        self.initial_conf = conf_thresh

        active_model = resolve_model_path(model_path)
        self.enhancer = UnderwaterEnhancer()

        self.detectors = {}
        primary = DebrisDetector(model_path=active_model, conf_thresh=conf_thresh, device=device)
        self.detectors[primary.model_type] = primary
        self.detector = primary

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
        self._pending_source = None

        threading.Thread(target=self._preload_secondary_detector, daemon=True).start()

    def _preload_secondary_detector(self):
        """Asynchronously pre-loads the alternate model to enable instant zero-lag switching."""
        try:
            cur_type = self.detector.model_type
            alt_path = "edge_model.pt" if cur_type == "yolo26n" else "yolo26n.pt"
            if not os.path.exists(alt_path) and alt_path == "edge_model.pt":
                alt_path = "yolov8n.pt" if os.path.exists("yolov8n.pt") else None

            if alt_path and os.path.exists(alt_path):
                alt_detector = DebrisDetector(model_path=alt_path, conf_thresh=self.detector.conf_thresh, device=self.device)
                self.detectors[alt_detector.model_type] = alt_detector
                logger.info(f"Pre-cached secondary detector: {alt_detector.model_display_name}")
        except Exception as e:
            logger.warning(f"Secondary detector pre-cache note: {e}")

    def start(self):
        """Starts the background worker thread."""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.thread.start()
        logger.info(f"Edge Video Pipeline worker started with source '{self.video_source}'.")

    def stop(self):
        """Stops the processing loop cleanly."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        logger.info("Edge Video Pipeline worker stopped.")

    def change_source(self, new_source: str | int) -> bool:
        """Schedules a video source change on the fly."""
        resolved = resolve_video_source(new_source)
        logger.info(f"Requesting video source change to: '{resolved}'")
        with self.lock:
            self._pending_source = resolved
        return True

    def update_config(self, apply_enh=None, side_by_side=None, conf=None, red_gain=None, clip_limit=None, split_orientation=None, model_choice=None):
        """Applies dynamic runtime parameter updates from the dashboard."""
        if apply_enh is not None:
            self.apply_enhancement = bool(apply_enh)
        if side_by_side is not None:
            self.side_by_side = bool(side_by_side)
        if split_orientation is not None:
            self.split_orientation = str(split_orientation)
        if conf is not None:
            for det in self.detectors.values():
                det.update_confidence(conf)
            self.detector.update_confidence(conf)
        self.enhancer.update_parameters(clip_limit=clip_limit, red_gain=red_gain)

        if model_choice is not None:
            choice = str(model_choice).lower().strip()
            target_key = "yolo26n" if "26" in choice else "yolo8n"
            if target_key in self.detectors:
                self.detector = self.detectors[target_key]
                logger.info(f"Instantly switched active detector to: {self.detector.model_display_name} (from cache)")
            else:
                target_model = "yolo26n.pt" if target_key == "yolo26n" else ("edge_model.pt" if os.path.exists("edge_model.pt") else "yolov8n.pt")
                if target_model and os.path.exists(target_model):
                    cur_conf = self.detector.conf_thresh
                    dev = self.detector.device
                    new_det = DebrisDetector(model_path=target_model, conf_thresh=cur_conf, device=dev)
                    self.detectors[new_det.model_type] = new_det
                    self.detector = new_det
                    logger.info(f"Switched active detector to: {new_det.model_display_name}")

    def get_telemetry(self) -> dict:
        """Returns instantaneous pipeline performance and detection metrics."""
        source_name = os.path.basename(str(self.video_source)) if isinstance(self.video_source, str) else f"Camera {self.video_source}"
        return {
            "fps": round(self.fps, 1),
            "enhancement_ms": round(self.t_enh, 1),
            "yolo_ms": round(self.t_yolo, 1),
            "inference_ms": round(self.t_yolo, 1),
            "active_model": self.detector.model_type,
            "model_name": self.detector.model_display_name,
            "video_source": source_name,
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

    def _open_capture(self, source: str | int) -> cv2.VideoCapture:
        """Helper to create and configure cv2.VideoCapture."""
        cap = cv2.VideoCapture(source)
        if isinstance(source, int) or (isinstance(source, str) and source.startswith("/dev/video")):
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
        return cap

    def _processing_loop(self):
        """Core frame ingestion, enhancement, inference, and encoding loop."""
        cap = self._open_capture(self.video_source)
        if not cap.isOpened():
            logger.error(f"Could not open initial video source: '{self.video_source}'")

        alpha = 0.1
        prev_time = time.time()

        while self.running:
            with self.lock:
                pending = self._pending_source
                self._pending_source = None

            if pending is not None:
                if cap is not None and cap.isOpened():
                    cap.release()
                new_cap = self._open_capture(pending)
                if new_cap.isOpened():
                    cap = new_cap
                    self.video_source = pending
                    logger.info(f"Video source switched to: '{pending}'")
                else:
                    logger.error(f"Failed to open new video source: '{pending}', reverting to '{self.video_source}'")
                    cap = self._open_capture(self.video_source)

            if cap is None or not cap.isOpened():
                time.sleep(0.1)
                continue

            ret, raw_frame = cap.read()
            if not ret:
                if not (isinstance(self.video_source, int) or (isinstance(self.video_source, str) and self.video_source.startswith("/dev/video"))):
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

            active_det = self.detector
            annotated_frame, counts, t_infer = active_det.detect_and_annotate(enhanced_frame)
            self.t_yolo = t_infer
            self.counts = counts

            if self.side_by_side:
                raw_labeled = raw_frame.copy()
                cv2.rectangle(raw_labeled, (0, 0), (raw_labeled.shape[1], 36), (15, 20, 25), -1)
                cv2.putText(raw_labeled, "RAW UNDERWATER FEED", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 100, 255), 2, cv2.LINE_AA)

                cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], 36), (15, 20, 25), -1)
                infer_label = f"ENHANCED + {active_det.model_display_name.upper()}"
                cv2.putText(annotated_frame, infer_label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 128), 2, cv2.LINE_AA)

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

        if cap is not None and cap.isOpened():
            cap.release()
