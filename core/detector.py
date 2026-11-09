"""
core/detector.py
================
YOLOv8 Nano edge inference module.
Handles device placement, confidence filtering, and non-blocking bounding box annotation.
"""

import os
import logging
import cv2
import torch
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger("core.detector")


CLASS_COLORS = {
    0: (0, 215, 255),   # Bio / Marine Life (Amber Gold)
    1: (255, 128, 0),   # Submersible Gear  (Azure Blue)
    2: (0, 0, 255)      # Marine Debris     (Coral Red)
}


def resolve_model_path(path: str = None) -> str:
    """Locates active weights file, checking root, models/ folder, and fallback candidates."""
    if path and os.path.exists(path):
        return path
    for candidate in [
        "edge_model.pt",
        os.path.join("models", "edge_model.pt"),
        "best.pt",
        os.path.join("models", "best.pt"),
        "yolov8n.pt"
    ]:
        if os.path.exists(candidate):
            return candidate
    return path or "edge_model.pt"


class DebrisDetector:
    """Manages YOLOv8 Nano inference and bounding box drawing."""

    def __init__(self, model_path: str = "edge_model.pt", conf_thresh: float = 0.20, device: str = None):
        self.model_path = resolve_model_path(model_path)
        self.conf_thresh = conf_thresh

        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = f"cuda:{device}" if device.isdigit() else device

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file '{self.model_path}' not found!")

        self.model = YOLO(self.model_path)
        self.model.to(self.device)

        raw_names = getattr(self.model, "names", {0: "Bio", 1: "Submersible_Gear", 2: "Trash"})
        self.class_names = {k: ("Submersible_Gear" if v == "Rov" else v) for k, v in raw_names.items()}
        logger.info(f"Loaded YOLOv8 model: '{self.model_path}' on {self.device.upper()} | Classes: {self.class_names}")

    def update_confidence(self, conf: float):
        """Updates minimum detection confidence threshold."""
        self.conf_thresh = max(0.05, min(0.95, float(conf)))

    def detect_and_annotate(self, frame: np.ndarray) -> tuple[np.ndarray, dict, float]:
        """
        Runs object detection inference on the input image.

        Returns:
            tuple: (annotated_frame, counts_dict, inference_latency_ms)
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf_thresh,
            device=self.device,
            verbose=False
        )

        annotated = frame.copy()
        counts = {name: 0 for name in self.class_names.values()}
        inference_ms = 0.0

        if results and len(results) > 0:
            res = results[0]
            if hasattr(res, "speed") and "inference" in res.speed:
                inference_ms = res.speed["inference"]

            for box in res.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = self.class_names.get(cls_id, f"Class {cls_id}")
                counts[cls_name] = counts.get(cls_name, 0) + 1

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                color = CLASS_COLORS.get(cls_id, (0, 255, 0))

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

                tag = f"{cls_name} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                tag_y1 = max(0, y1 - th - 6)
                cv2.rectangle(annotated, (x1, tag_y1), (x1 + tw + 8, y1), color, -1)
                cv2.putText(annotated, tag, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 15, 20), 1, cv2.LINE_AA)

        return annotated, counts, inference_ms
