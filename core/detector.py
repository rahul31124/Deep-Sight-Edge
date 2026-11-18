"""
core/detector.py
================
Unified Edge Object Detection Module.
Supports Ultralytics YOLO architectures:
1. YOLO26 Nano (yolo26n.pt): Ultra-lightweight NMS-free edge neural network.
2. YOLOv8 Nano (edge_model.pt): Fine-tuned underwater baseline detector.
"""

import os
import time
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

COCO_TO_MARINE = {
    "bird": "Bio", "fish": "Bio", "cat": "Bio", "dog": "Bio", "horse": "Bio", "sheep": "Bio",
    "cow": "Bio", "elephant": "Bio", "bear": "Bio", "zebra": "Bio", "giraffe": "Bio",
    "person": "Bio", "animal": "Bio", "potted plant": "Bio",
    "boat": "Submersible_Gear", "surfboard": "Submersible_Gear", "sports ball": "Submersible_Gear",
    "frisbee": "Submersible_Gear", "skis": "Submersible_Gear", "snowboard": "Submersible_Gear",
    "baseball bat": "Submersible_Gear", "baseball glove": "Submersible_Gear", "skateboard": "Submersible_Gear",
    "tennis racket": "Submersible_Gear", "chair": "Submersible_Gear", "bench": "Submersible_Gear",
    "clock": "Submersible_Gear"
}


def resolve_model_path(path: str = None) -> str:
    """Locates active weights file, prioritizing domain fine-tuned edge_model.pt then yolo26n.pt."""
    if path and os.path.exists(path):
        return path
    for candidate in [
        "edge_model.pt",
        "yolo26n.pt",
        "yolov8n.pt",
        os.path.join("models", "edge_model.pt"),
        os.path.join("models", "yolo26n.pt"),
        os.path.join("models", "yolov8n.pt"),
        "best.pt"
    ]:
        if os.path.exists(candidate):
            return candidate
    return path or "edge_model.pt"


class DebrisDetector:
    """Manages YOLO26 Nano or YOLOv8 Nano inference and bounding box drawing."""

    def __init__(self, model_path: str = "yolo26n.pt", conf_thresh: float = 0.35, device: str = None):
        self.model_path = resolve_model_path(model_path)
        self.conf_thresh = conf_thresh

        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = f"cuda:{device}" if device.isdigit() else device

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file '{self.model_path}' not found.")
        base_name = os.path.basename(self.model_path).lower()
        if "26" in base_name:
            self.model_type = "yolo26n"
            self.model_display_name = "YOLO26 Nano"
        else:
            self.model_type = "yolo8n"
            self.model_display_name = "YOLOv8 Nano"

        self.model = YOLO(self.model_path)
        self.model.to(self.device)

        raw_names = getattr(self.model, "names", {0: "Bio", 1: "Submersible_Gear", 2: "Trash"})
        if isinstance(raw_names, list):
            self.raw_class_names = {i: name for i, name in enumerate(raw_names)}
        else:
            self.raw_class_names = raw_names

        has_marine_classes = any(c in ["Bio", "Submersible_Gear", "Trash", "Rov"] for c in self.raw_class_names.values())
        self.is_custom_marine = has_marine_classes

        if self.is_custom_marine:
            self.class_names = {k: ("Submersible_Gear" if v == "Rov" else v) for k, v in self.raw_class_names.items()}
        else:
            self.class_names = self.raw_class_names

        logger.info(f"Loaded {self.model_display_name} weights: '{self.model_path}' on {self.device.upper()}")

    def update_confidence(self, conf: float):
        """Updates minimum detection confidence threshold."""
        self.conf_thresh = max(0.05, min(0.95, float(conf)))

    def detect_and_annotate(self, frame: np.ndarray) -> tuple[np.ndarray, dict, float]:
        """
        Runs object detection inference on the input image using Ultralytics YOLO.

        Returns:
            tuple: (annotated_frame, counts_dict, inference_latency_ms)
        """
        annotated = frame.copy()
        counts = {"Trash": 0, "Bio": 0, "Submersible_Gear": 0}

        results = self.model.predict(
            source=frame,
            conf=self.conf_thresh,
            device=self.device,
            verbose=False
        )

        inference_ms = 0.0
        if results and len(results) > 0:
            res = results[0]
            if hasattr(res, "speed") and "inference" in res.speed:
                inference_ms = res.speed["inference"]

            for box in res.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                raw_name = self.raw_class_names.get(cls_id, f"Class {cls_id}")

                if self.is_custom_marine:
                    category = self.class_names.get(cls_id, "Trash")
                    if category not in counts:
                        counts[category] = 0
                    counts[category] += 1
                    display_label = f"{category} {conf:.2f}"
                    color = CLASS_COLORS.get(cls_id, (0, 255, 128))
                else:
                    lower_name = raw_name.lower()
                    marine_cat = COCO_TO_MARINE.get(lower_name, "Trash")
                    counts[marine_cat] = counts.get(marine_cat, 0) + 1
                    display_label = f"{marine_cat} {conf:.2f}"
                    
                    if marine_cat == "Bio":
                        color = CLASS_COLORS[0]
                    elif marine_cat == "Submersible_Gear":
                        color = CLASS_COLORS[1]
                    else:
                        color = CLASS_COLORS[2]

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

                (tw, th), baseline = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                py1 = max(0, y1 - th - baseline - 4)
                py2 = y1
                cv2.rectangle(annotated, (x1, py1), (x1 + tw + 6, py2), color, -1)
                cv2.putText(
                    annotated, display_label,
                    (x1 + 3, py2 - baseline - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 15, 20), 1, cv2.LINE_AA
                )

        return annotated, counts, inference_ms
