"""
benchmark_pipeline.py
=====================
Quantitative Latency, Throughput (FPS), and VRAM Benchmark Tool
for the Underwater Optical Enhancement (Rust Core) & YOLOv8 Detection Pipeline.
"""

import argparse
import os
import sys
import time
import logging

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("tools.benchmark")

import cv2
import numpy as np

try:
    import torch
    from ultralytics import YOLO
except ImportError as err:
    logger.error(f"PyTorch or Ultralytics missing: {err}. Please run: pip install -r requirements.txt")
    sys.exit(1)

from core.enhancer import UnderwaterEnhancer
from core.detector import DebrisDetector
from core.pipeline import resolve_video_source


def run_benchmark(video_path: str, model_path: str, num_frames: int = 150, device: str = None):
    """Executes a multi-frame latency and throughput benchmark across all pipeline stages."""
    video_path = resolve_video_source(video_path)
    if not os.path.exists(video_path):
        logger.error(f"Video file '{video_path}' not found!")
        sys.exit(1)

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif device.isdigit():
        device = f"cuda:{device}"

    actual_model = model_path if os.path.exists(model_path) else "edge_model.pt"

    logger.info("=" * 65)
    logger.info("UNDERWATER PIPELINE QUANTITATIVE BENCHMARK")
    logger.info("=" * 65)
    logger.info(f"Input Video   : {video_path}")
    logger.info(f"YOLO Weights  : {actual_model}")
    logger.info(f"Test Device   : {device.upper()}")
    logger.info(f"Sample Window : {num_frames} frames")
    if torch.cuda.is_available() and device != "cpu":
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"GPU Model     : {gpu_name} ({vram_total:.2f} GB VRAM)")
        torch.cuda.reset_peak_memory_stats(0)

    enhancer = UnderwaterEnhancer()
    detector = DebrisDetector(model_path=actual_model, conf_thresh=0.20, device=device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        sys.exit(1)

    logger.info("Warming up pipeline components (10 iterations)...")
    for _ in range(10):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        if ret:
            enh = enhancer.enhance(frame)
            _ = detector.detect_and_annotate(enh)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    t_decode = []
    t_enh = []
    t_yolo = []
    t_total = []

    logger.info(f"Benchmarking across {num_frames} frames...")
    start_total_bench = time.perf_counter()

    for _ in range(num_frames):
        t0 = time.perf_counter()
        ret, raw_frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, raw_frame = cap.read()
            if not ret:
                break
        t1 = time.perf_counter()

        enhanced_frame = enhancer.enhance(raw_frame)
        t2 = time.perf_counter()

        annotated_frame, counts, infer_ms = detector.detect_and_annotate(enhanced_frame)
        t3 = time.perf_counter()

        t_decode.append((t1 - t0) * 1000.0)
        t_enh.append((t2 - t1) * 1000.0)
        t_yolo.append((t3 - t2) * 1000.0)
        t_total.append((t3 - t0) * 1000.0)

    cap.release()
    total_duration = time.perf_counter() - start_total_bench
    effective_fps = len(t_total) / max(total_duration, 1e-5)

    def stats(arr):
        return np.mean(arr), np.median(arr), np.std(arr), np.percentile(arr, 95)

    m_dec, med_dec, std_dec, p95_dec = stats(t_decode)
    m_enh, med_enh, std_enh, p95_enh = stats(t_enh)
    m_yolo, med_yolo, std_yolo, p95_yolo = stats(t_yolo)
    m_tot, med_tot, std_tot, p95_tot = stats(t_total)

    logger.info("=" * 65)
    logger.info(f"BENCHMARK RESULTS (Averaged over {len(t_total)} frames)")
    logger.info("=" * 65)
    logger.info(f"{'Pipeline Stage':<30} | {'Mean (ms)':<10} | {'Median':<8} | {'95th %':<8} | {'% Time':<6}")
    logger.info("-" * 65)
    logger.info(f"{'1. Video Ingest & Decode':<30} | {m_dec:8.2f}ms | {med_dec:6.2f}ms | {p95_dec:6.2f}ms | {(m_dec/m_tot)*100:5.1f}%")
    logger.info(f"{'2. Optical Restoration (Rust)':<30} | {m_enh:8.2f}ms | {med_enh:6.2f}ms | {p95_enh:6.2f}ms | {(m_enh/m_tot)*100:5.1f}%")
    logger.info(f"{'3. YOLOv8 Detection & Draw':<30} | {m_yolo:8.2f}ms | {med_yolo:6.2f}ms | {p95_yolo:6.2f}ms | {(m_yolo/m_tot)*100:5.1f}%")
    logger.info("-" * 65)
    logger.info(f"{'FULL PIPELINE LATENCY':<30} | {m_tot:8.2f}ms | {med_tot:6.2f}ms | {p95_tot:6.2f}ms | 100.0%")
    logger.info("=" * 65)
    logger.info(f"Effective Edge Throughput : {effective_fps:.1f} FPS")

    if torch.cuda.is_available() and device != "cpu":
        peak_vram_mb = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
        logger.info(f"Peak GPU VRAM Allocated   : {peak_vram_mb:.1f} MB")


def parse_args():
    """Parses benchmark configuration arguments."""
    parser = argparse.ArgumentParser(description="Benchmark Underwater AI Pipeline")
    parser.add_argument("--video", type=str, default="test.mp4", help="Video file to benchmark")
    parser.add_argument("--model", type=str, default="edge_model.pt", help="YOLO model path")
    parser.add_argument("--frames", type=int, default=100, help="Number of frames to sample")
    parser.add_argument("--device", type=str, default=None, help="Device ('0' for CUDA GPU, 'cpu')")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(
        video_path=args.video,
        model_path=args.model,
        num_frames=args.frames,
        device=args.device
    )
