"""
train_yolo.py
=============
Fine-tunes a YOLOv8 Nano model for underwater object detection.
Configured with memory-conservative defaults suitable for 4GB VRAM or CPU training.
"""

import argparse
import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("tools.train_yolo")

try:
    import torch
    from ultralytics import YOLO
except ImportError as err:
    logger.error(f"Dependency Missing: {err}\nPlease run: pip install -r requirements.txt")
    sys.exit(1)


def check_gpu_environment():
    """Detects available hardware accelerators and configures placement."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"CUDA Device 0: {gpu_name} ({total_vram_gb:.2f} GB VRAM)")
        return "0"
    logger.warning("CUDA not detected. PyTorch will train on CPU.")
    return "cpu"


def train_underwater_yolo(args):
    """Executes the YOLOv8 Nano fine-tuning routine."""
    device = args.device if args.device else check_gpu_environment()

    if not os.path.exists(args.data):
        logger.error(f"Dataset config '{args.data}' not found. Provide a valid path via --data.")
        sys.exit(1)

    logger.info(f"Loading base model: {args.model}")
    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        imgsz=args.imgsz,
        device=device,
        amp=args.amp,
        cache=args.cache,
        project=args.project,
        name=args.name,
        patience=args.patience,
        save=True,
        val=True,
        verbose=True
    )

    save_dir = getattr(model.trainer, "save_dir", f"{args.project}/{args.name}")
    logger.info(f"Training complete. Best weights: {os.path.join(str(save_dir), 'weights', 'best.pt')}")
    return results


def parse_arguments():
    """Parses training hyperparameters from the command line."""
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 Nano for Underwater Object Detection")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml dataset config file")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model weights (default: yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--workers", type=int, default=2, help="Dataloader worker processes")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--device", type=str, default=None, help="Compute device ('0', 'cpu', or auto-detect)")
    parser.add_argument("--amp", action="store_true", default=True, help="Enable Automatic Mixed Precision (FP16)")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable Automatic Mixed Precision")
    parser.add_argument("--cache", action="store_true", default=False, help="Cache dataset images in memory")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--project", type=str, default="runs/detect", help="Output directory")
    parser.add_argument("--name", type=str, default="underwater_yolov8n", help="Run experiment name")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    train_underwater_yolo(args)
