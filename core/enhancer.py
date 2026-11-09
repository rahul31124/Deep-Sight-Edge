"""
core/enhancer.py
================
Hybrid Python-Rust Underwater Optical Restoration Engine.
Executes red-light absorption compensation, simplest color balance,
and LAB CLAHE contrast enhancement via Rust C-FFI or vectorized OpenCV fallback.
"""

import os
import ctypes
import logging
import cv2
import numpy as np

logger = logging.getLogger("core.enhancer")


class RustEnhancerBridge:
    """Interfacing layer for the compiled Rust native C-FFI library."""

    def __init__(self):
        self.lib = None
        self.loaded = False

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        search_paths = [
            os.path.join(base_dir, "rust_core", "target", "release", "rust_core.dll"),
            os.path.join(base_dir, "rust_core", "target", "release", "librust_core.so"),
            os.path.join(base_dir, "rust_core.dll"),
            os.path.join(base_dir, "librust_core.so")
        ]

        for path in search_paths:
            if os.path.exists(path):
                try:
                    self.lib = ctypes.CDLL(path)
                    ver = self.lib.rust_engine_version()
                    self.loaded = True
                    logger.info(f"Loaded Rust Optical Engine: '{path}' (v{ver})")
                    break
                except Exception as err:
                    logger.warning(f"Could not load Rust library at '{path}': {err}")

        if not self.loaded:
            logger.info("Running in optimized OpenCV fallback mode (Rust core not loaded).")

    def red_compensate(self, frame: np.ndarray, red_gain: float) -> np.ndarray:
        """Applies in-place red channel amplification via Rust SIMD."""
        if not self.loaded or abs(red_gain - 1.0) < 0.02:
            return frame

        if not frame.flags['C_CONTIGUOUS']:
            frame = np.ascontiguousarray(frame)

        ptr = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        self.lib.rust_red_compensate(ptr, frame.size, ctypes.c_float(red_gain))
        return frame

    def color_balance(self, frame: np.ndarray, clip_percentile: float) -> np.ndarray:
        """Applies in-place simplest color balance contrast stretching via Rust."""
        if not self.loaded:
            return frame

        if not frame.flags['C_CONTIGUOUS']:
            frame = np.ascontiguousarray(frame)

        ptr = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        self.lib.rust_color_balance(ptr, frame.size, ctypes.c_float(clip_percentile))
        return frame


class UnderwaterEnhancer:
    """Three-stage optical color restoration and contrast recovery pipeline."""

    def __init__(
        self,
        clip_limit: float = 2.0,
        grid_size: int = 8,
        clip_percentile: float = 1.0,
        red_gain: float = 1.25,
        gamma: float = 1.0
    ):
        self.clip_limit = clip_limit
        self.grid_size = grid_size
        self.clip_percentile = clip_percentile
        self.red_gain = red_gain
        self.gamma = gamma

        self.clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(self.grid_size, self.grid_size))
        self.rust_bridge = RustEnhancerBridge()
        self.update_luts()

    def update_luts(self):
        """Precomputes lookup tables for sub-millisecond gamma transforms."""
        if abs(self.gamma - 1.0) > 0.05:
            inv_gamma = 1.0 / max(0.1, self.gamma)
            self.gamma_lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
        else:
            self.gamma_lut = None

    def update_parameters(self, clip_limit=None, red_gain=None, clip_percentile=None):
        """Dynamically applies updated calibration parameters from the web UI."""
        if clip_limit is not None and clip_limit != self.clip_limit:
            self.clip_limit = max(0.5, float(clip_limit))
            self.clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(self.grid_size, self.grid_size))

        if red_gain is not None:
            self.red_gain = max(0.5, min(3.0, float(red_gain)))

        if clip_percentile is not None:
            self.clip_percentile = max(0.01, min(5.0, float(clip_percentile)))

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """Executes full underwater color restoration pipeline on an input BGR frame."""
        out = frame.copy()

        if self.rust_bridge.loaded:
            out = self.rust_bridge.red_compensate(out, self.red_gain)
            out = self.rust_bridge.color_balance(out, self.clip_percentile)
        else:
            b, g, r = cv2.split(out)
            if abs(self.red_gain - 1.0) > 0.02:
                r_lut = np.clip(np.arange(256, dtype=np.float32) * self.red_gain, 0, 255).astype(np.uint8)
                r = cv2.LUT(r, r_lut)

            p = max(0.01, self.clip_percentile)
            ch_list = []
            for ch in (b, g, r):
                hist = cv2.calcHist([ch], [0], None, [256], [0, 256]).ravel()
                cum = np.cumsum(hist)
                tot = ch.size
                low = np.searchsorted(cum, tot * (p / 100.0))
                high = np.searchsorted(cum, tot * (1.0 - p / 100.0))
                if high > low:
                    lut = np.clip((np.arange(256, dtype=np.float32) - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
                    ch_list.append(cv2.LUT(ch, lut))
                else:
                    ch_list.append(ch)
            out = cv2.merge(ch_list)

        if self.gamma_lut is not None:
            out = cv2.LUT(out, self.gamma_lut)

        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b_ch = cv2.split(lab)
        l_clahe = self.clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l_clahe, a, b_ch)), cv2.COLOR_LAB2BGR)
