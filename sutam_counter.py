# -*- coding: utf-8 -*-
"""Single window stroke counter application for sofrim.

This module implements a Tkinter based GUI that streams a video feed from an
OpenCV compatible camera and performs two kinds of quill tracking:

``Stripe mode``
    Detects a coloured stripe pattern running along the quill using HSV or
    intensity thresholds.  The far end of the stripe is taken as reference and
    the actual tip position is estimated using a configurable offset.  A robust
    line (fitted with RANSAC) and temporal smoothing stabilise the overlay even
    when fingers occlude parts of the quill.

``Rings mode``
    Falls back to tracking two coloured rings as used by the legacy tool.

Both modes rely on ArUco markers to establish a metric scale on the parchment.
The window exposes menus for starting/stopping the capture, colour calibration,
manual tip alignment, camera selection and PDF printing utilities.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from logging.handlers import RotatingFileHandler
from PIL import Image, ImageTk

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except Exception as exc:  # pragma: no cover - Tkinter should always be present
    raise RuntimeError("Tkinter is required to run the GUI") from exc

from print_aruco import create_pdf as create_aruco_pdf, create_rails_pdf
from print_stripe import create_pdf as create_stripe_pdf


if not hasattr(cv2, "aruco"):
    raise ImportError(
        "cv2.aruco is unavailable. Install the 'opencv-contrib-python' package to "
        "enable ArUco marker support."
    )


APP_NAME = "Sofrim Stroke Counter"
APP_VERSION = "0.2.0"
CONFIG_FILE = Path("config.json")
DEFAULT_FRAME_SIZE = (1920, 1080)
DEFAULT_FPS = 30
CAMERA_PROBE_ORDER = [1, 2, 0, 3, 4]
CAMERA_BACKENDS = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
ARUCO_DICTIONARY_ID = cv2.aruco.DICT_4X4_1000
RANSAC_ITERATIONS = 200
RANSAC_THRESHOLD_PX = 2.5
EMA_ALPHA = 0.25


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("sofrim")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler("sofrim.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    return logger


def get_git_info() -> Tuple[str, str, str]:
    """Return (branch, short_hash, last_update_iso).

    last_update_iso is the ISO datetime of the last commit (git),
    or the file mtime if git is unavailable.
    """

    try:
        repo_dir = Path(__file__).resolve().parent
        env = os.environ.copy()
        short_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=str(repo_dir),
            env=env,
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=str(repo_dir),
            env=env,
        ).strip()
        # ISO-8601 strict commit datetime (e.g. 2025-10-28T23:41:12+02:00)
        last_update_iso = subprocess.check_output(
            ["git", "log", "-1", "--format=%cd", "--date=iso-strict"],
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=str(repo_dir),
            env=env,
        ).strip()
        return branch, short_hash, last_update_iso
    except Exception:
        # Fallback: use file mtime in local time as ISO string
        try:
            mtime = Path(__file__).stat().st_mtime
            last_update_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(mtime))
            # insert colon in timezone (+0200 -> +02:00) for ISO compliance
            if len(last_update_iso) >= 24 and (last_update_iso[-5] in ["+", "-"]):
                last_update_iso = last_update_iso[:-2] + ":" + last_update_iso[-2:]
        except Exception:
            last_update_iso = "unknown"
        return "unknown", "unknown", last_update_iso


def get_build_string() -> str:
    version = APP_VERSION
    try:
        here = Path(__file__).resolve().parent
        version_file = here / "VERSION"
        if version_file.exists():
            version = version_file.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    branch, short_hash, last_update_iso = get_git_info()
    core = f"v{version}"
    if short_hash != "unknown":
        core += f" ({short_hash})"
    if branch != "unknown":
        core += f" [{branch}]"
    if last_update_iso != "unknown":
        core += f" — updated {last_update_iso}"
    return core


# ---------------------------------------------------------------------------
# Configuration models
# ---------------------------------------------------------------------------


def _as_tuple(values: Iterable[float]) -> Tuple[float, ...]:
    return tuple(float(v) for v in values)


@dataclass
class StripeConfig:
    palette: str = "green-white"
    hsv_low: Tuple[int, int, int] = (55, 100, 100)
    hsv_high: Tuple[int, int, int] = (78, 255, 255)
    use_ticks: bool = True
    tick_mm: float = 5.0
    min_area_px: int = 500
    tip_reference: str = "far_end"
    tip_offset_mm: float = 5.0
    manual_offset_mm: Tuple[float, float] = (0.0, 0.0)
    ema_alpha: float = 0.25
    max_jump_mm: float = 3.0
    min_inlier_ratio: float = 0.6
    lost_reacq_frames: int = 12
    roi_pad_mm: float = 8.0
    angle_gate_deg: float = 25.0

    @staticmethod
    def from_mapping(data: Dict[str, object]) -> "StripeConfig":
        return StripeConfig(
            palette=str(data.get("palette", "green-white")),
            hsv_low=tuple(int(v) for v in data.get("hsv_low", (55, 100, 100))),
            hsv_high=tuple(int(v) for v in data.get("hsv_high", (78, 255, 255))),
            use_ticks=bool(data.get("use_ticks", True)),
            tick_mm=float(data.get("tick_mm", 5.0)),
            min_area_px=int(data.get("min_area_px", 500)),
            tip_reference=str(data.get("tip_reference", "far_end")),
            tip_offset_mm=float(data.get("tip_offset_mm", 5.0)),
            manual_offset_mm=_as_tuple(data.get("manual_offset_mm", (0.0, 0.0))),
            ema_alpha=float(data.get("ema_alpha", 0.25)),
            max_jump_mm=float(data.get("max_jump_mm", 3.0)),
            min_inlier_ratio=float(data.get("min_inlier_ratio", 0.6)),
            lost_reacq_frames=int(data.get("lost_reacq_frames", 12)),
            roi_pad_mm=float(data.get("roi_pad_mm", 8.0)),
            angle_gate_deg=float(data.get("angle_gate_deg", 25.0)),
        )

    def to_mapping(self) -> Dict[str, object]:
        return {
            "palette": self.palette,
            "hsv_low": list(self.hsv_low),
            "hsv_high": list(self.hsv_high),
            "use_ticks": self.use_ticks,
            "tick_mm": self.tick_mm,
            "min_area_px": self.min_area_px,
            "tip_reference": self.tip_reference,
            "tip_offset_mm": self.tip_offset_mm,
            "manual_offset_mm": list(self.manual_offset_mm),
            "ema_alpha": self.ema_alpha,
            "max_jump_mm": self.max_jump_mm,
            "min_inlier_ratio": self.min_inlier_ratio,
            "lost_reacq_frames": self.lost_reacq_frames,
            "roi_pad_mm": self.roi_pad_mm,
            "angle_gate_deg": self.angle_gate_deg,
        }


@dataclass
class RingsConfig:
    ringA_low: Tuple[int, int, int] = (20, 90, 130)
    ringA_high: Tuple[int, int, int] = (38, 255, 255)
    ringB_low: Tuple[int, int, int] = (58, 90, 100)
    ringB_high: Tuple[int, int, int] = (78, 255, 255)
    tip_distance_mm: float = 14.0
    manual_offset_mm: Tuple[float, float] = (0.0, 0.0)

    @staticmethod
    def from_mapping(data: Dict[str, object]) -> "RingsConfig":
        return RingsConfig(
            ringA_low=tuple(int(v) for v in data.get("ringA", {}).get("low", (20, 90, 130))),
            ringA_high=tuple(int(v) for v in data.get("ringA", {}).get("high", (38, 255, 255))),
            ringB_low=tuple(int(v) for v in data.get("ringB", {}).get("low", (58, 90, 100))),
            ringB_high=tuple(int(v) for v in data.get("ringB", {}).get("high", (78, 255, 255))),
            tip_distance_mm=float(data.get("tip_distance_mm", 14.0)),
            manual_offset_mm=_as_tuple(data.get("manual_offset_mm", (0.0, 0.0))),
        )

    def to_mapping(self) -> Dict[str, object]:
        return {
            "ringA": {"low": list(self.ringA_low), "high": list(self.ringA_high)},
            "ringB": {"low": list(self.ringB_low), "high": list(self.ringB_high)},
            "tip_distance_mm": self.tip_distance_mm,
            "manual_offset_mm": list(self.manual_offset_mm),
        }


@dataclass
class ColorDualRingsConfig:
    cyan_low: Tuple[int, int, int] = (85, 120, 120)
    cyan_high: Tuple[int, int, int] = (100, 255, 255)
    magenta_low: Tuple[int, int, int] = (160, 120, 120)
    magenta_high: Tuple[int, int, int] = (179, 255, 255)
    front_to_tip_mm: float = 12.0
    nib_lateral_mm: float = 0.0
    nib_side: str = "left"
    min_area_px: int = 150
    ema_alpha: float = 0.25
    max_jump_mm: float = 3.0
    lost_reacq_frames: int = 12
    roi_pad_mm: float = 8.0
    angle_gate_deg: float = 25.0
    manual_offset_mm: Tuple[float, float] = (0.0, 0.0)

    @staticmethod
    def from_mapping(data: Dict[str, object]) -> "ColorDualRingsConfig":
        return ColorDualRingsConfig(
            cyan_low=tuple(int(v) for v in data.get("cyan_low", (85, 120, 120))),
            cyan_high=tuple(int(v) for v in data.get("cyan_high", (100, 255, 255))),
            magenta_low=tuple(int(v) for v in data.get("magenta_low", (160, 120, 120))),
            magenta_high=tuple(int(v) for v in data.get("magenta_high", (179, 255, 255))),
            front_to_tip_mm=float(data.get("front_to_tip_mm", 12.0)),
            nib_lateral_mm=float(data.get("nib_lateral_mm", 0.0)),
            nib_side=str(data.get("nib_side", "left")),
            min_area_px=int(data.get("min_area_px", 150)),
            ema_alpha=float(data.get("ema_alpha", 0.25)),
            max_jump_mm=float(data.get("max_jump_mm", 3.0)),
            lost_reacq_frames=int(data.get("lost_reacq_frames", 12)),
            roi_pad_mm=float(data.get("roi_pad_mm", 8.0)),
            angle_gate_deg=float(data.get("angle_gate_deg", 25.0)),
            manual_offset_mm=_as_tuple(data.get("manual_offset_mm", (0.0, 0.0))),
        )

    def to_mapping(self) -> Dict[str, object]:
        return {
            "cyan_low": list(self.cyan_low),
            "cyan_high": list(self.cyan_high),
            "magenta_low": list(self.magenta_low),
            "magenta_high": list(self.magenta_high),
            "front_to_tip_mm": self.front_to_tip_mm,
            "nib_lateral_mm": self.nib_lateral_mm,
            "nib_side": self.nib_side,
            "min_area_px": self.min_area_px,
            "ema_alpha": self.ema_alpha,
            "max_jump_mm": self.max_jump_mm,
            "lost_reacq_frames": self.lost_reacq_frames,
            "roi_pad_mm": self.roi_pad_mm,
            "angle_gate_deg": self.angle_gate_deg,
            "manual_offset_mm": list(self.manual_offset_mm),
        }


@dataclass
class RailsConfig:
    aruco_dictionary: str = "DICT_4X4_1000"
    marker_size_mm: float = 15.0
    gap_h_mm: float = 12.0
    gap_v_mm: float = 18.0
    top_ids: Tuple[int, ...] = (10, 11, 12, 13)
    side_ids: Tuple[int, ...] = (14, 15, 16, 17)

    @staticmethod
    def from_mapping(data: Dict[str, object]) -> "RailsConfig":
        return RailsConfig(
            aruco_dictionary=str(data.get("aruco_dictionary", "DICT_4X4_1000")),
            marker_size_mm=float(data.get("marker_size_mm", 15.0)),
            gap_h_mm=float(data.get("gap_h_mm", 12.0)),
            gap_v_mm=float(data.get("gap_v_mm", 18.0)),
            top_ids=tuple(int(v) for v in data.get("top_ids", (10, 11, 12, 13))),
            side_ids=tuple(int(v) for v in data.get("side_ids", (14, 15, 16, 17))),
        )

    def to_mapping(self) -> Dict[str, object]:
        return {
            "aruco_dictionary": self.aruco_dictionary,
            "marker_size_mm": self.marker_size_mm,
            "gap_h_mm": self.gap_h_mm,
            "gap_v_mm": self.gap_v_mm,
            "top_ids": list(self.top_ids),
            "side_ids": list(self.side_ids),
        }


@dataclass
class UIConfig:
    language: str = "en"
    single_window: bool = True

    @staticmethod
    def from_mapping(data: Dict[str, object]) -> "UIConfig":
        return UIConfig(
            language=str(data.get("language", "en")),
            single_window=bool(data.get("single_window", True)),
        )

    def to_mapping(self) -> Dict[str, object]:
        return dataclasses.asdict(self)


@dataclass
class AppConfig:
    mode: str = "stripe"
    stripe: StripeConfig = field(default_factory=StripeConfig)
    rings: RingsConfig = field(default_factory=RingsConfig)
    color_dual_rings: ColorDualRingsConfig = field(default_factory=ColorDualRingsConfig)
    rails: RailsConfig = field(default_factory=RailsConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    ink_activity: Dict[str, float] = field(default_factory=dict)

    @staticmethod
    def load(path: Path) -> "AppConfig":
        if not path.exists():
            return AppConfig()
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        cfg = AppConfig(
            mode=str(data.get("mode", "stripe")),
            stripe=StripeConfig.from_mapping(data.get("stripe", {})),
            rings=RingsConfig.from_mapping(data.get("rings", {})),
            color_dual_rings=ColorDualRingsConfig.from_mapping(data.get("color_dual_rings", {})),
            rails=RailsConfig.from_mapping(data.get("rails", {})),
            ui=UIConfig.from_mapping(data.get("ui", {})),
        )
        cfg.ink_activity = dict(data.get("ink_activity", {}))
        return cfg
    def dump(self, path: Path) -> None:
        payload = {
            "mode": self.mode,
            "stripe": self.stripe.to_mapping(),
            "rings": self.rings.to_mapping(),
            "color_dual_rings": self.color_dual_rings.to_mapping(),
            "rails": self.rails.to_mapping(),
            "ui": self.ui.to_mapping(),
            "ink_activity": self.ink_activity,
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def mm_to_pixels(homography: np.ndarray, points_mm: np.ndarray) -> np.ndarray:
    pts = points_mm.reshape(-1, 1, 2).astype(np.float64)
    projected = cv2.perspectiveTransform(pts, homography)
    return projected.reshape(-1, 2)


def pixels_to_mm(homography: np.ndarray, points_px: np.ndarray) -> np.ndarray:
    pts = points_px.reshape(-1, 1, 2).astype(np.float64)
    projected = cv2.perspectiveTransform(pts, homography)
    return projected.reshape(-1, 2)


def normalise(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-6:
        return np.zeros_like(vector)
    return vector / norm


def angle_consistent(new_dir: np.ndarray, prev_dir: Optional[np.ndarray], max_deg: float) -> bool:
    if prev_dir is None:
        return True
    c = float(np.clip(np.dot(normalise(new_dir), normalise(prev_dir)), -1.0, 1.0))
    ang = math.degrees(math.acos(c))
    return ang <= max_deg


class ExponentialMovingAverage:
    """Simple EMA helper used to smooth tip motion."""

    def __init__(self, alpha: float = EMA_ALPHA):
        self.alpha = float(alpha)
        self.value: Optional[np.ndarray] = None

    def update(self, new_value: np.ndarray) -> np.ndarray:
        new_value = np.asarray(new_value, dtype=float)
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1.0 - self.alpha) * self.value
        return self.value

    def reset(self) -> None:
        self.value = None


class InkActivityTracker:
    """Ink-activity with smoothing + hysteresis (frames-on/off)."""

    def __init__(self, window_size: int = 24, threshold_on: float = 6.0, threshold_off: float = 3.0,
                 frames_on: int = 3, frames_off: int = 6):
        self.window_size = int(window_size)
        self.threshold_on = float(threshold_on)
        self.threshold_off = float(threshold_off)
        self.frames_on_req = int(frames_on)
        self.frames_off_req = int(frames_off)
        self.queue: List[float] = []
        self.prev_roi: Optional[np.ndarray] = None
        self.pen_down = False
        self._above = 0
        self._below = 0

    def clear(self) -> None:
        self.queue.clear()
        self.prev_roi = None
        self.pen_down = False
        self._above = 0
        self._below = 0

    def update(self, roi_gray: Optional[np.ndarray]) -> Tuple[bool, float]:
        if roi_gray is None or roi_gray.size == 0:
            self.clear()
            return False, 0.0

        activity = 0.0
        if self.prev_roi is not None and self.prev_roi.shape == roi_gray.shape:
            diff = cv2.absdiff(roi_gray, self.prev_roi)
            activity = float(np.mean(diff))
        self.prev_roi = roi_gray.copy()

        self.queue.append(activity)
        if len(self.queue) > self.window_size:
            self.queue.pop(0)
        smoothed = float(np.mean(self.queue)) if self.queue else 0.0

        # hysteresis
        if smoothed > self.threshold_on:
            self._above += 1; self._below = 0
        elif smoothed < self.threshold_off:
            self._below += 1; self._above = 0
        else:
            self._above = max(0, self._above - 1)
            self._below = max(0, self._below - 1)

        if not self.pen_down and self._above >= self.frames_on_req:
            self.pen_down = True; self._above = 0
        elif self.pen_down and self._below >= self.frames_off_req:
            self.pen_down = False; self._below = 0

        return self.pen_down, smoothed


# ---------------------------------------------------------------------------
# ArUco marker homography logic
# ---------------------------------------------------------------------------


def _aruco_dict_from_name(name: str) -> int:
    attr = name.strip().upper()
    if hasattr(cv2.aruco, attr):
        return getattr(cv2.aruco, attr)
    return ARUCO_DICTIONARY_ID


class MarkerHomography:
    """Handles detection of reference markers and conversion between units."""

    def __init__(self, rails_config: RailsConfig) -> None:
        self.rails_config = rails_config
        dict_id = _aruco_dict_from_name(self.rails_config.aruco_dictionary)
        self.dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
        try:
            self.parameters = cv2.aruco.DetectorParameters()
        except AttributeError:  # OpenCV < 4.7 fallback
            self.parameters = cv2.aruco.DetectorParameters_create()
        try:
            self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
        except AttributeError:  # pragma: no cover - legacy API
            self.detector = None
        self._homography_mm_from_px: Optional[np.ndarray] = None
        self._homography_px_from_mm: Optional[np.ndarray] = None
        self._locked = False
        self.last_corners: List[np.ndarray] = []
        self.last_ids: Optional[np.ndarray] = None
        self._world_lookup = self._build_world_lookup()
        self.marker_ids = list(self._world_lookup.keys())

    def clear(self) -> None:
        self._homography_mm_from_px = None
        self._homography_px_from_mm = None
        self._locked = False
        self.last_corners = []
        self.last_ids = None

    def update(self, frame: np.ndarray) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.detector is not None:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self.dictionary, parameters=self.parameters
            )
        self.last_corners = list(corners) if corners is not None else []
        self.last_ids = ids
        if not self._locked:
            self._attempt_lock(self.last_corners, self.last_ids)
        return self.last_corners, self.last_ids

    def _attempt_lock(
        self, corners: Sequence[np.ndarray], ids: Optional[np.ndarray]
    ) -> bool:
        if ids is None or len(ids) == 0:
            return False

        ids_flat = ids.flatten()
        valid_indices = [idx for idx, marker_id in enumerate(ids_flat) if int(marker_id) in self.marker_ids]
        if len(valid_indices) < 4:
            return False

        image_pts: List[np.ndarray] = []
        world_pts: List[np.ndarray] = []

        for idx in valid_indices:
            marker_int = int(ids_flat[idx])
            c = corners[idx].reshape(-1, 2)
            image_pts.extend(c)
            world_pts.extend(self._world_lookup[marker_int])

        if len(image_pts) < 8:
            return False

        image_pts_np = np.asarray(image_pts, dtype=np.float64)
        world_pts_np = np.asarray(world_pts, dtype=np.float64)

        H, mask = cv2.findHomography(image_pts_np, world_pts_np, cv2.RANSAC, 3.0)
        if H is None or mask is None or int(mask.sum()) < 8:
            return False

        ok, inv = cv2.invert(H)
        if not ok:
            return False

        self._homography_mm_from_px = H
        self._homography_px_from_mm = inv
        self._locked = True
        return True

    def relock(self) -> bool:
        self._homography_mm_from_px = None
        self._homography_px_from_mm = None
        self._locked = False
        return self._attempt_lock(self.last_corners, self.last_ids)

    @property
    def is_locked(self) -> bool:
        return self._locked

    def _build_world_lookup(self) -> Dict[int, List[List[float]]]:
        lookup: Dict[int, List[List[float]]] = {}
        marker_size = float(self.rails_config.marker_size_mm)
        step_x = marker_size + float(self.rails_config.gap_h_mm)
        step_y = marker_size + float(self.rails_config.gap_v_mm)
        margin = float(self.rails_config.gap_h_mm)

        for idx, marker_id in enumerate(self.rails_config.top_ids):
            origin_x = idx * step_x
            origin_y = 0.0
            lookup[int(marker_id)] = [
                [origin_x, origin_y],
                [origin_x + marker_size, origin_y],
                [origin_x + marker_size, origin_y + marker_size],
                [origin_x, origin_y + marker_size],
            ]

        column_x = len(self.rails_config.top_ids) * step_x + margin
        for idx, marker_id in enumerate(self.rails_config.side_ids):
            origin_x = column_x
            origin_y = (idx + 1) * step_y
            lookup[int(marker_id)] = [
                [origin_x, origin_y],
                [origin_x + marker_size, origin_y],
                [origin_x + marker_size, origin_y + marker_size],
                [origin_x, origin_y + marker_size],
            ]

        return lookup

    @property
    def ready(self) -> bool:
        return self._homography_mm_from_px is not None and self._homography_px_from_mm is not None

    @property
    def homography_mm_from_px(self) -> np.ndarray:
        if self._homography_mm_from_px is None:
            raise RuntimeError("Homography not available")
        return self._homography_mm_from_px

    @property
    def homography_px_from_mm(self) -> np.ndarray:
        if self._homography_px_from_mm is None:
            raise RuntimeError("Homography not available")
        return self._homography_px_from_mm

    def project_mm_to_px(self, points: np.ndarray) -> np.ndarray:
        return mm_to_pixels(self.homography_px_from_mm, points)

    def project_px_to_mm(self, points: np.ndarray) -> np.ndarray:
        return pixels_to_mm(self.homography_mm_from_px, points)


# ---------------------------------------------------------------------------
# Stripe mode implementation
# ---------------------------------------------------------------------------


def ransac_line(points: np.ndarray, iterations: int = RANSAC_ITERATIONS, threshold: float = RANSAC_THRESHOLD_PX) -> Tuple[np.ndarray, np.ndarray]:
    """Return point on line and normalised direction estimated with RANSAC."""

    if len(points) < 2:
        raise ValueError("At least two points required")

    best_inliers: List[int] = []
    points = points.reshape(-1, 2)
    rng = np.random.default_rng()

    for _ in range(iterations):
        sample = rng.choice(len(points), size=2, replace=False)
        p1, p2 = points[sample]
        direction = p2 - p1
        norm = np.linalg.norm(direction)
        if norm < 1e-4:
            continue
        direction /= norm
        residuals = np.abs(np.cross(points - p1, direction))
        inliers = np.where(residuals <= threshold)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = list(inliers)

    if not best_inliers:
        best_inliers = list(range(len(points)))

    pts = points[best_inliers]
    mean = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - mean)
    direction = vt[0]
    direction = normalise(direction)
    return mean, direction


@dataclass
class StripeDetectionResult:
    tip_px: Optional[np.ndarray] = None
    tip_mm: Optional[np.ndarray] = None
    far_end_px: Optional[np.ndarray] = None
    contour: Optional[np.ndarray] = None
    bounding_box: Optional[np.ndarray] = None
    roi: Optional[Tuple[int, int, int, int]] = None


class StripeMode:
    def __init__(self, config: StripeConfig, homography: MarkerHomography) -> None:
        self.config = config
        self.homography = homography
        self.tip_filter = ExponentialMovingAverage(alpha=self.config.ema_alpha)
        self.last_result: StripeDetectionResult = StripeDetectionResult()
        self.prev_tip_mm: Optional[np.ndarray] = None
        self.prev_line_dir_mm: Optional[np.ndarray] = None
        self.lost_counter: int = 0

    def reset(self) -> None:
        self.tip_filter.reset()
        self.last_result = StripeDetectionResult()
        self.prev_tip_mm = None
        self.prev_line_dir_mm = None
        self.lost_counter = 0

    def process(self, frame: np.ndarray) -> StripeDetectionResult:
        result = StripeDetectionResult()
        if not self.homography.ready:
            self.reset()
            return result

        rx, ry, rw, rh = self._search_roi(frame.shape)
        if rw <= 0 or rh <= 0:
            self.reset()
            return result

        frame_roi = frame[ry : ry + rh, rx : rx + rw]

        mask = self._create_mask(frame_roi)
        if mask is None:
            self.reset()
            return result

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, bridge_kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.lost_counter += 1
            return self._hold_last(result, roi=(rx, ry, rw, rh))

        kept = [c for c in contours if cv2.contourArea(c) >= self.config.min_area_px]
        if not kept:
            self.lost_counter += 1
            return self._hold_last(result, roi=(rx, ry, rw, rh))

        kept_shifted = []
        for contour in kept:
            offset = np.array([[[rx, ry]]], dtype=contour.dtype)
            kept_shifted.append(contour + offset)
        pts = np.vstack([c.reshape(-1, 2) for c in kept_shifted]).astype(np.float32)

        try:
            point_on_line, direction_px = ransac_line(pts)
        except ValueError:
            self.lost_counter += 1
            return self._hold_last(result, roi=(rx, ry, rw, rh))

        residuals = np.abs(np.cross(pts - point_on_line, direction_px))
        inliers = residuals <= RANSAC_THRESHOLD_PX
        inlier_ratio = float(np.count_nonzero(inliers)) / float(len(pts))
        total_area_px = sum(cv2.contourArea(c) for c in kept)
        area_ok = total_area_px >= self.config.min_area_px

        projections = pts @ direction_px
        far_end = pts[int(np.argmax(projections))]

        far_end_mm = self.homography.project_px_to_mm(far_end.reshape(1, 2))[0]
        near_point_px = far_end - direction_px * 10.0
        near_point_mm = self.homography.project_px_to_mm(near_point_px.reshape(1, 2))[0]
        axis_mm = normalise(far_end_mm - near_point_mm)
        if np.allclose(axis_mm, 0):
            self.lost_counter += 1
            return self._hold_last(result, roi=(rx, ry, rw, rh))

        angle_ok = angle_consistent(axis_mm, self.prev_line_dir_mm, self.config.angle_gate_deg)
        quality_ok = area_ok and (inlier_ratio >= self.config.min_inlier_ratio) and angle_ok

        if not quality_ok:
            self.lost_counter += 1
            return self._hold_last(result, roi=(rx, ry, rw, rh))

        tip_mm_new = far_end_mm - axis_mm * self.config.tip_offset_mm
        tip_mm_new = tip_mm_new + np.asarray(self.config.manual_offset_mm, dtype=float)

        if self.prev_tip_mm is not None:
            if float(np.linalg.norm(tip_mm_new - self.prev_tip_mm)) > self.config.max_jump_mm:
                self.lost_counter += 1
                return self._hold_last(result, roi=(rx, ry, rw, rh))

        tip_mm_smooth = self.tip_filter.update(tip_mm_new)
        tip_px = self.homography.project_mm_to_px(tip_mm_smooth.reshape(1, 2))[0]

        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect).astype(np.int32)

        self.prev_tip_mm = tip_mm_smooth.copy()
        self.prev_line_dir_mm = axis_mm.copy()
        self.lost_counter = 0

        result.tip_px = tip_px
        result.tip_mm = tip_mm_smooth
        result.far_end_px = far_end
        result.contour = max(kept_shifted, key=cv2.contourArea)
        result.bounding_box = box
        result.roi = (rx, ry, rw, rh)
        self.last_result = result
        return result

    def _search_roi(self, frame_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
        if (
            self.prev_tip_mm is None
            or self.lost_counter >= self.config.lost_reacq_frames
            or not self.homography.ready
        ):
            height, width = frame_shape[0], frame_shape[1]
            return (0, 0, width, height)

        pad_px = int(self.config.roi_pad_mm * 1.0)
        cx, cy = self.homography.project_mm_to_px(self.prev_tip_mm.reshape(1, 2))[0]
        cx = int(cx)
        cy = int(cy)

        anchor = self.prev_tip_mm + np.array([10.0, 0.0])
        a = self.homography.project_mm_to_px(anchor.reshape(1, 2))[0]
        mm2px = max(1.0, float(np.linalg.norm(a - np.array([cx, cy], dtype=float)) / 10.0))
        pad_px = int(self.config.roi_pad_mm * mm2px)

        x1 = max(0, cx - pad_px)
        y1 = max(0, cy - pad_px)
        x2 = min(frame_shape[1], cx + pad_px)
        y2 = min(frame_shape[0], cy + pad_px)
        return (x1, y1, x2 - x1, y2 - y1)

    def _hold_last(
        self, result: StripeDetectionResult, roi: Tuple[int, int, int, int]
    ) -> StripeDetectionResult:
        result.roi = roi
        if self.prev_tip_mm is not None and self.homography.ready:
            tip_px = self.homography.project_mm_to_px(self.prev_tip_mm.reshape(1, 2))[0]
            result.tip_px = tip_px
            result.tip_mm = self.prev_tip_mm.copy()
        self.last_result = result
        return result

    def _create_mask(self, frame: np.ndarray) -> Optional[np.ndarray]:
        palette = self.config.palette.lower()
        if palette in {"green-white", "black-green", "green-black"}:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower = np.array(self.config.hsv_low, dtype=np.uint8)
            upper = np.array(self.config.hsv_high, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            return mask
        if palette in {"black-white", "white-black"}:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if palette.startswith("black"):
                mask = cv2.bitwise_not(mask)
            return mask
        return None

    def _roi_from_tip(
        self,
        frame: np.ndarray,
        tip_px: np.ndarray,
        tip_mm: np.ndarray,
    ) -> Optional[Tuple[int, int, int, int]]:
        if not self.homography.ready:
            return None
        radius_mm = 5.0
        offsets_mm = np.array([
            tip_mm + [-radius_mm, -radius_mm],
            tip_mm + [radius_mm, radius_mm],
        ])
        roi_pts_px = self.homography.project_mm_to_px(offsets_mm)
        size_px = np.mean(np.abs(roi_pts_px[1] - roi_pts_px[0]))
        size = int(max(16, size_px))
        q = 8
        size = (size + q // 2) // q * q
        x = int(tip_px[0] - size)
        y = int(tip_px[1] - size)
        w = int(size * 2)
        h = int(size * 2)
        if x < 0 or y < 0 or x + w >= frame.shape[1] or y + h >= frame.shape[0]:
            return None
        return (x, y, w, h)


# ---------------------------------------------------------------------------
# Rings mode implementation
# ---------------------------------------------------------------------------


@dataclass
class RingsDetectionResult:
    tip_px: Optional[np.ndarray] = None
    tip_mm: Optional[np.ndarray] = None
    ringA_center: Optional[np.ndarray] = None
    ringB_center: Optional[np.ndarray] = None
    roi: Optional[Tuple[int, int, int, int]] = None


class RingsMode:
    def __init__(self, config: RingsConfig, homography: MarkerHomography) -> None:
        self.config = config
        self.homography = homography
        self.tip_filter = ExponentialMovingAverage()
        self.last_result: RingsDetectionResult = RingsDetectionResult()

    def reset(self) -> None:
        self.tip_filter.reset()
        self.last_result = RingsDetectionResult()

    def process(self, frame: np.ndarray) -> RingsDetectionResult:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ringA_mask = cv2.inRange(hsv, np.array(self.config.ringA_low), np.array(self.config.ringA_high))
        ringB_mask = cv2.inRange(hsv, np.array(self.config.ringB_low), np.array(self.config.ringB_high))

        centerA = self._find_largest_centroid(ringA_mask)
        centerB = self._find_largest_centroid(ringB_mask)

        if centerA is None or centerB is None or not self.homography.ready:
            self.reset()
            return RingsDetectionResult()

        axis = centerA - centerB
        axis = normalise(axis)
        if np.allclose(axis, 0):
            self.reset()
            return RingsDetectionResult()

        centerA_mm = self.homography.project_px_to_mm(centerA.reshape(1, 2))[0]
        centerB_mm = self.homography.project_px_to_mm(centerB.reshape(1, 2))[0]
        axis_mm = normalise(centerA_mm - centerB_mm)
        tip_mm = centerA_mm - axis_mm * self.config.tip_distance_mm
        tip_mm = tip_mm + np.asarray(self.config.manual_offset_mm, dtype=float)
        tip_px = self.homography.project_mm_to_px(tip_mm.reshape(1, 2))[0]
        tip_px = self.tip_filter.update(tip_px)

        roi = self._roi_from_tip(frame, tip_px, tip_mm)

        result = RingsDetectionResult(
            tip_px=tip_px,
            tip_mm=tip_mm,
            ringA_center=centerA,
            ringB_center=centerB,
            roi=roi,
        )
        self.last_result = result
        return result

    def _roi_from_tip(
        self,
        frame: np.ndarray,
        tip_px: np.ndarray,
        tip_mm: np.ndarray,
    ) -> Optional[Tuple[int, int, int, int]]:
        if not self.homography.ready:
            return None
        offsets_mm = np.array([
            tip_mm + [-5.0, -5.0],
            tip_mm + [5.0, 5.0],
        ])
        roi_pts_px = self.homography.project_mm_to_px(offsets_mm)
        size_px = np.mean(np.abs(roi_pts_px[1] - roi_pts_px[0]))
        size = int(max(16, size_px))
        q = 8
        size = (size + q // 2) // q * q
        x = int(tip_px[0] - size)
        y = int(tip_px[1] - size)
        w = h = int(size * 2)
        if x < 0 or y < 0 or x + w >= frame.shape[1] or y + h >= frame.shape[0]:
            return None
        return (x, y, w, h)

    @staticmethod
    def _find_largest_centroid(mask: np.ndarray) -> Optional[np.ndarray]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 80:
            return None
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None
        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])
        return np.array([cx, cy], dtype=float)


# ---------------------------------------------------------------------------
# Colour dual-rings mode implementation
# ---------------------------------------------------------------------------


@dataclass
class ColorDualRingsDetectionResult:
    tip_px: Optional[np.ndarray] = None
    tip_mm: Optional[np.ndarray] = None
    cyan_center_px: Optional[np.ndarray] = None
    magenta_center_px: Optional[np.ndarray] = None
    roi: Optional[Tuple[int, int, int, int]] = None


class ColorDualRingsMode:
    def __init__(self, config: ColorDualRingsConfig, homography: MarkerHomography) -> None:
        self.config = config
        self.homography = homography
        self.tip_filter = ExponentialMovingAverage(alpha=self.config.ema_alpha)
        self.prev_tip_mm: Optional[np.ndarray] = None
        self.prev_line_dir_mm: Optional[np.ndarray] = None
        self.lost_counter = 0
        self.last_result: ColorDualRingsDetectionResult = ColorDualRingsDetectionResult()

    def reset(self) -> None:
        self.tip_filter.alpha = float(self.config.ema_alpha)
        self.tip_filter.reset()
        self.prev_tip_mm = None
        self.prev_line_dir_mm = None
        self.lost_counter = 0
        self.last_result = ColorDualRingsDetectionResult()

    def process(self, frame: np.ndarray) -> ColorDualRingsDetectionResult:
        result = ColorDualRingsDetectionResult()
        if not self.homography.ready:
            self.reset()
            return result

        # ראשית: חפש באזור גדול כדי למצוא טבעות
        roi_search = self._search_roi(frame.shape)
        rx, ry, rw, rh = roi_search
        frame_roi = frame[ry : ry + rh, rx : rx + rw]

        hsv = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2HSV)
        cyan_mask = cv2.inRange(hsv, np.array(self.config.cyan_low), np.array(self.config.cyan_high))
        magenta_mask = cv2.inRange(hsv, np.array(self.config.magenta_low), np.array(self.config.magenta_high))

        kernel = np.ones((3, 3), np.uint8)
        cyan_mask = cv2.morphologyEx(cyan_mask, cv2.MORPH_OPEN, kernel)
        magenta_mask = cv2.morphologyEx(magenta_mask, cv2.MORPH_OPEN, kernel)
        cyan_mask = cv2.dilate(cyan_mask, kernel, iterations=1)
        magenta_mask = cv2.dilate(magenta_mask, kernel, iterations=1)

        cyan_center_roi, cyan_area = self._largest_centroid_and_area(cyan_mask)
        magenta_center_roi, magenta_area = self._largest_centroid_and_area(magenta_mask)

        if (
            cyan_center_roi is None
            or magenta_center_roi is None
            or cyan_area < self.config.min_area_px
            or magenta_area < self.config.min_area_px
        ):
            self.lost_counter += 1
            return self._hold_last(result, roi_search)

        cyan_center_px = cyan_center_roi + np.array([rx, ry], dtype=float)
        magenta_center_px = magenta_center_roi + np.array([rx, ry], dtype=float)

        cyan_center_mm = self.homography.project_px_to_mm(cyan_center_px.reshape(1, 2))[0]
        magenta_center_mm = self.homography.project_px_to_mm(magenta_center_px.reshape(1, 2))[0]

        axis_mm = normalise(cyan_center_mm - magenta_center_mm)
        if np.allclose(axis_mm, 0):
            self.lost_counter += 1
            return self._hold_last(result, roi_search)

        if not angle_consistent(axis_mm, self.prev_line_dir_mm, self.config.angle_gate_deg):
            self.lost_counter += 1
            return self._hold_last(result, roi_search)

        normal_mm = np.array([-axis_mm[1], axis_mm[0]], dtype=float)
        if self.config.nib_side.lower() == "right":
            normal_mm = -normal_mm

        tip_mm_new = (
            cyan_center_mm
            - axis_mm * float(self.config.front_to_tip_mm)
            + normal_mm * float(self.config.nib_lateral_mm)
            + np.asarray(self.config.manual_offset_mm, dtype=float)
        )

        if self.prev_tip_mm is not None:
            if float(np.linalg.norm(tip_mm_new - self.prev_tip_mm)) > float(self.config.max_jump_mm):
                self.lost_counter += 1
                return self._hold_last(result, roi_search)

        tip_mm_smooth = self.tip_filter.update(tip_mm_new)
        tip_px = self.homography.project_mm_to_px(tip_mm_smooth.reshape(1, 2))[0]

        self.prev_tip_mm = tip_mm_smooth.copy()
        self.prev_line_dir_mm = axis_mm.copy()
        self.lost_counter = 0

        result.tip_px = tip_px
        result.tip_mm = tip_mm_smooth
        result.cyan_center_px = cyan_center_px
        result.magenta_center_px = magenta_center_px
        # עכשיו: קבע ROI קטן סביב החוד כדי שחיישן הדיו ימדוד רק "דיו" ולא את כל החלון הגדול
        small_roi = self._roi_from_tip(frame, tip_px, tip_mm_smooth)
        result.roi = small_roi if small_roi is not None else roi_search
        self.last_result = result
        return result

    def _roi_from_tip(
        self, frame: np.ndarray, tip_px: np.ndarray, tip_mm: np.ndarray
    ) -> Optional[Tuple[int, int, int, int]]:
        # דומה ל-rings/stripe: חלון קטן סביב החוד, בקירוב ~5 מ"מ
        if not self.homography.ready:
            return None
        radius_mm = 5.0
        offsets_mm = np.array(
            [tip_mm + [-radius_mm, -radius_mm], tip_mm + [radius_mm, radius_mm]],
            dtype=float,
        )
        roi_pts_px = self.homography.project_mm_to_px(offsets_mm)
        size_px = float(np.mean(np.abs(roi_pts_px[1] - roi_pts_px[0])))
        size = int(max(16, size_px))
        q = 8
        size = (size + q // 2) // q * q
        x = int(tip_px[0] - size)
        y = int(tip_px[1] - size)
        w = int(size * 2)
        h = int(size * 2)
        if x < 0 or y < 0 or x + w >= frame.shape[1] or y + h >= frame.shape[0]:
            return None
        return (x, y, w, h)

    def _largest_centroid_and_area(
        self, mask: np.ndarray
    ) -> Tuple[Optional[np.ndarray], float]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0.0
        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        if area <= 0:
            return None, 0.0
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None, area
        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])
        return np.array([cx, cy], dtype=float), area

    def _search_roi(self, frame_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
        if (
            self.prev_tip_mm is None
            or self.lost_counter >= int(self.config.lost_reacq_frames)
            or not self.homography.ready
        ):
            height, width = frame_shape[0], frame_shape[1]
            return (0, 0, width, height)

        anchor = self.prev_tip_mm
        anchor_px = self.homography.project_mm_to_px(anchor.reshape(1, 2))[0]
        mm_anchor = anchor + np.array([10.0, 0.0])
        mm_anchor_px = self.homography.project_mm_to_px(mm_anchor.reshape(1, 2))[0]
        mm_to_px = max(1.0, float(np.linalg.norm(mm_anchor_px - anchor_px) / 10.0))
        pad_px = int(max(16, self.config.roi_pad_mm * mm_to_px))

        cx = int(anchor_px[0])
        cy = int(anchor_px[1])
        x1 = max(0, cx - pad_px)
        y1 = max(0, cy - pad_px)
        x2 = min(frame_shape[1], cx + pad_px)
        y2 = min(frame_shape[0], cy + pad_px)
        return (x1, y1, x2 - x1, y2 - y1)

    def _hold_last(
        self, result: ColorDualRingsDetectionResult, roi: Tuple[int, int, int, int]
    ) -> ColorDualRingsDetectionResult:
        result.roi = roi
        if self.prev_tip_mm is not None and self.homography.ready:
            tip_px = self.homography.project_mm_to_px(self.prev_tip_mm.reshape(1, 2))[0]
            result.tip_px = tip_px
            result.tip_mm = self.prev_tip_mm.copy()
        self.last_result = result
        return result


# ---------------------------------------------------------------------------
# GUI controller
# ---------------------------------------------------------------------------


class StrokeCounterApp:
    def __init__(self, config: AppConfig, mode_override: Optional[str] = None) -> None:
        self.config = config
        if mode_override:
            self.config.mode = mode_override
        self._build_string = get_build_string()
        self.root = tk.Tk()
        self.root.title(f"{APP_NAME} — {self._build_string}")
        self.log = setup_logger()
        self.root.protocol("WM_DELETE_WINDOW", self._on_exit)
        self.root.bind("<KeyPress>", self._on_key_press)
        self.video_label = ttk.Label(self.root)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        self._build_menu()

        self.capture: Optional[cv2.VideoCapture] = None
        self.current_frame: Optional[np.ndarray] = None
        self.running = False
        self.photo_image: Optional[ImageTk.PhotoImage] = None
        self.marker_homography = MarkerHomography(self.config.rails)
        self.stripe_mode = StripeMode(self.config.stripe, self.marker_homography)
        self.rings_mode = RingsMode(self.config.rings, self.marker_homography)
        self.color_dual_mode = ColorDualRingsMode(
            self.config.color_dual_rings, self.marker_homography
        )
        ia = self.config.ink_activity or {}
        self.ink_tracker = InkActivityTracker(
            window_size=int(ia.get("window_size", 24)),
            threshold_on=float(ia.get("threshold_on", 6.0)),
            threshold_off=float(ia.get("threshold_off", 3.0)),
            frames_on=int(ia.get("frames_on", 3)),
            frames_off=int(ia.get("frames_off", 6)),
        )
        self.stroke_count = 0
        self.refine_count = 0
        self.last_tip_mm: Optional[np.ndarray] = None
        self.pen_down = False
        self.last_pen_down_time: Optional[float] = None
        self.last_pen_up_time: Optional[float] = None
        self._frame_lock = threading.Lock()
        self._update_timer: Optional[str] = None
        self._color_dialog_geometry: Optional[str] = None
        self._last_sample_bucket: Optional[int] = None

    def _on_key_press(self, event: tk.Event) -> None:
        keysym = event.keysym.lower()
        if keysym == "r":
            if self.marker_homography.relock():
                self.status_var.set("Homography re-locked")
            else:
                self.status_var.set("Waiting for ArUco markers…")
        elif keysym == "s":
            self._take_screenshot()
        elif keysym == "escape":
            self._on_exit()

    # ------------------------------ GUI setup ------------------------------

    def _build_menu(self) -> None:
        menu_bar = tk.Menu(self.root)

        file_menu = tk.Menu(menu_bar, tearoff=False)
        file_menu.add_command(label="Start", command=self.start)
        file_menu.add_command(label="Stop", command=self.stop)
        file_menu.add_separator()
        file_menu.add_command(label="Screenshot…", command=self._take_screenshot)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_exit)
        menu_bar.add_cascade(label="File", menu=file_menu)

        tools_menu = tk.Menu(menu_bar, tearoff=False)
        tools_menu.add_command(label="Color calibration…", command=self._color_calibration)
        tools_menu.add_command(
            label="Calibrate colors from clicks…", command=self._click_color_calibration
        )
        tools_menu.add_command(label="Select camera…", command=self._select_camera)
        tools_menu.add_command(label="Manual tip set…", command=self._manual_tip_adjust)
        menu_bar.add_cascade(label="Tools", menu=tools_menu)

        print_menu = tk.Menu(menu_bar, tearoff=False)
        print_menu.add_command(label="Print stripe pattern…", command=self._print_stripe)
        print_menu.add_command(label="Print ArUco markers…", command=self._print_aruco)
        print_menu.add_command(label="Print rails…", command=self._print_rails)
        menu_bar.add_cascade(label="Print", menu=print_menu)

        help_menu = tk.Menu(menu_bar, tearoff=False)
        help_menu.add_command(label="About…", command=self._show_about)
        menu_bar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menu_bar)

    # --------------------------- Camera handling --------------------------

    def _open_capture(self, index: int) -> Optional[cv2.VideoCapture]:
        for backend in CAMERA_BACKENDS:
            cap = cv2.VideoCapture(index, backend)
            if cap is not None and cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_FRAME_SIZE[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_FRAME_SIZE[1])
                cap.set(cv2.CAP_PROP_FPS, DEFAULT_FPS)
                return cap
            if cap is not None:
                cap.release()
        return None

    def _auto_select_camera(self) -> Optional[int]:
        for index in CAMERA_PROBE_ORDER:
            cap = self._open_capture(index)
            if cap is not None:
                cap.release()
                return index
        return None

    def start(self) -> None:
        if self.running:
            return
        index = self._auto_select_camera()
        if index is None:
            messagebox.showerror(APP_NAME, "No camera detected.")
            return
        self._start_with_camera(index)

    def _start_with_camera(self, index: int) -> None:
        cap = self._open_capture(index)
        if cap is None:
            messagebox.showerror(APP_NAME, f"Failed to open camera {index}.")
            return
        self.capture = cap
        self.running = True
        self.stroke_count = 0
        self.refine_count = 0
        self.last_tip_mm = None
        self.ink_tracker.clear()
        self.marker_homography.clear()
        self.stripe_mode.reset()
        self.rings_mode.reset()
        self.color_dual_mode.reset()
        self._last_sample_bucket = None
        self.status_var.set(f"Camera {index} active - waiting for ArUco markers...")
        self._schedule_update()
        branch, short_hash, last_update_iso = get_git_info()
        print(
            f"[INFO] {APP_NAME} {self._build_string} — "
            f"branch={branch} commit={short_hash} updated={last_update_iso}"
        )
        self.log.info("app_start | build=%s | mode=%s", self._build_string, self.config.mode)

    def stop(self) -> None:
        self.running = False
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        if self._update_timer is not None:
            self.root.after_cancel(self._update_timer)
            self._update_timer = None
        self.status_var.set("Stopped")

    def _schedule_update(self) -> None:
        if not self.running:
            return
        self._update_frame()
        self._update_timer = self.root.after(10, self._schedule_update)

    def _update_frame(self) -> None:
        if not self.running or self.capture is None:
            return
        ret, frame = self.capture.read()
        if not ret:
            self.status_var.set("Frame grab failed")
            return
        with self._frame_lock:
            self.current_frame = frame.copy()
        self._process_frame(frame)

    # -------------------------- Frame processing -------------------------

    def _process_frame(self, frame: np.ndarray) -> None:
        was_locked = self.marker_homography.is_locked
        corners, ids = self.marker_homography.update(frame)
        if not was_locked and self.marker_homography.is_locked:
            self.status_var.set("Work area locked")
            ids_list = ids.flatten().tolist() if ids is not None else []
            self.log.info("aruco_lock | ids=%s", ids_list)
        elif was_locked and not self.marker_homography.is_locked:
            self.log.info("aruco_lost")
        mode = self.config.mode
        if mode == "stripe":
            result = self.stripe_mode.process(frame)
            tip_px = result.tip_px
        elif mode == "rings":
            result = self.rings_mode.process(frame)
            tip_px = result.tip_px
        else:
            result = self.color_dual_mode.process(frame)
            tip_px = result.tip_px

        roi_gray = None
        roi_for_log: Optional[Tuple[int, int, int, int]] = None

        roi_rect = result.roi
        if tip_px is not None:
            ia = self.config.ink_activity or {}
            side_px = int(ia.get("roi_side_px", 64))
            cx, cy = int(tip_px[0]), int(tip_px[1])
            x = max(0, cx - side_px // 2)
            y = max(0, cy - side_px // 2)
            x2 = min(frame.shape[1], x + side_px)
            y2 = min(frame.shape[0], y + side_px)
            w, h = x2 - x, y2 - y
            roi_rect = (x, y, w, h)

        if roi_rect is not None:
            x, y, w, h = roi_rect
            if w > 0 and h > 0:
                roi = frame[y : y + h, x : x + w]
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi_for_log = (x, y, w, h)
            else:
                roi_rect = None

        pen_down, activity = self.ink_tracker.update(roi_gray)

        if pen_down != self.pen_down:
            self.log.info(
                "pen_state | down=%s | I=%.2f | S=%d | R=%d | roi=%s",
                pen_down,
                activity,
                self.stroke_count,
                self.refine_count,
                roi_for_log,
            )

        if pen_down and not self.pen_down:
            self.stroke_count += 1
            self.last_pen_down_time = time.time()
        elif not pen_down and self.pen_down:
            self.last_pen_up_time = time.time()
        self.pen_down = pen_down

        bucket = int(time.time() * 10)
        if bucket != self._last_sample_bucket:
            self._last_sample_bucket = bucket
            self.log.info(
                "sample | I=%.2f | S=%d | R=%d | mode=%s",
                activity,
                self.stroke_count,
                self.refine_count,
                self.config.mode,
            )

        if pen_down and tip_px is not None:
            if self.last_tip_mm is not None and result.tip_mm is not None:
                dist = np.linalg.norm(result.tip_mm - self.last_tip_mm)
                if dist >= 0.3:
                    self.refine_count += 1
            if result.tip_mm is not None:
                self.last_tip_mm = result.tip_mm.copy()
        elif result.tip_mm is not None:
            self.last_tip_mm = result.tip_mm.copy()

        overlay = frame.copy()
        if corners and ids is not None:
            cv2.aruco.drawDetectedMarkers(overlay, corners, ids)
        if isinstance(result, StripeDetectionResult) and result.contour is not None:
            cv2.drawContours(overlay, [result.contour], -1, (0, 255, 0), 2)
        if isinstance(result, StripeDetectionResult) and result.bounding_box is not None:
            cv2.polylines(overlay, [result.bounding_box], True, (255, 0, 0), 2)
        if isinstance(result, RingsDetectionResult):
            if result.ringA_center is not None:
                cv2.circle(overlay, tuple(int(v) for v in result.ringA_center), 8, (0, 255, 255), 2)
            if result.ringB_center is not None:
                cv2.circle(overlay, tuple(int(v) for v in result.ringB_center), 8, (255, 255, 0), 2)
        if isinstance(result, ColorDualRingsDetectionResult):
            if result.cyan_center_px is not None:
                cv2.circle(
                    overlay,
                    tuple(int(v) for v in result.cyan_center_px),
                    8,
                    (255, 255, 0),
                    2,
                )
            if result.magenta_center_px is not None:
                cv2.circle(
                    overlay,
                    tuple(int(v) for v in result.magenta_center_px),
                    8,
                    (255, 0, 255),
                    2,
                )
        if tip_px is not None:
            cv2.circle(overlay, tuple(int(v) for v in tip_px), 6, (0, 0, 255), -1)
        if roi_rect is not None:
            x, y, w, h = roi_rect
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 255, 0), 2)

        # Compact, scalable HUD (prevents clipping on small windows)
        h, w = overlay.shape[:2]
        wm_scale = max(0.5, min(0.8, w / 1600.0))
        cv2.putText(
            overlay,
            self._build_string,
            (14, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            wm_scale,
            (200, 200, 200),
            2,
        )

        font_scale = 0.7 * (w / 1280.0)
        font_scale = max(0.5, min(font_scale, 0.9))  # clamp for typical laptops

        pd_flag = "  PEN DOWN" if self.pen_down else ""
        status_text = f"{self.config.mode}  S:{self.stroke_count}  R:{self.refine_count}  I:{activity:.1f}{pd_flag}"
        cv2.putText(
            overlay,
            status_text,
            (14, int(36 * font_scale / 0.7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            2,
        )

        help_text = "Keys: R=re-lock  S=screenshot  ESC=quit"
        cv2.putText(
            overlay,
            help_text,
            (14, h - int(14 * font_scale / 0.7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * 0.85,
            (180, 255, 180),
            2,
        )

        hud_scale = max(0.6, min(0.9, w / 1600.0))
        cv2.putText(
            overlay,
            f"I:{activity:.1f}",
            (14, int(58 * hud_scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            hud_scale * 0.85,
            (255, 255, 255),
            2,
        )
        if self.pen_down:
            cv2.putText(
                overlay,
                "PEN DOWN",
                (w // 2 - 90, int(32 * hud_scale)),
                cv2.FONT_HERSHEY_SIMPLEX,
                hud_scale,
                (0, 255, 255),
                2,
            )

        rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        self.photo_image = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=self.photo_image)

    # ------------------------------- Menus --------------------------------

    def _take_screenshot(self) -> None:
        if self.current_frame is None:
            messagebox.showinfo(APP_NAME, "No frame to save yet.")
            return
        filename = filedialog.asksaveasfilename(
            title="Save screenshot",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All files", "*.*")],
        )
        if not filename:
            return
        cv2.imwrite(filename, self.current_frame)
        self.status_var.set(f"Saved screenshot to {filename}")

    def _click_color_calibration(self) -> None:
        if self.current_frame is None:
            messagebox.showinfo(APP_NAME, "No frame available yet.")
            return
        if not self.marker_homography.ready:
            messagebox.showinfo(APP_NAME, "Lock the work area first (press R).")
            return

        frame = self.current_frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(image=image)

        dialog = tk.Toplevel(self.root)
        dialog.title("Dual-ring colour calibration")
        dialog.transient(self.root)
        dialog.grab_set()

        instructions = tk.StringVar(value="Click the CYAN ring")
        ttk.Label(dialog, textvariable=instructions).pack(padx=10, pady=(10, 4))

        canvas = tk.Canvas(dialog, width=image.width, height=image.height)
        canvas.pack()
        canvas_image = canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas.image = photo  # keep reference

        bounds_preview: Optional[int] = None
        patch_size = 15

        def bounds(arr: np.ndarray, low_pad: int, high_pad: int, lo_cap: int, hi_cap: int) -> Tuple[int, int]:
            lo = int(max(lo_cap, np.percentile(arr, 10) - low_pad))
            hi = int(min(hi_cap, np.percentile(arr, 90) + high_pad))
            return lo, hi

        def derive_hsv(x: int, y: int) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
            half = patch_size // 2
            x1 = max(0, x - half)
            x2 = min(frame.shape[1], x + half + 1)
            y1 = max(0, y - half)
            y2 = min(frame.shape[0], y + half + 1)
            patch = frame[y1:y2, x1:x2]
            if patch.size == 0:
                raise ValueError("Patch out of bounds")
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            h_channel = hsv[:, :, 0].ravel()
            s_channel = hsv[:, :, 1].ravel()
            v_channel = hsv[:, :, 2].ravel()
            h_low, h_high = bounds(h_channel, 5, 5, 0, 179)
            s_low, s_high = bounds(s_channel, 30, 30, 0, 255)
            v_low, v_high = bounds(v_channel, 30, 30, 0, 255)
            return (h_low, s_low, v_low), (h_high, s_high, v_high)

        result_cyan: Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = None
        result_magenta: Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = None

        cyan_label = tk.StringVar(value="Cyan HSV: –")
        magenta_label = tk.StringVar(value="Magenta HSV: –")

        ttk.Label(dialog, textvariable=cyan_label).pack(padx=10, pady=(6, 0))
        ttk.Label(dialog, textvariable=magenta_label).pack(padx=10, pady=(0, 6))

        def update_preview(x: int, y: int) -> None:
            nonlocal bounds_preview
            if bounds_preview is not None:
                canvas.delete(bounds_preview)
            half = patch_size // 2
            bounds_preview = canvas.create_rectangle(
                x - half,
                y - half,
                x + half,
                y + half,
                outline="#00ffff" if instructions.get().startswith("Click the CYAN") else "#ff00ff",
                width=2,
            )

        step = "cyan"

        def on_click(event: tk.Event) -> str:
            nonlocal result_cyan, result_magenta, step
            x = int(np.clip(event.x, 0, frame.shape[1] - 1))
            y = int(np.clip(event.y, 0, frame.shape[0] - 1))
            try:
                low, high = derive_hsv(x, y)
            except ValueError:
                return "break"
            update_preview(x, y)
            if step == "cyan":
                result_cyan = (low, high)
                cyan_label.set(
                    "Cyan HSV: low={} high={}".format(
                        list(low),
                        list(high),
                    )
                )
                instructions.set("Click the MAGENTA ring")
                step = "magenta"
            else:
                result_magenta = (low, high)
                magenta_label.set(
                    "Magenta HSV: low={} high={}".format(
                        list(low),
                        list(high),
                    )
                )
                instructions.set("Review ranges and press Save")
            return "break"

        canvas.bind("<ButtonPress-1>", on_click)
        canvas.tag_bind(canvas_image, "<ButtonPress-1>", on_click)

        buttons = ttk.Frame(dialog)
        buttons.pack(pady=8)

        def close_dialog() -> None:
            try:
                dialog.grab_release()
            except tk.TclError:
                pass
            dialog.destroy()

        def on_save() -> None:
            if result_cyan is None or result_magenta is None:
                messagebox.showinfo(APP_NAME, "Click both rings before saving.")
                return
            cfg = self.config.color_dual_rings
            cfg.cyan_low = tuple(int(v) for v in result_cyan[0])
            cfg.cyan_high = tuple(int(v) for v in result_cyan[1])
            cfg.magenta_low = tuple(int(v) for v in result_magenta[0])
            cfg.magenta_high = tuple(int(v) for v in result_magenta[1])
            self.config.dump(CONFIG_FILE)
            self.color_dual_mode.reset()
            self.last_tip_mm = None
            self.status_var.set("Updated cyan/magenta HSV ranges from clicks")
            close_dialog()

        ttk.Button(buttons, text="Save", command=on_save).grid(row=0, column=0, padx=6)
        ttk.Button(buttons, text="Cancel", command=close_dialog).grid(row=0, column=1, padx=6)

        dialog.bind("<Escape>", lambda event: (close_dialog(), "break"))
        dialog.protocol("WM_DELETE_WINDOW", close_dialog)

    def _color_calibration(self) -> None:
        class ScrollableFrame(ttk.Frame):
            def __init__(self, parent: tk.Widget):
                super().__init__(parent)
                self.canvas = tk.Canvas(self, highlightthickness=0)
                self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
                self.inner = ttk.Frame(self.canvas)
                self._inner_window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
                self.canvas.configure(yscrollcommand=self.vsb.set)
                self.canvas.grid(row=0, column=0, sticky="nsew")
                self.vsb.grid(row=0, column=1, sticky="ns")
                self.grid_rowconfigure(0, weight=1)
                self.grid_columnconfigure(0, weight=1)

                self.inner.bind(
                    "<Configure>",
                    lambda _: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
                )
                self.canvas.bind(
                    "<Configure>",
                    lambda event: self.canvas.itemconfigure(self._inner_window, width=event.width),
                )

                def _on_wheel(event: tk.Event) -> None:
                    if getattr(event, "delta", 0):
                        delta = -1 * int(event.delta / 120) if abs(event.delta) >= 1 else 0
                        if delta == 0:
                            delta = -1 if event.delta > 0 else 1
                    else:
                        delta = 1 if getattr(event, "num", 0) == 5 else -1
                    self.canvas.yview_scroll(delta, "units")

                self._bindings = []
                for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                    self.canvas.bind_all(sequence, _on_wheel, add=True)
                    self._bindings.append(sequence)

            def cleanup(self) -> None:
                for sequence in self._bindings:
                    self.canvas.unbind_all(sequence)

            def destroy(self) -> None:  # type: ignore[override]
                self.cleanup()
                super().destroy()

        dialog = tk.Toplevel(self.root)
        dialog.title("Colour calibration")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(True, True)

        screen_height = dialog.winfo_screenheight()
        screen_width = dialog.winfo_screenwidth()
        max_height = int(screen_height * 0.8)
        max_width = min(760, int(screen_width * 0.9))
        dialog.geometry(f"{max_width}x{max_height}")
        if self._color_dialog_geometry:
            dialog.geometry(self._color_dialog_geometry)

        content = ScrollableFrame(dialog)
        content.grid(row=0, column=0, sticky="nsew")
        button_bar = ttk.Frame(dialog)
        button_bar.grid(row=1, column=0, sticky="ew")
        dialog.grid_rowconfigure(0, weight=1)
        dialog.grid_columnconfigure(0, weight=1)

        labels = ["H", "S", "V"]
        scale_length = 240 if screen_width < 1366 else 300
        columns = 2 if screen_width >= 1366 else 1
        for col in range(columns):
            content.inner.grid_columnconfigure(col, weight=1)

        next_row = 0
        next_col = 0

        def place_group(frame: ttk.LabelFrame) -> None:
            nonlocal next_row, next_col
            frame.grid(row=next_row, column=next_col, sticky="nsew", padx=10, pady=6)
            next_col += 1
            if next_col >= columns:
                next_col = 0
                next_row += 1

        def create_scale_group(
            parent: ttk.Frame,
            title: str,
            low: Sequence[int],
            high: Sequence[int],
        ) -> Tuple[List[tk.Scale], List[tk.Scale]]:
            frame = ttk.LabelFrame(parent, text=title)
            frame.grid_columnconfigure(0, weight=1)
            low_scales: List[tk.Scale] = []
            high_scales: List[tk.Scale] = []
            for idx, label in enumerate(labels):
                scale = tk.Scale(
                    frame,
                    from_=0,
                    to=255,
                    orient=tk.HORIZONTAL,
                    label=f"Low {label}",
                    length=scale_length,
                )
                scale.set(int(low[idx]))
                scale.grid(row=idx, column=0, padx=6, pady=3, sticky="ew")
                low_scales.append(scale)
            for idx, label in enumerate(labels):
                scale = tk.Scale(
                    frame,
                    from_=0,
                    to=255,
                    orient=tk.HORIZONTAL,
                    label=f"High {label}",
                    length=scale_length,
                )
                scale.set(int(high[idx]))
                scale.grid(row=idx + 3, column=0, padx=6, pady=3, sticky="ew")
                high_scales.append(scale)
            place_group(frame)
            return low_scales, high_scales

        def close_dialog() -> None:
            self._color_dialog_geometry = dialog.winfo_geometry()
            content.cleanup()
            try:
                dialog.grab_release()
            except tk.TclError:
                pass
            dialog.destroy()

        def cancel() -> None:
            close_dialog()

        save_handler: Optional[Callable[[], None]] = None

        mode = self.config.mode
        if mode == "stripe":
            low_scales, high_scales = create_scale_group(
                content.inner,
                "Stripe HSV range",
                self.config.stripe.hsv_low,
                self.config.stripe.hsv_high,
            )

            def save_handler() -> None:
                new_low = [scale.get() for scale in low_scales]
                new_high = [scale.get() for scale in high_scales]
                self.config.stripe.hsv_low = tuple(new_low)
                self.config.stripe.hsv_high = tuple(new_high)
                self.config.dump(CONFIG_FILE)

        elif mode == "rings":
            ringA_low_scales, ringA_high_scales = create_scale_group(
                content.inner,
                "Ring A HSV range",
                self.config.rings.ringA_low,
                self.config.rings.ringA_high,
            )
            ringB_low_scales, ringB_high_scales = create_scale_group(
                content.inner,
                "Ring B HSV range",
                self.config.rings.ringB_low,
                self.config.rings.ringB_high,
            )

            def save_handler() -> None:
                self.config.rings.ringA_low = tuple(scale.get() for scale in ringA_low_scales)
                self.config.rings.ringA_high = tuple(scale.get() for scale in ringA_high_scales)
                self.config.rings.ringB_low = tuple(scale.get() for scale in ringB_low_scales)
                self.config.rings.ringB_high = tuple(scale.get() for scale in ringB_high_scales)
                self.config.dump(CONFIG_FILE)

        else:
            cyan_low_scales, cyan_high_scales = create_scale_group(
                content.inner,
                "Cyan band HSV range",
                self.config.color_dual_rings.cyan_low,
                self.config.color_dual_rings.cyan_high,
            )
            magenta_low_scales, magenta_high_scales = create_scale_group(
                content.inner,
                "Magenta band HSV range",
                self.config.color_dual_rings.magenta_low,
                self.config.color_dual_rings.magenta_high,
            )

            def save_handler() -> None:
                self.config.color_dual_rings.cyan_low = tuple(
                    scale.get() for scale in cyan_low_scales
                )
                self.config.color_dual_rings.cyan_high = tuple(
                    scale.get() for scale in cyan_high_scales
                )
                self.config.color_dual_rings.magenta_low = tuple(
                    scale.get() for scale in magenta_low_scales
                )
                self.config.color_dual_rings.magenta_high = tuple(
                    scale.get() for scale in magenta_high_scales
                )
                self.config.dump(CONFIG_FILE)

        def on_save() -> None:
            if save_handler is not None:
                save_handler()
            close_dialog()

        def bind_scroll(sequence: str, amount: int, units: str = "units") -> None:
            def handler(event: tk.Event) -> Optional[str]:
                if isinstance(event.widget, tk.Scale):
                    return None
                content.canvas.yview_scroll(amount, units)
                return "break"

            dialog.bind(sequence, handler)

        bind_scroll("<Down>", 1)
        bind_scroll("<Up>", -1)
        bind_scroll("<Next>", 1, "pages")
        bind_scroll("<Prior>", -1, "pages")

        dialog.bind("<Escape>", lambda event: (cancel(), "break"))
        dialog.bind("<Control-s>", lambda event: (on_save(), "break"))
        dialog.protocol("WM_DELETE_WINDOW", cancel)

        button_bar.columnconfigure(0, weight=1)
        ttk.Button(button_bar, text="Save", command=on_save).pack(side="right", padx=8, pady=8)
        ttk.Button(button_bar, text="Cancel", command=cancel).pack(side="right", padx=4, pady=8)

    def _select_camera(self) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title("Select camera")
        dialog.grab_set()
        ttk.Label(dialog, text="Detected cameras").pack(padx=10, pady=5)
        listbox = tk.Listbox(dialog, width=24)
        listbox.pack(padx=10, pady=5)

        available_indices: List[int] = []
        for index in range(0, 8):
            cap = self._open_capture(index)
            if cap is not None:
                available_indices.append(index)
                listbox.insert(tk.END, f"Camera {index}")
                cap.release()
        if not available_indices:
            listbox.insert(tk.END, "No cameras detected")
            listbox.configure(state=tk.DISABLED)

        def on_select() -> None:
            if not listbox.curselection():
                return
            idx = listbox.curselection()[0]
            camera_index = available_indices[idx]
            dialog.destroy()
            self.stop()
            self._start_with_camera(camera_index)

        ttk.Button(dialog, text="Use camera", command=on_select).pack(padx=10, pady=10)

    def _manual_tip_adjust(self) -> None:
        if self.current_frame is None:
            messagebox.showinfo(APP_NAME, "No frame available yet.")
            return
        if not self.marker_homography.ready:
            messagebox.showinfo(APP_NAME, "Lock the work area first (press R).")
            return

        mode = self.config.mode
        if mode == "stripe":
            result = self.stripe_mode.last_result
            mode_config = self.config.stripe
            mode_reset = self.stripe_mode.reset
        elif mode == "rings":
            result = self.rings_mode.last_result
            mode_config = self.config.rings
            mode_reset = self.rings_mode.reset
        else:
            result = self.color_dual_mode.last_result
            mode_config = self.config.color_dual_rings
            mode_reset = self.color_dual_mode.reset

        if result.tip_px is None or result.tip_mm is None:
            messagebox.showinfo(APP_NAME, "Tip not detected yet.")
            return

        frame = self.current_frame.copy()
        overlay = frame.copy()
        cv2.circle(overlay, tuple(int(v) for v in result.tip_px), 8, (0, 0, 255), -1)
        rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(image=image)

        dialog = tk.Toplevel(self.root)
        dialog.title("Manual tip alignment")
        dialog.transient(self.root)
        dialog.grab_set()

        canvas = tk.Canvas(dialog, width=image.width, height=image.height)
        canvas.pack()
        canvas_image = canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas.image = photo  # keep reference

        base_tip_px = np.array(result.tip_px, dtype=float)
        base_tip_mm = np.array(result.tip_mm, dtype=float)
        marker_radius = 8
        marker = canvas.create_oval(
            base_tip_px[0] - marker_radius,
            base_tip_px[1] - marker_radius,
            base_tip_px[0] + marker_radius,
            base_tip_px[1] + marker_radius,
            outline="red",
            fill="#ff0000",
            width=3,
        )

        dx_var = tk.DoubleVar(value=0.0)
        dy_var = tk.DoubleVar(value=0.0)
        offset_label = tk.StringVar()

        def update_marker_from_mm() -> None:
            tip_mm_live = base_tip_mm + np.array([dx_var.get(), dy_var.get()], dtype=float)
            tip_px_live = self.marker_homography.project_mm_to_px(tip_mm_live.reshape(1, 2))[0]
            canvas.coords(
                marker,
                tip_px_live[0] - marker_radius,
                tip_px_live[1] - marker_radius,
                tip_px_live[0] + marker_radius,
                tip_px_live[1] + marker_radius,
            )
            offset_label.set(f"dx={dx_var.get():+.2f} mm  dy={dy_var.get():+.2f} mm")

        def set_offsets_from_px(px: float, py: float) -> None:
            px = float(np.clip(px, 0, image.width - 1))
            py = float(np.clip(py, 0, image.height - 1))
            tip_mm_live = self.marker_homography.project_px_to_mm(
                np.array([[px, py]], dtype=float)
            )[0]
            delta = tip_mm_live - base_tip_mm
            dx_var.set(float(np.round(delta[0], 3)))
            dy_var.set(float(np.round(delta[1], 3)))

        def on_click(event: tk.Event) -> str:
            canvas.focus_set()
            set_offsets_from_px(event.x, event.y)
            return "break"

        def on_drag(event: tk.Event) -> str:
            set_offsets_from_px(event.x, event.y)
            return "break"

        def nudge_marker(dx_px: float, dy_px: float) -> None:
            tip_mm_live = base_tip_mm + np.array([dx_var.get(), dy_var.get()], dtype=float)
            tip_px_live = self.marker_homography.project_mm_to_px(
                tip_mm_live.reshape(1, 2)
            )[0]
            tip_px_live[0] = float(np.clip(tip_px_live[0] + dx_px, 0, image.width - 1))
            tip_px_live[1] = float(np.clip(tip_px_live[1] + dy_px, 0, image.height - 1))
            tip_mm_live = self.marker_homography.project_px_to_mm(
                tip_px_live.reshape(1, 2)
            )[0]
            delta = tip_mm_live - base_tip_mm
            dx_var.set(float(np.round(delta[0], 3)))
            dy_var.set(float(np.round(delta[1], 3)))

        def on_key(event: tk.Event) -> None:
            step_px = 2.0
            if event.state & 0x0001:
                step_px *= 5.0
            if event.keysym == "Left":
                nudge_marker(-step_px, 0.0)
            elif event.keysym == "Right":
                nudge_marker(step_px, 0.0)
            elif event.keysym == "Up":
                nudge_marker(0.0, -step_px)
            elif event.keysym == "Down":
                nudge_marker(0.0, step_px)
            else:
                return
            return "break"

        canvas.bind("<ButtonPress-1>", on_click)
        canvas.bind("<B1-Motion>", on_drag)
        canvas.tag_bind(canvas_image, "<ButtonPress-1>", on_click)
        canvas.tag_bind(canvas_image, "<B1-Motion>", on_drag)
        canvas.bind("<Key>", on_key)
        dialog.bind("<Key>", on_key)
        canvas.focus_set()

        controls = ttk.Frame(dialog)
        controls.pack(fill=tk.X, padx=10, pady=6)

        ttk.Label(controls, text="dx (mm)").grid(row=0, column=0, padx=4, pady=4)
        dx_spin = ttk.Spinbox(
            controls,
            from_=-50.0,
            to=50.0,
            increment=0.1,
            textvariable=dx_var,
            width=8,
        )
        dx_spin.grid(row=0, column=1, padx=4, pady=4)

        ttk.Label(controls, text="dy (mm)").grid(row=0, column=2, padx=4, pady=4)
        dy_spin = ttk.Spinbox(
            controls,
            from_=-50.0,
            to=50.0,
            increment=0.1,
            textvariable=dy_var,
            width=8,
        )
        dy_spin.grid(row=0, column=3, padx=4, pady=4)

        ttk.Label(controls, textvariable=offset_label).grid(
            row=1, column=0, columnspan=4, padx=4, pady=(0, 4)
        )

        def on_spinbox_change(*_: object) -> None:
            update_marker_from_mm()

        dx_var.trace_add("write", on_spinbox_change)
        dy_var.trace_add("write", on_spinbox_change)

        update_marker_from_mm()

        buttons = ttk.Frame(dialog)
        buttons.pack(pady=8)

        def save() -> None:
            delta_mm = np.array([dx_var.get(), dy_var.get()], dtype=float)
            if float(np.linalg.norm(delta_mm)) < 1e-6:
                dialog.destroy()
                return
            updated = tuple((np.asarray(mode_config.manual_offset_mm) + delta_mm).tolist())
            mode_config.manual_offset_mm = updated
            self.config.dump(CONFIG_FILE)
            mode_reset()
            self.last_tip_mm = None
            dialog.destroy()

        def auto_derive_offsets() -> None:
            if mode != "color_dual_rings":
                messagebox.showinfo(
                    APP_NAME,
                    "Auto-derived offsets are only available in the color dual-rings mode.",
                )
                return
            axis = self.color_dual_mode.prev_line_dir_mm
            if axis is None or float(np.linalg.norm(axis)) <= 1e-6:
                messagebox.showinfo(APP_NAME, "Tip direction unavailable; move the pen and try again.")
                return
            axis = normalise(axis)
            tip_mm_live = base_tip_mm + np.array([dx_var.get(), dy_var.get()], dtype=float)
            delta = tip_mm_live - base_tip_mm
            normal_vec = np.array([-axis[1], axis[0]], dtype=float)
            parallel = float(np.dot(delta, axis))
            lateral_signed = float(np.dot(delta, normal_vec))
            cfg = self.config.color_dual_rings
            cfg.front_to_tip_mm = max(0.0, float(cfg.front_to_tip_mm - parallel))
            cfg.nib_lateral_mm = abs(lateral_signed)
            cfg.nib_side = "left" if lateral_signed >= 0 else "right"
            cfg.manual_offset_mm = (0.0, 0.0)
            self.config.dump(CONFIG_FILE)
            mode_reset()
            self.last_tip_mm = None
            self.status_var.set("Derived nib offsets from manual alignment")
            dialog.destroy()

        ttk.Button(buttons, text="Save", command=save).grid(row=0, column=0, padx=6)
        ttk.Button(buttons, text="Auto-derive nib offsets", command=auto_derive_offsets).grid(
            row=0, column=1, padx=6
        )
        ttk.Button(buttons, text="Cancel", command=dialog.destroy).grid(row=0, column=2, padx=6)


    def _print_stripe(self) -> None:
        try:
            create_stripe_pdf(self.config.stripe)
            messagebox.showinfo(APP_NAME, "Stripe pattern PDF generated.")
        except Exception as exc:  # pragma: no cover
            messagebox.showerror(APP_NAME, f"Failed to generate PDF: {exc}")

    def _print_aruco(self) -> None:
        try:
            create_aruco_pdf()
            messagebox.showinfo(APP_NAME, "ArUco markers PDF generated.")
        except Exception as exc:  # pragma: no cover
            messagebox.showerror(APP_NAME, f"Failed to generate PDF: {exc}")

    def _print_rails(self) -> None:
        try:
            create_rails_pdf(
                self.config.rails.top_ids,
                self.config.rails.side_ids,
                self.config.rails.marker_size_mm,
                self.config.rails.gap_h_mm,
                self.config.rails.gap_v_mm,
                self.config.rails.aruco_dictionary,
            )
            messagebox.showinfo(APP_NAME, "Rails PDF generated.")
        except Exception as exc:  # pragma: no cover
            messagebox.showerror(APP_NAME, f"Failed to generate PDF: {exc}")

    def _show_about(self) -> None:
        branch, short_hash, last_update_iso = get_git_info()
        messagebox.showinfo(
            APP_NAME,
            f"Sofrim Stroke Counter\n"
            f"Build: {self._build_string}\n"
            f"Branch: {branch}\n"
            f"Commit: {short_hash}\n"
            f"Updated: {last_update_iso}\n\n"
            "Tracks a Torah scribe's quill to estimate writing strokes.\n"
            "Modes: Stripe, Rings, or Color dual-rings.\n"
            "Python 3.12 / OpenCV / Tkinter",
        )

    # ------------------------------- Exit ---------------------------------

    def _on_exit(self) -> None:
        self.stop()
        self.config.dump(CONFIG_FILE)
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sofrim stroke counter")
    parser.add_argument(
        "--mode",
        choices=["stripe", "rings", "color_dual_rings"],
        help="Override detection mode",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print version information and exit",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if getattr(args, "version", False):
        print(get_build_string())
        return
    config = AppConfig.load(CONFIG_FILE)
    app = StrokeCounterApp(config, mode_override=getattr(args, "mode", None))
    app.run()


if __name__ == "__main__":  # pragma: no cover
    main()
