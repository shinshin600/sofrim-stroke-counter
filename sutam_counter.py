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
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except Exception as exc:  # pragma: no cover - Tkinter should always be present
    raise RuntimeError("Tkinter is required to run the GUI") from exc

from print_aruco import create_pdf as create_aruco_pdf
from print_stripe import create_pdf as create_stripe_pdf


if not hasattr(cv2, "aruco"):
    raise ImportError(
        "cv2.aruco is unavailable. Install the 'opencv-contrib-python' package to "
        "enable ArUco marker support."
    )


APP_NAME = "Sofrim Stroke Counter"
CONFIG_FILE = Path("config.json")
DEFAULT_FRAME_SIZE = (1920, 1080)
DEFAULT_FPS = 30
CAMERA_PROBE_ORDER = [1, 2, 0, 3, 4]
CAMERA_BACKENDS = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
MARKER_IDS = list(range(10, 18))
MARKER_SIZE_MM = 10.0
MARKER_HORIZONTAL_GAP_MM = 12.0
MARKER_VERTICAL_GAP_MM = 18.0
MARKER_COLUMNS = 4
MARKER_ROWS = 2
ARUCO_DICTIONARY_ID = cv2.aruco.DICT_4X4_1000
REQUIRED_LOCK_MARKERS = {10, 11, 12, 13}
RANSAC_ITERATIONS = 200
RANSAC_THRESHOLD_PX = 2.5
EMA_ALPHA = 0.25


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
    ui: UIConfig = field(default_factory=UIConfig)

    @staticmethod
    def load(path: Path) -> "AppConfig":
        if not path.exists():
            return AppConfig()
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return AppConfig(
            mode=str(data.get("mode", "stripe")),
            stripe=StripeConfig.from_mapping(data.get("stripe", {})),
            rings=RingsConfig.from_mapping(data.get("rings", {})),
            ui=UIConfig.from_mapping(data.get("ui", {})),
        )

    def dump(self, path: Path) -> None:
        payload = {
            "mode": self.mode,
            "stripe": self.stripe.to_mapping(),
            "rings": self.rings.to_mapping(),
            "ui": self.ui.to_mapping(),
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
    """Maintains a smoothed ink activity signal and pen state."""

    def __init__(self, window_size: int = 24, threshold_on: float = 14.0, threshold_off: float = 7.0):
        self.window_size = int(window_size)
        self.threshold_on = float(threshold_on)
        self.threshold_off = float(threshold_off)
        self.queue: List[float] = []
        self.prev_roi: Optional[np.ndarray] = None
        self.pen_down = False

    def clear(self) -> None:
        self.queue.clear()
        self.prev_roi = None
        self.pen_down = False

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

        if self.pen_down:
            if smoothed < self.threshold_off:
                self.pen_down = False
        else:
            if smoothed > self.threshold_on:
                self.pen_down = True

        return self.pen_down, smoothed


# ---------------------------------------------------------------------------
# ArUco marker homography logic
# ---------------------------------------------------------------------------


class MarkerHomography:
    """Handles detection of reference markers and conversion between units."""

    def __init__(self) -> None:
        self.dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY_ID)
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
        detected_ids = {int(marker_id) for marker_id in ids_flat}
        if not REQUIRED_LOCK_MARKERS.issubset(detected_ids):
            return False

        image_pts: List[np.ndarray] = []
        world_pts: List[np.ndarray] = []

        for idx, marker_id in enumerate(ids_flat):
            marker_int = int(marker_id)
            if marker_int not in MARKER_IDS:
                continue
            c = corners[idx].reshape(-1, 2)
            image_pts.extend(c)
            world_pts.extend(self._world_corners(marker_int))

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

    def _world_corners(self, marker_id: int) -> List[List[float]]:
        position = MARKER_IDS.index(marker_id)
        row = position // MARKER_COLUMNS
        col = position % MARKER_COLUMNS
        step_x = MARKER_SIZE_MM + MARKER_HORIZONTAL_GAP_MM
        step_y = MARKER_SIZE_MM + MARKER_VERTICAL_GAP_MM
        origin_x = col * step_x
        origin_y = row * step_y
        return [
            [origin_x, origin_y],
            [origin_x + MARKER_SIZE_MM, origin_y],
            [origin_x + MARKER_SIZE_MM, origin_y + MARKER_SIZE_MM],
            [origin_x, origin_y + MARKER_SIZE_MM],
        ]

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
        self.tip_filter = ExponentialMovingAverage()
        self.last_result: StripeDetectionResult = StripeDetectionResult()

    def reset(self) -> None:
        self.tip_filter.reset()
        self.last_result = StripeDetectionResult()

    def process(self, frame: np.ndarray) -> StripeDetectionResult:
        result = StripeDetectionResult()
        mask = self._create_mask(frame)
        if mask is None:
            self.reset()
            return result

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.reset()
            return result

        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < self.config.min_area_px:
            self.reset()
            return result

        contour_pts = contour.reshape(-1, 2).astype(np.float32)
        try:
            point_on_line, direction = ransac_line(contour_pts)
        except ValueError:
            self.reset()
            return result

        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect).astype(np.int32)

        # Determine far end of stripe
        projections = contour_pts @ direction
        far_idx = int(np.argmax(projections))
        far_end = contour_pts[far_idx]

        # Determine axis direction in millimetres
        if not self.homography.ready:
            self.reset()
            return result

        far_end_mm = self.homography.project_px_to_mm(far_end.reshape(1, 2))[0]
        near_point_px = far_end - direction * 10.0
        near_point_mm = self.homography.project_px_to_mm(near_point_px.reshape(1, 2))[0]
        axis_mm = far_end_mm - near_point_mm
        axis_mm = normalise(axis_mm)
        if np.allclose(axis_mm, 0):
            self.reset()
            return result

        tip_mm = far_end_mm - axis_mm * self.config.tip_offset_mm
        manual_offset = np.asarray(self.config.manual_offset_mm, dtype=float)
        tip_mm = tip_mm + manual_offset

        tip_px = self.homography.project_mm_to_px(tip_mm.reshape(1, 2))[0]
        tip_px = self.tip_filter.update(tip_px)

        roi = self._roi_from_tip(frame, tip_px, tip_mm)

        result.tip_px = tip_px
        result.tip_mm = tip_mm
        result.far_end_px = far_end
        result.contour = contour
        result.bounding_box = box
        result.roi = roi
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
        radius_mm = 3.0
        offsets_mm = np.array([
            tip_mm + [-radius_mm, -radius_mm],
            tip_mm + [radius_mm, radius_mm],
        ])
        roi_pts_px = self.homography.project_mm_to_px(offsets_mm)
        size_px = np.mean(np.abs(roi_pts_px[1] - roi_pts_px[0]))
        size = int(max(4, size_px))
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
            tip_mm + [-3.0, -3.0],
            tip_mm + [3.0, 3.0],
        ])
        roi_pts_px = self.homography.project_mm_to_px(offsets_mm)
        size_px = np.mean(np.abs(roi_pts_px[1] - roi_pts_px[0]))
        size = int(max(4, size_px))
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
# GUI controller
# ---------------------------------------------------------------------------


class StrokeCounterApp:
    def __init__(self, config: AppConfig, mode_override: Optional[str] = None) -> None:
        self.config = config
        if mode_override:
            self.config.mode = mode_override
        self.root = tk.Tk()
        self.root.title(APP_NAME)
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
        self.marker_homography = MarkerHomography()
        self.stripe_mode = StripeMode(self.config.stripe, self.marker_homography)
        self.rings_mode = RingsMode(self.config.rings, self.marker_homography)
        self.ink_tracker = InkActivityTracker()
        self.stroke_count = 0
        self.refine_count = 0
        self.last_tip_mm: Optional[np.ndarray] = None
        self.pen_down = False
        self.last_pen_down_time: Optional[float] = None
        self.last_pen_up_time: Optional[float] = None
        self._frame_lock = threading.Lock()
        self._update_timer: Optional[str] = None

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
        tools_menu.add_command(label="Select camera…", command=self._select_camera)
        tools_menu.add_command(label="Manual tip set…", command=self._manual_tip_adjust)
        menu_bar.add_cascade(label="Tools", menu=tools_menu)

        print_menu = tk.Menu(menu_bar, tearoff=False)
        print_menu.add_command(label="Print stripe pattern…", command=self._print_stripe)
        print_menu.add_command(label="Print ArUco markers…", command=self._print_aruco)
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
        self.status_var.set(f"Camera {index} active - waiting for ArUco markers...")
        self._schedule_update()

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
        if self.config.mode == "stripe":
            result = self.stripe_mode.process(frame)
            tip_px = result.tip_px
        else:
            result = self.rings_mode.process(frame)
            tip_px = result.tip_px

        roi_gray = None
        if result.roi is not None:
            x, y, w, h = result.roi
            roi = frame[y : y + h, x : x + w]
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        pen_down, activity = self.ink_tracker.update(roi_gray)

        if pen_down and not self.pen_down:
            self.stroke_count += 1
            self.last_pen_down_time = time.time()
        if not pen_down and self.pen_down:
            self.last_pen_up_time = time.time()
        self.pen_down = pen_down

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
        if tip_px is not None:
            cv2.circle(overlay, tuple(int(v) for v in tip_px), 6, (0, 0, 255), -1)
        if result.roi is not None:
            x, y, w, h = result.roi
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 255, 0), 1)

        text = f"Mode: {self.config.mode}  Strokes: {self.stroke_count}  Refine: {self.refine_count}  Ink:{activity:.1f}"
        cv2.putText(overlay, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if pen_down:
            cv2.putText(overlay, "PEN DOWN", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        helper_text = "Keys: R=re-lock S=screenshot ESC=quit"
        cv2.putText(
            overlay,
            helper_text,
            (20, overlay.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (180, 255, 180),
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

    def _color_calibration(self) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title("Colour calibration")
        dialog.grab_set()
        dialog.columnconfigure(0, weight=1)

        labels = ["H", "S", "V"]

        def create_scale_group(parent: tk.Widget, title: str, low: Sequence[int], high: Sequence[int]):
            frame = ttk.LabelFrame(parent, text=title)
            frame.grid_columnconfigure(0, weight=1)
            frame.pack(fill=tk.X, padx=10, pady=5)
            low_scales = []
            high_scales = []
            for idx, label in enumerate(labels):
                scale = tk.Scale(
                    frame,
                    from_=0,
                    to=255,
                    orient=tk.HORIZONTAL,
                    label=f"Low {label}",
                    length=280,
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
                    length=280,
                )
                scale.set(int(high[idx]))
                scale.grid(row=idx + 3, column=0, padx=6, pady=3, sticky="ew")
                high_scales.append(scale)
            return low_scales, high_scales

        if self.config.mode == "stripe":
            low_scales, high_scales = create_scale_group(
                dialog,
                "Stripe HSV range",
                self.config.stripe.hsv_low,
                self.config.stripe.hsv_high,
            )

            def save() -> None:
                new_low = [scale.get() for scale in low_scales]
                new_high = [scale.get() for scale in high_scales]
                self.config.stripe.hsv_low = tuple(new_low)
                self.config.stripe.hsv_high = tuple(new_high)
                self.config.dump(CONFIG_FILE)
                dialog.destroy()

        else:
            ringA_low_scales, ringA_high_scales = create_scale_group(
                dialog,
                "Ring A HSV range",
                self.config.rings.ringA_low,
                self.config.rings.ringA_high,
            )
            ringB_low_scales, ringB_high_scales = create_scale_group(
                dialog,
                "Ring B HSV range",
                self.config.rings.ringB_low,
                self.config.rings.ringB_high,
            )

            def save() -> None:
                self.config.rings.ringA_low = tuple(scale.get() for scale in ringA_low_scales)
                self.config.rings.ringA_high = tuple(scale.get() for scale in ringA_high_scales)
                self.config.rings.ringB_low = tuple(scale.get() for scale in ringB_low_scales)
                self.config.rings.ringB_high = tuple(scale.get() for scale in ringB_high_scales)
                self.config.dump(CONFIG_FILE)
                dialog.destroy()

        ttk.Button(dialog, text="Save", command=save).pack(pady=10)

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
        if self.config.mode == "stripe":
            result = self.stripe_mode.last_result
        else:
            result = self.rings_mode.last_result
        if result.tip_px is None or not self.marker_homography.ready:
            messagebox.showinfo(APP_NAME, "Tip not detected yet.")
            return

        frame = self.current_frame.copy()
        overlay = frame.copy()
        cv2.circle(overlay, tuple(int(v) for v in result.tip_px), 6, (0, 0, 255), -1)
        rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(image=image)

        dialog = tk.Toplevel(self.root)
        dialog.title("Manual tip alignment")
        dialog.grab_set()
        canvas = tk.Canvas(dialog, width=image.width, height=image.height)
        canvas.pack()
        canvas_img = canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        marker = canvas.create_oval(
            result.tip_px[0] - 6,
            result.tip_px[1] - 6,
            result.tip_px[0] + 6,
            result.tip_px[1] + 6,
            outline="red",
            fill="",
            width=2,
        )
        canvas.image = photo  # keep reference

        drag_data = {"x": result.tip_px[0], "y": result.tip_px[1]}

        def on_press(event: tk.Event) -> None:
            drag_data["x"] = event.x
            drag_data["y"] = event.y

        def on_drag(event: tk.Event) -> None:
            dx = event.x - drag_data["x"]
            dy = event.y - drag_data["y"]
            canvas.move(marker, dx, dy)
            drag_data["x"] = event.x
            drag_data["y"] = event.y

        canvas.tag_bind(marker, "<ButtonPress-1>", on_press)
        canvas.tag_bind(marker, "<B1-Motion>", on_drag)

        def save() -> None:
            coords = canvas.coords(marker)
            x = (coords[0] + coords[2]) / 2.0
            y = (coords[1] + coords[3]) / 2.0
            original_tip = np.array(result.tip_px, dtype=float)
            new_tip = np.array([x, y], dtype=float)
            delta_mm = self.marker_homography.project_px_to_mm(new_tip.reshape(1, 2))[0] - \
                self.marker_homography.project_px_to_mm(original_tip.reshape(1, 2))[0]
            if self.config.mode == "stripe":
                self.config.stripe.manual_offset_mm = tuple(
                    (np.asarray(self.config.stripe.manual_offset_mm) + delta_mm).tolist()
                )
            else:
                self.config.rings.manual_offset_mm = tuple(
                    (np.asarray(self.config.rings.manual_offset_mm) + delta_mm).tolist()
                )
            self.config.dump(CONFIG_FILE)
            dialog.destroy()

        ttk.Button(dialog, text="Save", command=save).pack(pady=10)

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

    def _show_about(self) -> None:
        messagebox.showinfo(
            APP_NAME,
            "Sofrim Stroke Counter\n\n"
            "Tracks a Torah scribe's quill to estimate writing strokes.\n"
            "Mode: Stripe (preferred) or Rings (fallback).\n"
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
    parser.add_argument("--mode", choices=["stripe", "rings"], help="Override detection mode")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    config = AppConfig.load(CONFIG_FILE)
    app = StrokeCounterApp(config, mode_override=getattr(args, "mode", None))
    app.run()


if __name__ == "__main__":  # pragma: no cover
    main()
