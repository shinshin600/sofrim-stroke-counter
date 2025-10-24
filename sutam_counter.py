# -*- coding: utf-8 -*-
"""Interactive quill stroke counter for sofrim.

This script locks onto a parchment using ArUco markers, tracks two coloured rings
on the quill and counts writing strokes by monitoring fresh ink deposition near the
quill tip. It targets Windows with Python 3.12 but works anywhere OpenCV can access a
camera.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from collections.abc import Deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np


if not hasattr(cv2, "aruco"):
    raise ImportError(
        "cv2.aruco is unavailable. Install the 'opencv-contrib-python' package to "
        "enable ArUco marker support."
    )


CONFIG_FILENAME = "config.json"
WINDOW_TITLE = "Sofrim Stroke Counter"
CALIBRATION_WINDOW = "HSV Calibration"

CAMERA_INDEX = 0
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
FRAME_FPS = 30

PREFERRED_DICTIONARY = cv2.aruco.DICT_4X4_1000
MARKER_IDS = list(range(10, 18))


@dataclass
class RingHSVConfig:
    """HSV threshold ranges for a coloured ring."""

    low: Tuple[int, int, int]
    high: Tuple[int, int, int]

    @staticmethod
    def from_mapping(mapping: Dict[str, Iterable[int]]) -> "RingHSVConfig":
        return RingHSVConfig(
            low=tuple(int(v) for v in mapping["low"]),
            high=tuple(int(v) for v in mapping["high"]),
        )

    def as_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        return (
            np.array(self.low, dtype=np.uint8),
            np.array(self.high, dtype=np.uint8),
        )


@dataclass
class AppConfig:
    """Runtime configuration loaded from JSON."""

    ringA: RingHSVConfig
    ringB: RingHSVConfig
    smoothing_window: int = 32
    threshold_on: float = 14.0
    threshold_off: float = 7.0
    parchment_width_mm: float = 400.0
    parchment_height_mm: float = 300.0
    roi_half_size_mm: float = 3.0
    refine_radius_mm: float = 0.8
    ring_tip_offset_mm: float = 10.0
    ring_spacing_mm: float = 10.0

    @staticmethod
    def from_json(path: Path) -> "AppConfig":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        ring_a = RingHSVConfig.from_mapping(data["ringA"])
        ring_b = RingHSVConfig.from_mapping(data["ringB"])
        return AppConfig(ringA=ring_a, ringB=ring_b)

    def to_json(self) -> Dict[str, Dict[str, Tuple[int, int, int]]]:
        return {
            "ringA": {"low": list(self.ringA.low), "high": list(self.ringA.high)},
            "ringB": {"low": list(self.ringB.low), "high": list(self.ringB.high)},
        }


@dataclass
class MarkerLock:
    """Stores the current homography and detected marker centres."""

    homography_mm_from_px: Optional[np.ndarray] = None
    homography_px_from_mm: Optional[np.ndarray] = None
    markers: Dict[int, np.ndarray] = field(default_factory=dict)

    def clear(self) -> None:
        self.homography_mm_from_px = None
        self.homography_px_from_mm = None
        self.markers.clear()

    def locked(self) -> bool:
        return self.homography_mm_from_px is not None and self.homography_px_from_mm is not None


@dataclass
class InkActivityTracker:
    """Maintains a smoothed ink activity signal and pen state."""

    threshold_on: float
    threshold_off: float
    window_size: int
    queue: Deque[float] = field(default_factory=lambda: deque(maxlen=32))
    previous_roi: Optional[np.ndarray] = None
    previous_shape: Optional[Tuple[int, int]] = None
    pen_down: bool = False

    def update(self, roi_gray: Optional[np.ndarray]) -> Tuple[bool, float]:
        if roi_gray is None:
            self.clear()
            return False, 0.0

        if self.queue.maxlen != self.window_size:
            self.queue = deque(self.queue, maxlen=self.window_size)

        activity = 0.0
        if self.previous_roi is not None and self.previous_shape == roi_gray.shape:
            diff = cv2.absdiff(roi_gray, self.previous_roi)
            activity = float(np.mean(diff))

        self.previous_roi = roi_gray.copy()
        self.previous_shape = roi_gray.shape
        self.queue.append(activity)

        smoothed = float(np.mean(self.queue)) if self.queue else 0.0

        if self.pen_down:
            if smoothed < self.threshold_off:
                self.pen_down = False
        else:
            if smoothed > self.threshold_on:
                self.pen_down = True

        return self.pen_down, smoothed

    def clear(self) -> None:
        self.queue.clear()
        self.previous_roi = None
        self.previous_shape = None
        self.pen_down = False


@dataclass
class StrokeCounter:
    """Counts strokes and refine oscillations."""

    strokes: int = 0
    refine: int = 0
    pen_down: bool = False
    circle_center_mm: Optional[np.ndarray] = None
    inside_circle: Optional[bool] = None

    def update(self, pen_down: bool, tip_mm: Optional[np.ndarray], radius_mm: float) -> None:
        if not pen_down or tip_mm is None:
            if self.pen_down and not pen_down:
                self.circle_center_mm = None
                self.inside_circle = None
            self.pen_down = pen_down
            return

        if not self.pen_down and pen_down:
            self.strokes += 1
            self.circle_center_mm = tip_mm.copy()
            self.inside_circle = True
        elif self.circle_center_mm is None:
            self.circle_center_mm = tip_mm.copy()
            self.inside_circle = True

        if self.circle_center_mm is not None:
            distance = float(np.linalg.norm(tip_mm - self.circle_center_mm))
            inside = distance <= radius_mm
            if self.inside_circle is None:
                self.inside_circle = inside
            elif inside != self.inside_circle:
                self.refine += 1
                self.inside_circle = inside

        self.pen_down = pen_down

    def reset(self) -> None:
        self.strokes = 0
        self.refine = 0
        self.pen_down = False
        self.circle_center_mm = None
        self.inside_circle = None


class HSVCalibrator:
    """Trackbar-based calibration UI for HSV ranges."""

    _instance: Optional["HSVCalibrator"] = None

    def __init__(self, window_name: str, config_path: Path, config: AppConfig) -> None:
        HSVCalibrator._instance = self
        self.window_name = window_name
        self.config_path = config_path
        self.config = config
        self._suspend = False

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 480, 240)

        self._suspend = True
        for ring_name, ring_cfg in ("ringA", self.config.ringA), ("ringB", self.config.ringB):
            for bound_name in ("low", "high"):
                values = getattr(ring_cfg, bound_name)
                for channel, max_value, initial in zip("HSV", (179, 255, 255), values):
                    trackbar_name = f"{ring_name}_{bound_name}_{channel}"
                    cv2.createTrackbar(
                        trackbar_name,
                        self.window_name,
                        int(initial),
                        max_value,
                        HSVCalibrator._trackbar_callback,
                    )
        self._suspend = False
        self._update_config(save=False)

    @staticmethod
    def _trackbar_callback(_: int) -> None:
        if HSVCalibrator._instance is not None:
            HSVCalibrator._instance._handle_trackbar_change()

    def _handle_trackbar_change(self) -> None:
        if self._suspend:
            return
        self._update_config(save=True)

    def _update_config(self, save: bool) -> None:
        for ring_attr in ("ringA", "ringB"):
            low = []
            high = []
            for channel, max_value in zip("HSV", (179, 255, 255)):
                low_name = f"{ring_attr}_low_{channel}"
                high_name = f"{ring_attr}_high_{channel}"
                low.append(cv2.getTrackbarPos(low_name, self.window_name))
                high.append(cv2.getTrackbarPos(high_name, self.window_name))
            setattr(
                self.config,
                ring_attr,
                RingHSVConfig(low=tuple(low), high=tuple(high)),
            )
        if save:
            self.save()

    def save(self) -> None:
        data = self.config.to_json()
        with self.config_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)


def ensure_config(path: Path) -> AppConfig:
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file '{path}' is missing. Provide HSV ranges in JSON as documented."
        )
    return AppConfig.from_json(path)


def create_capture(index: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    capture.set(cv2.CAP_PROP_FPS, FRAME_FPS)
    return capture


def detect_markers(
    frame: np.ndarray,
    detector: cv2.aruco.ArucoDetector,
) -> Dict[int, np.ndarray]:
    corners, ids, _ = detector.detectMarkers(frame)
    marker_centres: Dict[int, np.ndarray] = {}
    if ids is None:
        return marker_centres
    for marker_corners, marker_id in zip(corners, ids.flatten().tolist()):
        if marker_id in MARKER_IDS:
            pts = marker_corners.reshape(-1, 2)
            marker_centres[marker_id] = pts.mean(axis=0)
    return marker_centres


def compute_homography(
    centres: Dict[int, np.ndarray],
    width_mm: float,
    height_mm: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not all(marker_id in centres for marker_id in MARKER_IDS[:4]):
        return None, None

    src = np.array(
        [
            centres[10],
            centres[11],
            centres[12],
            centres[13],
        ],
        dtype=np.float32,
    )
    dst = np.array(
        [
            [0.0, 0.0],
            [width_mm, 0.0],
            [width_mm, height_mm],
            [0.0, height_mm],
        ],
        dtype=np.float32,
    )

    homography_mm_from_px, mask = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if homography_mm_from_px is None or mask is None:
        return None, None

    homography_px_from_mm = np.linalg.inv(homography_mm_from_px)
    return homography_mm_from_px, homography_px_from_mm


def project_point(point: np.ndarray, homography: np.ndarray) -> np.ndarray:
    pts = cv2.perspectiveTransform(point.reshape(-1, 1, 2).astype(np.float32), homography)
    return pts.reshape(-1, 2)[0]


def project_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    pts = cv2.perspectiveTransform(points.reshape(-1, 1, 2).astype(np.float32), homography)
    return pts.reshape(-1, 2)


def detect_ring(frame_hsv: np.ndarray, threshold: RingHSVConfig) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    low, high = threshold.as_arrays()
    mask = cv2.inRange(frame_hsv, low, high)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < 50:
        return None, None
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None, None
    centre = np.array(
        [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
        dtype=np.float32,
    )
    return centre, contour


def compute_tip(
    ring_a_px: np.ndarray,
    ring_b_px: np.ndarray,
    homography_mm_from_px: np.ndarray,
    homography_px_from_mm: np.ndarray,
    tip_offset_mm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    ring_a_mm = project_point(ring_a_px, homography_mm_from_px)
    ring_b_mm = project_point(ring_b_px, homography_mm_from_px)
    direction = ring_b_mm - ring_a_mm
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        raise ValueError("Ring detections are degenerate; cannot estimate quill direction")
    unit = direction / norm
    tip_mm = ring_a_mm - unit * tip_offset_mm
    tip_px = project_point(tip_mm, homography_px_from_mm)
    return tip_mm, tip_px


def warp_to_work_area(
    frame: np.ndarray,
    homography_mm_from_px: np.ndarray,
    width_mm: float,
    height_mm: float,
) -> np.ndarray:
    width_px = int(round(width_mm))
    height_px = int(round(height_mm))
    width_px = max(width_px, 1)
    height_px = max(height_px, 1)
    return cv2.warpPerspective(frame, homography_mm_from_px, (width_px, height_px))


def roi_polygon_mm(tip_mm: np.ndarray, half_size_mm: float) -> np.ndarray:
    offsets = np.array(
        [
            [-half_size_mm, -half_size_mm],
            [half_size_mm, -half_size_mm],
            [half_size_mm, half_size_mm],
            [-half_size_mm, half_size_mm],
        ],
        dtype=np.float32,
    )
    return tip_mm.reshape(1, 2) + offsets


def extract_roi_from_mm_view(
    mm_gray: np.ndarray,
    tip_mm: np.ndarray,
    half_size_mm: float,
) -> Optional[np.ndarray]:
    height, width = mm_gray.shape[:2]
    x_min = max(int(np.floor(tip_mm[0] - half_size_mm)), 0)
    x_max = min(int(np.ceil(tip_mm[0] + half_size_mm)), width - 1)
    y_min = max(int(np.floor(tip_mm[1] - half_size_mm)), 0)
    y_max = min(int(np.ceil(tip_mm[1] + half_size_mm)), height - 1)
    if x_max <= x_min or y_max <= y_min:
        return None
    return mm_gray[y_min : y_max + 1, x_min : x_max + 1]


def save_screenshot(frame: np.ndarray, directory: Path = Path("screenshots")) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = directory / f"diagnostic_{timestamp}.png"
    cv2.imwrite(str(filename), frame)
    return filename


def draw_overlays(
    frame: np.ndarray,
    lock: MarkerLock,
    ring_a: Tuple[Optional[np.ndarray], Optional[np.ndarray]],
    ring_b: Tuple[Optional[np.ndarray], Optional[np.ndarray]],
    tip_px: Optional[np.ndarray],
    tip_mm: Optional[np.ndarray],
    roi_mm: Optional[np.ndarray],
    stroke_counter: StrokeCounter,
    config: AppConfig,
) -> np.ndarray:
    overlay = frame.copy()

    if lock.markers:
        for marker_id, centre in lock.markers.items():
            cv2.circle(overlay, tuple(np.round(centre).astype(int)), 8, (255, 0, 0), 2)
            cv2.putText(
                overlay,
                f"ID {marker_id}",
                tuple(np.round(centre + np.array([10, -10])).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )

    if lock.locked():
        rect_mm = np.array(
            [
                [0.0, 0.0],
                [config.parchment_width_mm, 0.0],
                [config.parchment_width_mm, config.parchment_height_mm],
                [0.0, config.parchment_height_mm],
            ],
            dtype=np.float32,
        )
        rect_px = project_points(rect_mm, lock.homography_px_from_mm)
        cv2.polylines(overlay, [np.round(rect_px).astype(int)], True, (0, 255, 255), 2)

        if stroke_counter.circle_center_mm is not None:
            center_px = project_point(stroke_counter.circle_center_mm, lock.homography_px_from_mm)
            boundary_pt_mm = stroke_counter.circle_center_mm + np.array([config.refine_radius_mm, 0.0])
            boundary_px = project_point(boundary_pt_mm, lock.homography_px_from_mm)
            radius_px = int(max(1.0, np.linalg.norm(boundary_px - center_px)))
            cv2.circle(overlay, tuple(np.round(center_px).astype(int)), radius_px, (0, 165, 255), 1)

        if roi_mm is not None:
            roi_px = project_points(roi_mm, lock.homography_px_from_mm)
            cv2.polylines(overlay, [np.round(roi_px).astype(int)], True, (255, 0, 255), 2)

    for centre, contour, colour in (
        (ring_a[0], ring_a[1], (0, 255, 255)),
        (ring_b[0], ring_b[1], (0, 255, 0)),
    ):
        if centre is not None:
            cv2.circle(overlay, tuple(np.round(centre).astype(int)), 8, colour, 2)
        if contour is not None:
            cv2.drawContours(overlay, [contour], -1, colour, 1)

    if tip_px is not None:
        cv2.circle(overlay, tuple(np.round(tip_px).astype(int)), 6, (0, 0, 255), -1)

    info_lines = [
        f"Strokes: {stroke_counter.strokes}",
        f"Pen down: {'Yes' if stroke_counter.pen_down else 'No'}",
        f"Refine counter: {stroke_counter.refine}",
        "Keys: R=re-lock  S=screenshot  ESC=quit",
    ]
    for idx, line in enumerate(info_lines):
        cv2.putText(
            overlay,
            line,
            (20, 40 + idx * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (50, 255, 50),
            2,
        )

    return overlay


def main() -> None:
    parser = argparse.ArgumentParser(description="Torah quill stroke counter")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help="Camera index to use")
    parser.add_argument("--config", type=Path, default=Path(CONFIG_FILENAME), help="HSV configuration JSON")
    args = parser.parse_args()

    config = ensure_config(args.config)
    HSVCalibrator(CALIBRATION_WINDOW, args.config, config)

    dictionary = cv2.aruco.getPredefinedDictionary(PREFERRED_DICTIONARY)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    capture = create_capture(args.camera)
    if not capture.isOpened():
        raise RuntimeError("Unable to open the webcam. Check camera connection and permissions.")

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, 1280, 720)

    lock = MarkerLock()
    ink_tracker = InkActivityTracker(
        threshold_on=config.threshold_on,
        threshold_off=config.threshold_off,
        window_size=config.smoothing_window,
    )
    stroke_counter = StrokeCounter()

    last_screenshot_time = 0.0
    relock = True

    try:
        while True:
            success, frame = capture.read()
            if not success:
                raise RuntimeError("Camera frame could not be read.")

            ring_a = (None, None)
            ring_b = (None, None)
            tip_px: Optional[np.ndarray] = None
            tip_mm: Optional[np.ndarray] = None
            roi_mm: Optional[np.ndarray] = None

            if relock or not lock.locked():
                centres = detect_markers(frame, detector)
                lock.markers = centres
                homographies = compute_homography(centres, config.parchment_width_mm, config.parchment_height_mm)
                lock.homography_mm_from_px, lock.homography_px_from_mm = homographies
                relock = False
                ink_tracker.clear()
                stroke_counter.reset()

            if lock.locked():
                frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                ring_a = detect_ring(frame_hsv, config.ringA)
                ring_b = detect_ring(frame_hsv, config.ringB)

                if ring_a[0] is not None and ring_b[0] is not None:
                    try:
                        tip_mm, tip_px = compute_tip(
                            ring_a[0],
                            ring_b[0],
                            lock.homography_mm_from_px,
                            lock.homography_px_from_mm,
                            config.ring_tip_offset_mm,
                        )
                    except ValueError:
                        tip_mm = None
                        tip_px = None

            pen_down = False
            if tip_mm is not None and lock.locked():
                mm_view = warp_to_work_area(frame, lock.homography_mm_from_px, config.parchment_width_mm, config.parchment_height_mm)
                mm_gray = cv2.cvtColor(mm_view, cv2.COLOR_BGR2GRAY)
                roi_mm_polygon = roi_polygon_mm(tip_mm, config.roi_half_size_mm)
                roi_mm = roi_mm_polygon
                roi_gray = extract_roi_from_mm_view(mm_gray, tip_mm, config.roi_half_size_mm)
                pen_down, _ = ink_tracker.update(roi_gray)
            else:
                ink_tracker.clear()
                roi_gray = None

            stroke_counter.update(pen_down, tip_mm, config.refine_radius_mm)

            frame_overlay = draw_overlays(
                frame,
                lock,
                ring_a,
                ring_b,
                tip_px,
                tip_mm,
                roi_mm,
                stroke_counter,
                config,
            )

            cv2.imshow(WINDOW_TITLE, frame_overlay)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key in (ord("r"), ord("R")):
                relock = True
            elif key in (ord("s"), ord("S")):
                if time.time() - last_screenshot_time > 1.0:
                    path = save_screenshot(frame_overlay)
                    print(f"Saved diagnostic screenshot to {path}")
                    last_screenshot_time = time.time()
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
