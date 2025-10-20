# -*- coding: utf-8 -*-
"""Real-time quill stroke counter for sofrim.

This module provides an interactive OpenCV application that detects the quill tip
using two coloured rings, locks the writing surface via ArUco markers and counts
strokes based on fresh ink appearance in the vicinity of the quill tip.

The implementation is designed to run on Windows but works on any platform where
OpenCV can access a camera.
"""
from __future__ import annotations

import argparse
import collections
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


CONFIG_FILE = "config.json"
WINDOW_TITLE = "Sofrim Stroke Counter"
CAMERA_INDEX = 0
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
FRAME_FPS = 30

# ArUco marker IDs expected for the four corners of the parchment
CORNER_IDS = {
    10: "top_left",
    11: "top_right",
    12: "bottom_right",
    13: "bottom_left",
}

# Dimensions of the parchment area (in millimetres) used for the homography
# These values can be adjusted to match the real parchment size
PARCHMENT_WIDTH_MM = 300.0
PARCHMENT_HEIGHT_MM = 200.0

# Rolling buffer size for ink activity smoothing
INK_BUFFER_SIZE = 15

# Size of the ROI (in millimetres) around the quill tip used for ink detection
ROI_HALF_SIZE_MM = 4.0

# Distance from ring A (yellow) to the tip in millimetres
TIP_OFFSET_MM = 10.0

# Radius of refine counter area in millimetres
REFINE_RADIUS_MM = 0.8

# Location of refine counter area (centre in millimetres relative to parchment origin)
REFINE_CENTER_MM = (PARCHMENT_WIDTH_MM / 2.0, PARCHMENT_HEIGHT_MM / 2.0)


@dataclass
class RingThreshold:
    """HSV range for a ring."""

    low: np.ndarray
    high: np.ndarray

    @staticmethod
    def from_dict(data: Dict[str, Iterable[int]]) -> "RingThreshold":
        return RingThreshold(
            low=np.array(list(data["low"]), dtype=np.uint8),
            high=np.array(list(data["high"]), dtype=np.uint8),
        )


@dataclass
class RingDetection:
    """Result of detecting a ring."""

    centre_px: Optional[np.ndarray]
    contour: Optional[np.ndarray]

    def is_valid(self) -> bool:
        return self.centre_px is not None


@dataclass
class TrackingState:
    homography_px_from_mm: Optional[np.ndarray] = None
    homography_mm_from_px: Optional[np.ndarray] = None
    strokes: int = 0
    pen_down: bool = False
    refine_count: int = 0
    inside_refine: bool = False
    last_screenshot_time: float = 0.0
    ink_activity_queue: Deque[float] = collections.deque(maxlen=INK_BUFFER_SIZE)


def load_config(config_path: Path) -> Dict[str, RingThreshold]:
    """Load HSV threshold configuration for the quill rings."""

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file '{config_path}' not found. "
            "Create it based on the example in the README."
        )
    with config_path.open("r", encoding="utf-8") as fh:
        config_data = json.load(fh)
    return {
        "ringA": RingThreshold.from_dict(config_data["ringA"]),
        "ringB": RingThreshold.from_dict(config_data["ringB"]),
    }


def save_screenshot(frame: np.ndarray, output_dir: Path = Path("screenshots")) -> Path:
    """Save a diagnostic screenshot with timestamp."""

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = output_dir / f"snapshot-{timestamp}.png"
    cv2.imwrite(str(filename), frame)
    return filename


def create_capture(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FRAME_FPS)
    return cap


def detect_markers(frame: np.ndarray) -> Tuple[List[np.ndarray], List[int]]:
    """Detect ArUco markers in the frame."""

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    corners, ids, _ = detector.detectMarkers(frame)
    if ids is None:
        return [], []
    return corners, ids.flatten().tolist()


def compute_homography_from_markers(
    corners: List[np.ndarray], ids: List[int]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, np.ndarray]]:
    """Compute homography matrices from detected marker corners."""

    # Map of marker name to centre pixel coordinate
    corner_map: Dict[str, np.ndarray] = {}

    for marker_corners, marker_id in zip(corners, ids):
        if marker_id in CORNER_IDS:
            pts = marker_corners.reshape(-1, 2)
            centre = pts.mean(axis=0)
            corner_map[CORNER_IDS[marker_id]] = centre

    if len(corner_map) < 4:
        return None, None, corner_map

    src_pts = np.array(
        [
            corner_map["top_left"],
            corner_map["top_right"],
            corner_map["bottom_right"],
            corner_map["bottom_left"],
        ],
        dtype=np.float32,
    )

    dst_pts = np.array(
        [
            [0.0, 0.0],
            [PARCHMENT_WIDTH_MM, 0.0],
            [PARCHMENT_WIDTH_MM, PARCHMENT_HEIGHT_MM],
            [0.0, PARCHMENT_HEIGHT_MM],
        ],
        dtype=np.float32,
    )

    homography_mm_from_px, mask = cv2.findHomography(
        src_pts, dst_pts, method=cv2.RANSAC
    )
    if homography_mm_from_px is None:
        return None, None, corner_map

    homography_px_from_mm = np.linalg.inv(homography_mm_from_px)

    return homography_px_from_mm, homography_mm_from_px, corner_map


def project_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Apply perspective transform to a list of points."""

    pts = cv2.perspectiveTransform(points.reshape(-1, 1, 2).astype(np.float32), homography)
    return pts.reshape(-1, 2)


def detect_ring(frame_hsv: np.ndarray, threshold: RingThreshold) -> RingDetection:
    """Detect a coloured ring and return its centre."""

    mask = cv2.inRange(frame_hsv, threshold.low, threshold.high)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return RingDetection(None, None)

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < 50:  # Ignore tiny detections
        return RingDetection(None, None)

    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return RingDetection(None, None)

    centre = np.array([
        float(moments["m10"] / moments["m00"]),
        float(moments["m01"] / moments["m00"]),
    ])

    return RingDetection(centre, contour)


def mm_to_pixels(point_mm: np.ndarray, homography_px_from_mm: np.ndarray) -> np.ndarray:
    """Convert a point from millimetres to pixel coordinates."""

    pts_px = project_points(np.array([point_mm], dtype=np.float32), homography_px_from_mm)
    return pts_px[0]


def pixels_to_mm(point_px: np.ndarray, homography_mm_from_px: np.ndarray) -> np.ndarray:
    """Convert a point from pixels to millimetres."""

    pts_mm = project_points(np.array([point_px], dtype=np.float32), homography_mm_from_px)
    return pts_mm[0]


def compute_quill_tip(
    ring_a_px: np.ndarray,
    ring_b_px: np.ndarray,
    homography_mm_from_px: np.ndarray,
    homography_px_from_mm: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute quill tip in both millimetres and pixels."""

    ring_a_mm = pixels_to_mm(ring_a_px, homography_mm_from_px)
    ring_b_mm = pixels_to_mm(ring_b_px, homography_mm_from_px)

    direction = ring_a_mm - ring_b_mm
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        raise ValueError("Ring positions are too close to determine direction")
    v = direction / norm
    tip_mm = ring_a_mm - v * TIP_OFFSET_MM

    tip_px = mm_to_pixels(tip_mm, homography_px_from_mm)
    return tip_mm, tip_px, v


def compute_roi_pixels(
    tip_mm: np.ndarray, homography_px_from_mm: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute top-left and bottom-right pixel coordinates of the ROI around the tip."""

    offsets = np.array(
        [
            [-ROI_HALF_SIZE_MM, -ROI_HALF_SIZE_MM],
            [ROI_HALF_SIZE_MM, -ROI_HALF_SIZE_MM],
            [ROI_HALF_SIZE_MM, ROI_HALF_SIZE_MM],
            [-ROI_HALF_SIZE_MM, ROI_HALF_SIZE_MM],
        ],
        dtype=np.float32,
    )
    corners_mm = tip_mm + offsets
    corners_px = project_points(corners_mm, homography_px_from_mm)
    min_xy = np.clip(np.min(corners_px, axis=0), [0, 0], [FRAME_WIDTH - 1, FRAME_HEIGHT - 1])
    max_xy = np.clip(np.max(corners_px, axis=0), [0, 0], [FRAME_WIDTH - 1, FRAME_HEIGHT - 1])
    return min_xy.astype(int), max_xy.astype(int)


def evaluate_ink_activity(
    roi_gray: np.ndarray,
    roi_prev_gray: Optional[np.ndarray],
    activity_queue: Deque[float],
) -> float:
    """Detect new ink by comparing ROI with previous frame and return smoothed activity."""

    if roi_prev_gray is None or roi_prev_gray.shape != roi_gray.shape:
        activity_queue.append(0.0)
        return 0.0

    diff = cv2.absdiff(roi_gray, roi_prev_gray)
    _, diff_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    activity_value = float(np.sum(diff_mask) / 255.0)

    activity_queue.append(activity_value)
    smoothed = float(np.mean(activity_queue))
    return smoothed


def update_pen_state(
    activity_value: float,
    state: TrackingState,
    pen_down_threshold: float = 80.0,
    pen_up_threshold: float = 40.0,
) -> None:
    """Update pen-down state based on ink activity with hysteresis."""

    if state.pen_down:
        if activity_value < pen_up_threshold:
            state.pen_down = False
    else:
        if activity_value > pen_down_threshold:
            state.pen_down = True
            state.strokes += 1


def update_refine_counter(tip_mm: np.ndarray, state: TrackingState) -> None:
    """Count entries/exits into the refine area while the pen is down."""

    if state.homography_mm_from_px is None:
        return

    dist = np.linalg.norm(tip_mm - np.array(REFINE_CENTER_MM))
    inside = dist <= REFINE_RADIUS_MM
    if state.pen_down:
        if inside != state.inside_refine:
            state.refine_count += 1
    state.inside_refine = inside


def draw_overlays(
    frame: np.ndarray,
    state: TrackingState,
    corner_map: Dict[str, np.ndarray],
    ring_a: RingDetection,
    ring_b: RingDetection,
    tip_px: Optional[np.ndarray],
    roi_bounds: Optional[Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Draw UI overlays on the frame."""

    overlay = frame.copy()

    # Draw markers
    for name, point in corner_map.items():
        cv2.circle(overlay, tuple(point.astype(int)), 8, (255, 0, 0), 2)
        cv2.putText(
            overlay,
            name,
            tuple(point.astype(int) + np.array([10, -10])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )

    if state.homography_px_from_mm is not None:
        rect_mm = np.array(
            [
                [0.0, 0.0],
                [PARCHMENT_WIDTH_MM, 0.0],
                [PARCHMENT_WIDTH_MM, PARCHMENT_HEIGHT_MM],
                [0.0, PARCHMENT_HEIGHT_MM],
            ],
            dtype=np.float32,
        )
        rect_px = project_points(rect_mm, state.homography_px_from_mm).astype(int)
        cv2.polylines(overlay, [rect_px], True, (0, 255, 255), 2)

        refine_center_px = mm_to_pixels(np.array(REFINE_CENTER_MM), state.homography_px_from_mm)
        refine_radius_px = int(
            np.linalg.norm(
                mm_to_pixels(np.array([REFINE_CENTER_MM[0] + REFINE_RADIUS_MM, REFINE_CENTER_MM[1]]), state.homography_px_from_mm)
                - refine_center_px
            )
        )
        cv2.circle(overlay, tuple(refine_center_px.astype(int)), refine_radius_px, (0, 128, 255), 2)

    # Draw rings
    for ring, color in ((ring_a, (0, 255, 255)), (ring_b, (0, 255, 0))):
        if ring.is_valid():
            cv2.circle(overlay, tuple(ring.centre_px.astype(int)), 8, color, 2)
            if ring.contour is not None:
                cv2.drawContours(overlay, [ring.contour], -1, color, 1)

    if tip_px is not None:
        cv2.circle(overlay, tuple(tip_px.astype(int)), 6, (0, 0, 255), -1)

    if roi_bounds is not None:
        tl, br = roi_bounds
        cv2.rectangle(overlay, tuple(tl), tuple(br), (255, 0, 255), 2)

    info_lines = [
        f"Strokes: {state.strokes}",
        f"Pen down: {'Yes' if state.pen_down else 'No'}",
        f"Refine count: {state.refine_count}",
        "Keys: R=Re-lock S=Screenshot ESC=Quit",
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
    parser = argparse.ArgumentParser(description="Sofrim quill stroke counter")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help="Camera index")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(CONFIG_FILE),
        help="Path to HSV configuration JSON file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    ring_a_threshold = config["ringA"]
    ring_b_threshold = config["ringB"]

    state = TrackingState()
    cap = create_capture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Unable to access camera")

    prev_gray_roi: Optional[np.ndarray] = None

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

    relock_requested = True

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_display = frame.copy()
            corner_map: Dict[str, np.ndarray] = {}
            tip_px: Optional[np.ndarray] = None
            roi_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
            tip_mm: Optional[np.ndarray] = None

            if relock_requested or state.homography_px_from_mm is None:
                corners, ids = detect_markers(frame)
                homographies = compute_homography_from_markers(corners, ids)
                (state.homography_px_from_mm, state.homography_mm_from_px, corner_map) = homographies
                relock_requested = False
                prev_gray_roi = None

            if state.homography_mm_from_px is not None:
                frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                ring_a = detect_ring(frame_hsv, ring_a_threshold)
                ring_b = detect_ring(frame_hsv, ring_b_threshold)
            else:
                ring_a = RingDetection(None, None)
                ring_b = RingDetection(None, None)

            if ring_a.is_valid() and ring_b.is_valid() and state.homography_mm_from_px is not None:
                try:
                    tip_mm, tip_px, direction = compute_quill_tip(
                        ring_a.centre_px,
                        ring_b.centre_px,
                        state.homography_mm_from_px,
                        state.homography_px_from_mm,
                    )
                except ValueError:
                    tip_mm = None
                    tip_px = None
                else:
                    roi_tl, roi_br = compute_roi_pixels(tip_mm, state.homography_px_from_mm)
                    roi_bounds = (roi_tl, roi_br)

                    if roi_br[0] - roi_tl[0] > 2 and roi_br[1] - roi_tl[1] > 2:
                        roi = frame[roi_tl[1] : roi_br[1], roi_tl[0] : roi_br[0]]
                        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        activity_value = evaluate_ink_activity(
                            roi_gray, prev_gray_roi, state.ink_activity_queue
                        )
                        prev_gray_roi = roi_gray.copy()
                        update_pen_state(activity_value, state)
                        update_refine_counter(tip_mm, state)
                    else:
                        prev_gray_roi = None
                        state.pen_down = False
            else:
                # Reset ROI tracking if we lose the quill
                prev_gray_roi = None
                state.pen_down = False
                state.inside_refine = False

            frame_display = draw_overlays(frame_display, state, corner_map, ring_a, ring_b, tip_px, roi_bounds)
            cv2.imshow(WINDOW_TITLE, frame_display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key in (ord("r"), ord("R")):
                relock_requested = True
                state.homography_px_from_mm = None
                state.homography_mm_from_px = None
            elif key in (ord("s"), ord("S")):
                if time.time() - state.last_screenshot_time > 1.0:
                    path = save_screenshot(frame_display)
                    print(f"Saved screenshot to {path}")
                    state.last_screenshot_time = time.time()

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
