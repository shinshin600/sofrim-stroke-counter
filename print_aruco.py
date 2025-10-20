"""Generate a PDF with ArUco markers for sofrim stroke counter calibration."""
from __future__ import annotations

from pathlib import Path

import cv2
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


OUTPUT_PDF = "aruco_markers.pdf"
MARKER_IDS = list(range(10, 18))
MARKER_SIZE_MM = 10.0
MARGIN_MM = 15.0
SPACING_MM = 10.0


def mm_to_points(mm: float) -> float:
    """Convert millimetres to PDF points."""

    return mm * 72.0 / 25.4


def generate_marker_image(dictionary, marker_id: int, pixels: int = 300) -> Path:
    """Create a temporary PNG image for an ArUco marker."""

    marker_image = cv2.aruco.generateImageMarker(dictionary, marker_id, pixels)
    temp_path = Path(f"aruco_{marker_id}.png")
    cv2.imwrite(str(temp_path), marker_image)
    return temp_path


def create_pdf(output_path: Path = Path(OUTPUT_PDF)) -> None:
    """Generate the PDF file containing the markers."""

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    c = canvas.Canvas(str(output_path), pagesize=A4)
    page_width, page_height = A4

    marker_size_pts = mm_to_points(MARKER_SIZE_MM)
    margin_pts = mm_to_points(MARGIN_MM)
    spacing_pts = mm_to_points(SPACING_MM)

    cols = 4

    x = margin_pts
    y = page_height - margin_pts - marker_size_pts

    temp_files = []
    try:
        for idx, marker_id in enumerate(MARKER_IDS):
            image_path = generate_marker_image(dictionary, marker_id)
            temp_files.append(image_path)

            c.drawImage(
                str(image_path),
                x,
                y,
                width=marker_size_pts,
                height=marker_size_pts,
            )
            c.drawString(x, y - mm_to_points(3), f"ID {marker_id}")

            x += marker_size_pts + spacing_pts
            if (idx + 1) % cols == 0:
                x = margin_pts
                y -= marker_size_pts + spacing_pts
        c.save()
    finally:
        for temp_path in temp_files:
            if temp_path.exists():
                temp_path.unlink()


if __name__ == "__main__":
    create_pdf()
