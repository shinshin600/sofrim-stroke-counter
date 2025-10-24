"""Generate a printable PDF containing ArUco markers for the stroke counter setup."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Iterable, List

import cv2
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


if not hasattr(cv2, "aruco"):
    raise ImportError(
        "cv2.aruco is unavailable. Install the 'opencv-contrib-python' package to generate markers."
    )


OUTPUT_PDF = Path("aruco_markers.pdf")
MARKER_IDS: List[int] = list(range(10, 18))
MARKER_SIZE_MM = 10.0
MARGIN_MM = 15.0
H_SPACING_MM = 12.0
V_SPACING_MM = 18.0
COLUMNS = 4
ROWS = 2
MM_TO_PT = 2.83465
MARKER_PIXELS = 600


def mm_to_points(values: Iterable[float]) -> List[float]:
    return [float(v) * MM_TO_PT for v in values]


def create_marker_image(dictionary, marker_id: int) -> ImageReader:
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, MARKER_PIXELS)
    marker_rgb = cv2.cvtColor(marker, cv2.COLOR_GRAY2RGB)
    image = Image.fromarray(marker_rgb)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return ImageReader(buffer)


def create_pdf(output_path: Path = OUTPUT_PDF) -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    page_width, page_height = A4
    marker_size_pt = MARKER_SIZE_MM * MM_TO_PT
    margin_x_pt, margin_y_pt = mm_to_points([MARGIN_MM, MARGIN_MM])
    h_spacing_pt = H_SPACING_MM * MM_TO_PT
    v_spacing_pt = V_SPACING_MM * MM_TO_PT

    canv = canvas.Canvas(str(output_path), pagesize=A4)

    readers = [create_marker_image(dictionary, marker_id) for marker_id in MARKER_IDS]

    x = margin_x_pt
    y = page_height - margin_y_pt - marker_size_pt
    for idx, (marker_id, reader) in enumerate(zip(MARKER_IDS, readers)):
        canv.drawImage(reader, x, y, width=marker_size_pt, height=marker_size_pt, preserveAspectRatio=True)
        canv.setFont("Helvetica", 9)
        canv.drawCentredString(x + marker_size_pt / 2, y - 8, f"ID {marker_id}")

        x += marker_size_pt + h_spacing_pt
        if (idx + 1) % COLUMNS == 0:
            x = margin_x_pt
            y -= marker_size_pt + v_spacing_pt

    canv.showPage()
    canv.save()


if __name__ == "__main__":
    create_pdf()
