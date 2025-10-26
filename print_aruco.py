"""Generate a printable PDF containing ArUco markers for the stroke counter setup."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Iterable, List, Sequence

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
RAILS_PDF = Path("aruco_rails.pdf")
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


def create_rails_pdf(
    top_ids: Sequence[int],
    side_ids: Sequence[int],
    marker_size_mm: float,
    gap_h_mm: float,
    gap_v_mm: float,
    dictionary_name: str = "DICT_4X4_1000",
    output_path: Path = RAILS_PDF,
) -> None:
    dict_attr = dictionary_name.strip().upper()
    dict_id = getattr(cv2.aruco, dict_attr, cv2.aruco.DICT_4X4_1000)
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)

    page_width, page_height = A4
    marker_size_pt = marker_size_mm * MM_TO_PT
    gap_h_pt = gap_h_mm * MM_TO_PT
    gap_v_pt = gap_v_mm * MM_TO_PT
    margin_pt = 20.0 * MM_TO_PT

    canv = canvas.Canvas(str(output_path), pagesize=A4)

    readers = {marker_id: create_marker_image(dictionary, marker_id) for marker_id in (*top_ids, *side_ids)}

    # Draw top rail left to right
    x = margin_pt
    y = page_height - margin_pt - marker_size_pt
    for marker_id in top_ids:
        reader = readers[marker_id]
        canv.drawImage(reader, x, y, width=marker_size_pt, height=marker_size_pt, preserveAspectRatio=True)
        canv.setFont("Helvetica", 9)
        canv.drawCentredString(x + marker_size_pt / 2, y - 8, f"ID {marker_id}")
        x += marker_size_pt + gap_h_pt

    # Draw side rail column, one gap below the top rail
    side_x = margin_pt + len(top_ids) * (marker_size_pt + gap_h_pt) + gap_h_pt
    side_y = y - (marker_size_pt + gap_v_pt)
    for marker_id in side_ids:
        reader = readers[marker_id]
        canv.drawImage(
            reader,
            side_x,
            side_y,
            width=marker_size_pt,
            height=marker_size_pt,
            preserveAspectRatio=True,
        )
        canv.setFont("Helvetica", 9)
        canv.drawCentredString(side_x + marker_size_pt / 2, side_y - 8, f"ID {marker_id}")
        side_y -= marker_size_pt + gap_v_pt

    canv.showPage()
    canv.save()


if __name__ == "__main__":
    create_pdf()
