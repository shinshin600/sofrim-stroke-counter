"""Generate stripe pattern PDF for the stripe tracking mode."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

MM_TO_PT = 2.83465
OUTPUT_PDF = Path("stripe_pattern.pdf")
STRIPE_LENGTH_MM = 70.0
STRIPE_HEIGHT_MM = 14.0
PADDING_MM = 20.0
LABEL_GAP_MM = 6.0


def mm_to_pt(value: float) -> float:
    return value * MM_TO_PT


def mm_list(values: Iterable[float]) -> Sequence[float]:
    return [mm_to_pt(v) for v in values]


def _palette_colors(palette: str) -> Sequence[colors.Color]:
    palette = palette.lower()
    if palette == "black-white":
        return (colors.black, colors.white)
    if palette == "white-black":
        return (colors.white, colors.black)
    if palette in {"green-white", "white-green"}:
        return (colors.Color(0, 0.45, 0), colors.white)
    if palette in {"black-green", "green-black"}:
        return (colors.black, colors.Color(0, 0.45, 0))
    return (colors.black, colors.white)


def _draw_ticks(canv: canvas.Canvas, x: float, y: float, length_pt: float, tick_mm: float) -> None:
    if tick_mm <= 0:
        return
    tick_pt = mm_to_pt(tick_mm)
    position = 0.0
    while position <= length_pt + 0.1:
        canv.line(x + position, y, x + position, y - mm_to_pt(3))
        position += tick_pt


def draw_stripe(
    canv: canvas.Canvas,
    origin_x_pt: float,
    origin_y_pt: float,
    palette_name: str,
    label: str,
    tick_mm: float,
    use_ticks: bool,
) -> None:
    blocks = 14  # results in 5 mm blocks for 70 mm stripe
    colors_pair = _palette_colors(palette_name)
    block_length_pt = mm_to_pt(STRIPE_LENGTH_MM / blocks)
    stripe_height_pt = mm_to_pt(STRIPE_HEIGHT_MM)
    total_length_pt = block_length_pt * blocks

    for idx in range(blocks):
        canv.setFillColor(colors_pair[idx % 2])
        canv.rect(
            origin_x_pt + idx * block_length_pt,
            origin_y_pt,
            block_length_pt,
            stripe_height_pt,
            stroke=1,
            fill=1,
        )

    canv.setFillColor(colors.black)
    canv.setStrokeColor(colors.black)
    canv.rect(origin_x_pt, origin_y_pt, total_length_pt, stripe_height_pt, stroke=1, fill=0)

    if use_ticks:
        _draw_ticks(canv, origin_x_pt, origin_y_pt, total_length_pt, tick_mm)

    canv.setFont("Helvetica", 12)
    canv.drawString(origin_x_pt, origin_y_pt + stripe_height_pt + mm_to_pt(3), label)


def create_pdf(stripe_config=None, output_path: Path = OUTPUT_PDF) -> None:
    palette = getattr(stripe_config, "palette", "green-white")
    use_ticks = bool(getattr(stripe_config, "use_ticks", True))
    tick_mm = float(getattr(stripe_config, "tick_mm", 5.0))

    canv = canvas.Canvas(str(output_path), pagesize=A4)
    width_pt, height_pt = A4

    padding_x_pt, padding_y_pt = mm_list([PADDING_MM, PADDING_MM])
    stripe_gap_pt = mm_to_pt(STRIPE_HEIGHT_MM + LABEL_GAP_MM)

    y = height_pt - padding_y_pt - mm_to_pt(STRIPE_HEIGHT_MM)
    draw_stripe(
        canv,
        padding_x_pt,
        y,
        "black-white",
        "Stripe pattern (black/white)",
        tick_mm,
        use_ticks,
    )

    y -= stripe_gap_pt
    draw_stripe(
        canv,
        padding_x_pt,
        y,
        palette,
        f"Stripe pattern ({palette})",
        tick_mm,
        use_ticks,
    )

    canv.setFont("Helvetica", 10)
    canv.drawString(
        padding_x_pt,
        mm_to_pt(12),
        "Each stripe is 70 mm long. Print at 100% scaling on US Letter or A4.",
    )

    canv.showPage()
    canv.save()


if __name__ == "__main__":  # pragma: no cover
    create_pdf()
