# Sofrim Stroke Counter

A single-window Tkinter application that tracks a Torah scribe's quill using
OpenCV and ArUco markers.  The preferred **Stripe Mode** locks onto a coloured
stripe glued along the quill and estimates the writing tip using a metric
homography.  **Rings Mode** is available as a fallback when the stripe is not
present.

## Features

- One window that displays the live video feed with overlays.
- Menu bar with actions for starting/stopping capture, taking screenshots,
  configuring colours, selecting cameras and manual tip alignment.
- Stripe Mode with RANSAC-assisted axis estimation and exponential smoothing.
- Rings Mode that tracks two coloured bands near the tip.
- Ink activity tracking and stroke/refine counters.
- Built-in PDF generators for stripe patterns and ArUco marker sheets.
- Configuration saved in `config.json` and optionally overridden with
  `--mode stripe|rings`.

## Requirements

- Windows 10 or later (the code is cross-platform but tuned for Windows).
- Python 3.12 (64-bit recommended).
- A webcam capable of 1080p @ 30fps.
- The packages listed in `requirements.txt`.

Install dependencies in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Running the application

```powershell
python sutam_counter.py          # uses mode from config.json
python sutam_counter.py --mode rings  # override detection mode
```

The application prefers external USB cameras automatically.  You can manually
choose a camera from **Tools → Select camera…**.

## Configuration

The default `config.json` contains reasonable HSV thresholds for both modes.
Use **Tools → Color calibration…** to tweak the ranges interactively and save
back to disk.  Manual tip alignment writes an additional offset in millimetres
for the active mode.

Key parameters:

- `stripe.tip_offset_mm`: distance from the detected far end of the stripe to
  the actual tip.
- `rings.tip_distance_mm`: offset from the lower ring to the real tip.
- `manual_offset_mm`: manual adjustment stored by **Manual tip set…**.

## Printing utilities

Generate helper printouts from the **Print** menu or by running the modules:

```powershell
python print_stripe.py
python print_aruco.py
```

- `stripe_pattern.pdf`: includes a 70 mm stripe with black/white and colour
  variants and optional tick marks every 5 mm.
- `aruco_markers.pdf`: includes ArUco IDs 10–17 arranged in two rows of four,
  each marker 10×10 mm with labels.

Print at 100% scaling on A4 or US Letter paper.  The metric coordinates are
used to compute the homography that converts pixels to millimetres.

## Troubleshooting

- **No camera detected** – ensure the camera is connected and not in use by
  another application.  The app probes DirectShow, Media Foundation and the
  default OpenCV backend.
- **Markers not recognised** – verify that the PDF was printed without scaling
  and that the markers are well lit.
- **Stripe not detected** – re-run colour calibration and check that the
  pattern occupies at least 500 pixels in the image.
- **Stroke counter erratic** – adjust the lighting and make sure the tip ROI is
  not obstructed.  Manual tip alignment can also help.

## License

MIT License.  See `LICENSE` if provided by the repository owner.
