# Sofrim Stroke Counter

This project helps Torah scribes (sofrim) count quill strokes in real time. A Windows PC with a standard 1080p webcam watches the parchment, locks onto four ArUco markers, finds coloured rings on the quill, and measures pen-down activity to tally strokes.

## Windows Setup (Python 3.12)

Open **PowerShell** in the project folder and run:

```powershell
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python sutam_counter.py
```

The script opens two windows: the main viewer and an HSV calibration panel. If you see an error that `cv2.aruco` is missing, install `opencv-contrib-python` as noted in `requirements.txt`.

## Printing the Markers

1. Generate the PDF:
   ```powershell
   python print_aruco.py
   ```
2. Print `aruco_markers.pdf` at 100 % scale. Each marker (IDs 10–17) is 10×10 mm. Place IDs 10–13 at the parchment corners, clockwise starting at the top-left. The remaining IDs are spares.
3. Fix the printed markers to magnets or clips so they stay flat and do not move during writing.

## Preparing the Quill

- Add a **yellow** ring roughly 10 mm above the quill tip (Ring A).
- Add a **green** ring roughly 20 mm above the tip (Ring B).
- Ensure both rings remain in the camera view while writing. Adjust lighting to keep colours vibrant.

## Operating the Stroke Counter

- The camera is forced to capture at 1920×1080 @ 30 fps. Position the parchment so all four corner markers are visible.
- Once the homography locks, the software converts between pixels and millimetres. The overlay shows the work-area rectangle, ring detections, the estimated quill tip, and two counters.
- Pen-down detection uses frame differencing in a tiny region around the tip, with a smoothed signal to suppress noise. The refine counter measures each entry or exit through a 0.8 mm circle while the pen is down.
- Keyboard controls:
  - **R** — Re-lock (re-detect the ArUco markers and recompute the homography).
  - **S** — Save a diagnostic screenshot in the `screenshots/` folder.
  - **ESC** — Exit the application.

## Troubleshooting

- **Markers lost or warped:** Improve lighting, keep markers flat, and ensure the webcam is orthogonal to the parchment. Press **R** to re-lock if the parchment shifts.
- **Rings not detected:** Use the HSV calibration window to adjust the low/high thresholds for each ring. The values are saved back to `config.json` automatically.
- **No stroke counts:** Confirm the quill tip falls within the ROI overlay. Increase contrast between fresh ink and parchment or adjust room lighting.
- **`cv2.aruco` import error:** Install `opencv-contrib-python==4.12.0.88` inside the virtual environment.

## License

Provided for ritual-support and educational use; adapt it as needed for your own sofrut workflow.
