# Sofrim Stroke Counter

Sofrim (Torah scribes) rely on consistent, precise strokes when writing on parchment. This project provides a computer-vision tool that monitors a scribe's quill in real time, detects the quill tip, and counts writing strokes using a standard webcam.

## Features

- 1080p video capture with OpenCV.
- Homography lock using four ArUco markers (IDs 10–13) to convert between pixel and millimetre coordinates.
- Detection of two coloured rings mounted on the quill to estimate the true tip position.
- Pen-down detection by tracking fresh ink within a millimetre-scale region of interest.
- Automatic stroke counting and a "Refine" counter that tracks quill movement inside a sub-millimetre circle while the quill is down.
- Diagnostics overlays and hotkeys: **R** to re-lock the homography, **S** to capture a screenshot, **ESC** to exit.

## Windows Setup

1. Install Python 3.10 or newer and ensure it is on your PATH.
2. Open **PowerShell** and create a project folder, then clone or download this repository.
3. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
4. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
5. Ensure your webcam supports 1080p at 30 fps and is connected.
6. Place `config.json` next to `sutam_counter.py`. Adjust HSV thresholds if the rings' colours differ under your lighting.

## Printing the ArUco Markers

1. Run the marker generator script:
   ```powershell
   python print_aruco.py
   ```
2. Print the generated `aruco_markers.pdf` at **100% scale** on sturdy paper.
3. Cut out the markers (IDs 10–13 serve as the parchment corners; IDs 14–17 are spares).
4. Mount each marker on a small magnet or clip and position them around the writing area in clockwise order starting from the top-left corner.

## Preparing the Quill

- Attach a narrow **yellow** ring approximately 10 mm above the quill tip (Ring A).
- Attach a **green** ring approximately 20 mm above the tip (Ring B).
- Adjust the HSV ranges in `config.json` if the colours appear washed out or oversaturated.

## Running the Stroke Counter

```powershell
python sutam_counter.py
```

- When the window opens, ensure all four corner markers are visible. The work area rectangle will draw once the homography locks.
- Place the quill so the coloured rings are in view. The application shows the estimated tip, ring detections, the pen-down status, and counters.
- **R** — Force a re-lock of the homography (useful if the parchment shifts or markers were occluded).
- **S** — Save a timestamped screenshot to the `screenshots/` folder for diagnostics.
- **ESC** — Quit the application.

## Troubleshooting

- **Markers not detected**: Improve lighting, ensure markers are not reflective, and keep them flat relative to the parchment.
- **Incorrect scale**: Verify the markers are spaced to match the configured parchment size (defaults to 300×200 mm). Adjust the constants in `sutam_counter.py` if your setup differs.
- **Rings not detected**: Tune the HSV thresholds in `config.json`. Use the screenshot feature to inspect colour balance.
- **No stroke counts**: Ensure the ROI covers the writing tip and that the camera captures enough contrast between fresh ink and parchment. Adjust ambient lighting or pen-down thresholds in the code if necessary.
- **Lag or dropped frames**: Close other applications using the webcam and ensure the camera is set to 1080p30. USB 2.0 hubs may limit bandwidth.

## License

This project is provided "as is" to assist Torah scribes and may be adapted for educational or ritual-support contexts.
