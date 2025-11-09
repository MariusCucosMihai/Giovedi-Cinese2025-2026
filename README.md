# Giovedi-Cinese2025-2026

Rescue maze vision tooling for detecting letter victims (H/S/U) via OpenCV template matching and color victims (red/yellow/green squares) via HSV segmentation.

## Vision (Template Matching)

- Entry: `vision_detect.py`
- Modes:
  - Camera: `python vision_detect.py --mode camera`
  - Images: `python vision_detect.py --mode images --input samples --demo-samples --output-dir out`
- Options:
  - `--letter-threshold 0.62` adjust matching strictness
  - `--headless` disable display windows

## Build Windows EXE

Requires Python and PyInstaller installed.

PowerShell:

```
./build_exe.ps1 -Install
./build_exe.ps1
```

The executable appears at `dist/RescueVision.exe`.

## Legacy/Training

- `createDataset.py` generates synthetic letter images (S/H/U) for CNN training.
- `train_letters.py` trains a simple CNN and saves `letter_model_v2.h5`.
- `detect_letters_tf_final.py` loads `letter_model_v2.h5` for TF-based detection (alternative to template matching).

Note: `rescue_vision.py` uses Tesseract OCR and may be considered legacy compared to `vision_detect.py`.
