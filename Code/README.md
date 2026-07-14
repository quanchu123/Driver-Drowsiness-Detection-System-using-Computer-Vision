# Code — Driver Drowsiness Detection

Pipeline: **MediaPipe FaceMesh → EAR/MAR (+ temporal PERCLOS/blink) → ML (LR/RF/XGBoost) → realtime state machine**.

Aligned with optimization plan 2025–2026 (geometry core first; YOLO/Transformer optional later).

## Setup

```bash
# Prefer venv in home if project drive is noexec:
python3 -m venv ~/.venvs/drowsiness-dds
source ~/.venvs/drowsiness-dds/bin/activate
pip install -r requirements.txt

cd Code
```

## 1) Audit labels

```bash
python -m src.audit_labels
# → artifacts/label_audit.json
```

Raw CSV often has **inverted** EAR↔Drowsy relationship vs physiology.
Default training **flips** labels (`configs/default.yaml` → `data.flip_labels: true`).

> **Important:** After flip, EAR alone almost perfectly separates classes
> (`ear_gt` threshold scan ≈ 100% on raw labels). Labels look **rule-derived from EAR**,
> so train/test F1≈1.0 is **not** real-world proof. Always validate with webcam / public video.

## 2) Train

```bash
python -m src.train --config configs/default.yaml
# models: --model logistic | rf | xgboost
```

Outputs:
- `artifacts/best_model.joblib`
- `artifacts/metrics.json`
- `artifacts/label_audit.json`

## 3) Realtime demo

```bash
python -m src.realtime --config configs/default.yaml
# or video file:
python -m src.realtime --video path/to/video.mp4
# headless smoke:
python -m src.realtime --no-display --max-frames 30 --video path.mp4
```

Keys: `q` quit · `r` reset buffer.

## Layout

```
src/
  landmarks.py      MediaPipe FaceMesh
  features.py       EAR, MAR, PERCLOS, blink, head pose
  preprocess.py     CLAHE / low-light
  dataset.py        load + stratified split
  audit_labels.py   label inversion check
  models/classic.py LR, RF, XGBoost
  train.py          train + metrics
  evaluate.py       metrics helpers
  realtime.py       webcam + state machine
configs/default.yaml
```

## Config notes

| Key | Meaning |
|-----|---------|
| `data.flip_labels` | Invert Drowsy after audit |
| `features.ear_closed_threshold` | Realtime eye-closed EAR |
| `realtime.suspicious_frames` / `drowsy_frames` | State machine hysteresis |
| `realtime.adaptive_calibration_seconds` | Personal EAR baseline |

## Next (Phase 2+)

- BiLSTM/TCN sequence models
- ONNX export
- Optional YOLOv11n path
- SHAP / Grad-CAM XAI
