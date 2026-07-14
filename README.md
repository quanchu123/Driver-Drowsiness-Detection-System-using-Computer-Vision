# Driver Drowsiness Detection System using Computer Vision

Hệ thống phát hiện buồn ngủ tài xế (DAP391 — Nhóm 3) dùng **Computer Vision**.

**Baseline report:** MediaPipe → EAR + MAR + threshold / LR.  
**Code tối ưu:** realtime geometry **không cần train model** — multi-cue (EAR/MAR/PERCLOS/nod) + adaptive threshold + state machine + low-light.

## Cấu trúc

```
.
├── Code/                 # Realtime optimized pipeline
├── Data/                 # CSV (tham khảo)
├── Papers/
├── Reports/
├── Slide/
└── docs/                 # Plan + related work 2025–2026
```

## Quick start (realtime only)

```bash
python3 -m venv ~/.venvs/drowsiness-dds
source ~/.venvs/drowsiness-dds/bin/activate
pip install -r Code/requirements.txt
cd Code
python -m src.realtime --config configs/default.yaml
```

Chi tiết tối ưu: [`Code/README.md`](Code/README.md) · Paper mới: [`docs/RELATED_WORK_2025_2026.md`](docs/RELATED_WORK_2025_2026.md)

## Repo

https://github.com/quanchu123/Driver-Drowsiness-Detection-System-using-Computer-Vision
