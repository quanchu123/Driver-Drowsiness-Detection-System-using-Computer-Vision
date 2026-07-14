# Driver Drowsiness Detection System using Computer Vision

Hệ thống phát hiện buồn ngủ tài xế (DAP391 — Nhóm 3) dùng **Computer Vision**.

**Baseline report:** MediaPipe FaceMesh → EAR + MAR → Logistic Regression / threshold rules.  
**Code tối ưu (2026):** geometry + temporal features + XGBoost/RF + realtime state machine, bám literature 2025–2026.

## Cấu trúc

```
.
├── Code/                 # Source (train / audit / realtime)
├── Data/                 # drowsiness_data_shuffled.csv
├── Papers/               # Tài liệu tham khảo (2020–2023)
├── Reports/              # Báo cáo
├── Slide/
└── docs/
    ├── IMPROVEMENT_PLAN.md
    └── RELATED_WORK_2025_2026.md
```

## Quick start

```bash
python3 -m venv ~/.venvs/drowsiness-dds
source ~/.venvs/drowsiness-dds/bin/activate
pip install -r Code/requirements.txt

cd Code
python -m src.audit_labels
python -m src.train --config configs/default.yaml
python -m src.realtime --config configs/default.yaml
```

Xem chi tiết: [`Code/README.md`](Code/README.md).

## Lưu ý data

Audit cho thấy CSV gốc có quan hệ EAR–Drowsy **đảo chiều** so với định nghĩa sinh lý (EAR thấp = mắt nhắm).  
Train mặc định **`flip_labels: true`** — xem `artifacts/label_audit.json`.

## Related work mới

Xem [`docs/RELATED_WORK_2025_2026.md`](docs/RELATED_WORK_2025_2026.md) (YOLO v11, Transformer, low-light XAI, surveys 2026).

## Repo

https://github.com/quanchu123/Driver-Drowsiness-Detection-System-using-Computer-Vision
