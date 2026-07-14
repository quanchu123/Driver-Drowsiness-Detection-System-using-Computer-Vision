# Driver Drowsiness Detection System using Computer Vision

Hệ thống **phát hiện buồn ngủ của tài xế theo thời gian thực** bằng Computer Vision — monocular camera, không cần cảm biến sinh lý.

| | |
|---|---|
| **Môn / dự án** | DAP391 — Nhóm 3 |
| **Nhóm** | Trần Trọng Chinh (Leader), Trần Quốc Việt, Nguyễn Danh Tám |
| **GVHD** | GS.TS Phan Duy Hùng |
| **Stack** | Python · OpenCV · MediaPipe FaceLandmarker · EAR / MAR / PERCLOS |
| **Repo** | https://github.com/quanchu123/Driver-Drowsiness-Detection-System-using-Computer-Vision |

---

## Demo nhanh

```bash
# 1. Tạo môi trường
python3 -m venv ~/.venvs/drowsiness-dds
source ~/.venvs/drowsiness-dds/bin/activate   # Windows: .venv\Scripts\activate

# 2. Cài dependency
pip install -r Code/requirements.txt

# 3. Chạy realtime (webcam)
cd Code
python -m src.realtime --config configs/default.yaml
```

| Phím | Chức năng |
|------|-----------|
| `q` | Thoát |
| `r` | Reset buffer / state machine |
| `d` | Bật/tắt vẽ landmark |

**Chạy với video file:**

```bash
python -m src.realtime --video path/to/driver.mp4
```

**Chạy nhanh hơn (không vẽ landmark):**

```bash
python -m src.realtime --no-draw
```

> Lần đầu chạy, MediaPipe sẽ tự tải model `face_landmarker.task` (~3.6 MB) vào cache.

---

## Tổng quan hệ thống

Pipeline **geometry-first** (không bắt buộc train deep model), tối ưu cho CPU realtime:

```
Webcam / Video
      │
      ▼
 Downscale (detect_width)  +  preprocess low-light (CLAHE khi tối)
      │
      ▼
 MediaPipe FaceLandmarker (VIDEO mode, tracking)
      │
      ▼
 EAR · MAR · head-nod  →  EMA smooth (giảm jitter)
      │
      ▼
 Temporal window: PERCLOS · blink · microsleep · yawn
      │
      ▼
 Multi-cue drowsiness score  [0 → 1]
      │
      ▼
 State machine: NORMAL → SUSPICIOUS → DROWSY → ALERT
      │
      ▼
 Overlay + cảnh báo (khung đỏ / text)
```

### Các chỉ số chính

| Feature | Ý nghĩa |
|---------|---------|
| **EAR** (Eye Aspect Ratio) | Độ mở mắt — thấp → nhắm / buồn ngủ |
| **MAR** (Mouth Aspect Ratio) | Độ há miệng — cao → ngáp |
| **PERCLOS** | % thời gian mắt đóng trong cửa sổ trượt |
| **Head nod** | Proxy gật đầu (không dùng solvePnP mỗi frame) |
| **Adaptive EAR** | Threshold cá nhân sau ~20 s calibrate |
| **Multi-cue score** | Trộn eye + PERCLOS + microsleep + yawn + nod |

---

## Tính năng & tối ưu

- Realtime webcam / file video
- Không cần GPU; dependency nhẹ (`numpy`, `opencv`, `mediapipe`, `pyyaml`)
- **FPS**: detect ở độ phân giải thấp hơn frame hiển thị (`detect_width: 480`)
- **Low-light**: CLAHE **chỉ khi tối** (không tốn CPU lúc sáng)
- **Giảm false alarm**: state machine + cooldown (không alert vì 1 nháy mắt)
- **Adaptive threshold** theo từng người lái
- Cấu hình tập trung trong `Code/configs/default.yaml`

---

## Cấu trúc repo

```
.
├── Code/                          # Source code chính
│   ├── configs/default.yaml       # Ngưỡng, camera, state machine
│   ├── requirements.txt
│   ├── models/                    # face_landmarker.task (auto-download)
│   ├── src/
│   │   ├── realtime.py            # Entry realtime ⭐
│   │   ├── landmarks.py           # MediaPipe FaceLandmarker
│   │   ├── features.py            # EAR, MAR, PERCLOS, multi-cue score
│   │   ├── preprocess.py          # CLAHE / gamma có điều kiện
│   │   ├── audit_labels.py        # (optional) audit CSV
│   │   ├── train.py               # (optional) train tabular ML
│   │   └── models/classic.py      # LR / RF / XGBoost (optional)
│   └── README.md
├── Data/
│   └── drowsiness_data_shuffled.csv   # EAR, MAR, Drowsy (~8k mẫu)
├── Papers/                        # Tài liệu tham khảo (PDF)
├── Reports/                       # Báo cáo đồ án
├── Slide/                         # Slide thuyết trình
└── docs/
    ├── IMPROVEMENT_PLAN.md        # Kế hoạch cải tiến
    └── RELATED_WORK_2025_2026.md  # Related work paper mới
```

---

## Cài đặt chi tiết

### Yêu cầu

- Python **3.10+**
- Webcam (hoặc file video)
- OS: Linux / Windows / macOS

### Dependency

```bash
pip install -r Code/requirements.txt
```

| Package | Vai trò |
|---------|---------|
| `opencv-python-headless` | Camera, vẽ overlay |
| `mediapipe` | Face landmarks |
| `numpy` | Tính EAR/MAR |
| `pyyaml` | Đọc config |

> Các package train (`scikit-learn`, `xgboost`, …) **không bắt buộc** cho demo realtime. Xem comment trong `Code/requirements.txt`.

---

## Cấu hình (`Code/configs/default.yaml`)

Một số key hay chỉnh:

```yaml
features:
  ear_closed_threshold: 0.22   # EAR mặc định trước khi calibrate
  mar_yawn_threshold: 0.55
  perclos_window: 60           # độ dài cửa sổ temporal (frames)

realtime:
  camera_id: 0
  width: 640
  height: 480
  detect_width: 480            # ↓ nhỏ hơn = FPS cao hơn
  process_every_n: 1           # 2 = xử lý landmark cách frame
  adaptive_calibration_seconds: 20
  alert_score: 0.70            # ngưỡng score để ALERT
  alert_frames: 22             # số frame duy trì trước khi báo
  alert_cooldown_frames: 45
```

---

## Dataset

File `Data/drowsiness_data_shuffled.csv`:

| Cột | Mô tả |
|-----|--------|
| `Image` | Tên ảnh nguồn |
| `EAR` | Eye Aspect Ratio |
| `MAR` | Mouth Aspect Ratio |
| `Drowsy` | Nhãn `0` / `1` |

**Lưu ý:** quan hệ EAR–nhãn trên CSV có thể **không khớp** định nghĩa sinh lý (EAR thấp = mắt nhắm). Pipeline realtime dùng **rule + temporal multi-cue**, không phụ thuộc CSV để demo.

Audit nhãn (tuỳ chọn):

```bash
cd Code
python -m src.audit_labels
```

---

## Optional — train model tabular

Chỉ khi cần so sánh baseline ML trên CSV:

```bash
cd Code
pip install pandas scikit-learn joblib xgboost   # thêm nếu chưa có
python -m src.train --config configs/default.yaml
```

---

## Related work (tóm tắt)

Hệ thống bám hướng **vision-based non-intrusive** phổ biến trong literature:

- EAR / MAR / PERCLOS + adaptive threshold (classical, lightweight)
- Temporal cues (blink, microsleep) để giảm false alarm
- Low-light preprocess (CLAHE) cho lái đêm
- Literature 2025–2026: YOLO-family, Transformer, multimodal — xem [`docs/RELATED_WORK_2025_2026.md`](docs/RELATED_WORK_2025_2026.md)

Papers có sẵn trong `Papers/`.

---

## Hạn chế & hướng phát triển

| Hiện tại | Hướng mở rộng |
|----------|----------------|
| Camera RGB only | IR camera / night enhance sâu hơn |
| Geometry path | Optional YOLOv11n / CNN–LSTM (paper 2025) |
| Single driver face | Multi-face / occlusion (kính đen) |
| Local alert | Log session, tích hợp ADAS |

Chi tiết roadmap: [`docs/IMPROVEMENT_PLAN.md`](docs/IMPROVEMENT_PLAN.md)

---

## Nhóm thực hiện

| Thành viên | Vai trò |
|------------|---------|
| Trần Trọng Chinh | Team leader |
| Trần Quốc Việt | Thành viên |
| Nguyễn Danh Tám | Thành viên |

**Advisor:** GS.TS Phan Duy Hùng

---

## License

Đồ án học thuật — dùng cho mục đích giáo dục / nghiên cứu. Tham khảo paper và dataset gốc khi tái sử dụng.

---

## Liên kết

- **Code guide:** [`Code/README.md`](Code/README.md)
- **Improvement plan:** [`docs/IMPROVEMENT_PLAN.md`](docs/IMPROVEMENT_PLAN.md)
- **Related work 2025–2026:** [`docs/RELATED_WORK_2025_2026.md`](docs/RELATED_WORK_2025_2026.md)
