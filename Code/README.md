# Code — Driver Drowsiness Detection (optimized, no train)

Pipeline realtime **không train model**:

```
Camera → downscale → night preprocess → MediaPipe FaceLandmarker (VIDEO)
      → EMA(EAR/MAR/nod) → PERCLOS/blink/microsleep
      → multi-cue score → state machine alert
```

## Setup

```bash
python3 -m venv ~/.venvs/drowsiness-dds
source ~/.venvs/drowsiness-dds/bin/activate
pip install -r requirements.txt   # numpy opencv mediapipe pyyaml (không cần xgboost)
cd Code
```

## Chạy realtime

```bash
python -m src.realtime --config configs/default.yaml
# video file:
python -m src.realtime --video path/to/clip.mp4
# nhanh hơn (không vẽ landmark):
python -m src.realtime --no-draw
```

Phím: `q` thoát · `r` reset buffer · `d` bật/tắt vẽ landmark.

## Tối ưu đã áp dụng

| Mục | Cách |
|-----|------|
| FPS | `detect_width: 480`, VIDEO mode tracking, sparse landmark draw |
| Night | CLAHE **chỉ khi tối** (không CLAHE mọi frame) |
| Jitter | EMA smooth EAR/MAR |
| Adaptive EAR | calibrate 20s → threshold theo người |
| Multi-cue | score = eye + PERCLOS + microsleep + yawn + head-nod |
| False alarm | state machine + cooldown (không alert 1 blink) |
| CPU feature | EAR/MAR float thuần, không solvePnP mỗi frame |

Tune trong `configs/default.yaml` → block `realtime` / `features`.

## Cấu trúc

```
src/
  realtime.py      entry realtime (chính)
  landmarks.py     FaceLandmarker VIDEO
  features.py      EAR/MAR/PERCLOS/score
  preprocess.py    CLAHE có điều kiện
configs/default.yaml
```

Script `train.py` / `audit_labels.py` vẫn nằm trong repo nếu cần phân tích CSV sau — **không bắt buộc** cho demo.
