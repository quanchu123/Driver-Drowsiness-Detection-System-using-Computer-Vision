# Kế hoạch cải tiến & tối ưu — Driver Drowsiness Detection System

**Repo:** https://github.com/quanchu123/Driver-Drowsiness-Detection-System-using-Computer-Vision  
**Nhóm:** Trần Trọng Chinh, Trần Quốc Việt, Nguyễn Danh Tám  
**Baseline (theo Report Part1):** MediaPipe FaceMesh → EAR + MAR → Logistic Regression / threshold rules  
**Ngày lập kế hoạch:** 2026-07-14

---

## 0. Hiện trạng hệ thống (baseline)

### 0.1. Pipeline hiện tại (từ report)

| Bước | Kỹ thuật | Ghi chú |
|------|----------|---------|
| Face landmarks | MediaPipe FaceMesh | Nhẹ, real-time, không cần GPU |
| Feature | EAR, MAR | Hình học, dễ giải thích |
| Label | 0 = awake, 1 = drowsy | Từ ảnh Kaggle |
| Classifier | Logistic Regression + rule (EAR < 0.41, MAR > 0.9, yawn ≥ 3/phút) | Report mâu thuẫn: vừa threshold vừa ML |
| Runtime | OpenCV + webcam loop | Real-time target |

### 0.2. Dữ liệu hiện có

| Item | Giá trị |
|------|---------|
| File | `Data/drowsiness_data_shuffled.csv` |
| Samples | 8,110 |
| Features | `Image`, `EAR`, `MAR`, `Drowsy` |
| Class balance | Drowsy=1: **5,001** (61.7%) · Awake=0: **3,109** (38.3%) |
| EAR range | ~0.32 – 0.59 (mean 0.416) |
| MAR range | ~0.40 – 1.11 (mean 0.588) |

### 0.3. Lỗ hổng kỹ thuật nghiêm trọng (audit)

1. **`Code/` trống** — không tái lập được experiment, không có train/eval script, không demo real-time.
2. **Nhãn có dấu hiệu đảo chiều so với định nghĩa sinh lý EAR:**
   - Report: EAR **thấp** → mắt nhắm → drowsy.
   - CSV thực tế: class `Drowsy=1` có **EAR mean cao hơn** (0.442 vs 0.376).
   - `corr(EAR, Drowsy) ≈ +0.79` (lẽ ra phải **âm**).
   - Rule `EAR > 0.41` dự đoán class 1 đạt ~**92% accuracy** — ngược hoàn toàn với rule trong report (`EAR < 0.41`).
   - → **Ưu tiên #0:** audit lại cách gán nhãn / công thức EAR trước khi train model mới.
3. **Feature quá mỏng** — chỉ 2 số (EAR, MAR) per frame; mất temporal context (PERCLOS, blink rate, microsleep).
4. **Không có split train/val/test chuẩn**, không report metrics số (chỉ “Figure 2”).
5. **Papers tham chiếu cũ** (2020–2023), thiếu SOTA vision 2025–2026 (đã bổ sung `docs/RELATED_WORK_2025_2026.md`).
6. **CSV labels có vẻ rule-based theo EAR** (tách gần hoàn hảo bằng threshold) → metric 100% trên hold-out **không** phản ánh hiệu năng thực tế; cần eval webcam / public video.
6. **Không robust** với kính, ánh sáng yếu, mặt nghiêng, che mặt, đa chủng tộc, camera rung.

---

## 1. Landscape kỹ thuật mới nhất (2024–2025)

### 1.1. Hướng tiếp cận chính trong literature

```
                    Driver State Monitoring
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   Visual (camera)      Vehicle behavior      Physiological
   face/eye/mouth/      steering, lane,       EEG, ECG, EOG
   head pose            pedal, CAN bus        (intrusive)
        │
        ├── Classical geometry: EAR, MAR, PERCLOS, blink
        ├── CNN / YOLO end-to-end eye-state
        ├── Temporal: LSTM / BiLSTM / TCN / Transformer
        └── Multimodal fusion + edge deploy (TFLite/ONNX)
```

### 1.2. Kỹ thuật đáng áp dụng (ưu tiên theo ROI cho đồ án)

| # | Kỹ thuật | Paper / nguồn tiêu biểu | Lợi ích | Độ khó | Ưu tiên |
|---|----------|-------------------------|---------|--------|---------|
| A | **PERCLOS + blink rate + temporal window** | IEEE classic + surveys 2024–25 | Bắt microsleep, giảm false alarm | Thấp | **P0** |
| B | **Head pose (pitch/yaw/roll)** | FAN / MediaPipe / 6DoF | Bổ sung gục đầu | Thấp–TB | **P0** |
| C | **Multi-feature ML** (RF, XGBoost, LightGBM) | RF fatigue systems (MDPI 2022+) | Tận dụng CSV hiện tại, dễ so baseline LR | Thấp | **P0** |
| D | **Adaptive threshold / per-user calibration** | ICACCS 2021 (adaptable EAR) | Giảm bias người/kính | TB | **P1** |
| E | **YOLOv8/v11 eye–mouth–yawn detection** | Multiple 2024–25 DDS papers | Robust hơn landmark khi occlusion | TB–Cao | **P1** |
| F | **CNN + BiLSTM / TCN temporal** | Sensors 2025 hybrid reviews | Sequence modeling, SOTA accuracy | Cao | **P1** |
| G | **Night / low-light enhancement** | CLAHE, Zero-DCE, low-light CNN | Thực tế lái đêm | TB | **P1** |
| H | **Distraction + drowsiness multi-task** | Appl. Sci. RF+CNN 2022 | Mở rộng use-case | Cao | **P2** |
| I | **Vision Transformer / hybrid ViT** | ViT drowsiness papers 2023–25 | Accuracy cao, nặng | Cao | **P2** |
| J | **Edge optimize** (ONNX, TFLite, INT8) | Jetson / RPi deploy papers | Real-time on device | TB | **P1** |
| K | **Multimodal** (vision + steering/CAN hoặc PPG) | Multimodal surveys 2024–25 | Accuracy cao nhất | Rất cao | **P3** |

### 1.3. Dataset chuẩn nên dùng (ngoài Kaggle hiện tại)

| Dataset | Nội dung | Ghi chú |
|---------|----------|---------|
| **NTHU-DDD** | Video drowsiness đa điều kiện (ngày/đêm, kính) | Benchmark phổ biến |
| **YawDD** | Yawning detection | Tốt cho MAR / yawn branch |
| **UTA-RLDD** | Real-life drowsiness | Gần thực tế hơn lab |
| **DROZY** | Video + physiological | Multimodal research |
| Custom webcam | Team tự thu (có consent) | Domain adaptation VN lighting |

### 1.4. Metrics bắt buộc (thay vì chỉ accuracy)

- Accuracy, Precision, Recall, F1, **AUC-ROC**, **AUC-PR**
- **Confusion matrix**, per-class report
- **False Positive Rate** (quan trọng: tránh báo ồn khi lái)
- **Detection latency** (ms/frame), FPS
- **Time-to-alert** (từ onset drowsiness → warning)
- Cross-condition: glasses / night / head pose

---

## 2. Mục tiêu cải tiến (SMART)

| Mục tiêu | Metric | Baseline (ước) | Target |
|----------|--------|----------------|--------|
| Sửa data + tái lập code | Reproducible pipeline | Code trống | Train/eval/demo chạy 1 lệnh |
| Chất lượng nhãn | Consistency EAR↔label | corr ≈ +0.79 (sai chiều) | corr EAR–drowsy **âm**, audit report |
| Classification (frame-level) | F1 / AUC | Chưa có số chuẩn | F1 ≥ 0.90, AUC ≥ 0.95 trên hold-out |
| Temporal drowsiness | F1 event-level | N/A | F1 ≥ 0.85, FPR alert ≤ 5%/giờ lái giả lập |
| Real-time | FPS (CPU laptop) | Unknown | ≥ 20 FPS CPU, ≥ 30 FPS GPU |
| Robustness | Acc drop night/glasses | Unknown | Drop ≤ 8% so với điều kiện chuẩn |

---

## 3. Roadmap theo phase

### Phase 0 — Audit & nền tảng (3–5 ngày) — **BẮT BUỘC**

**Mục tiêu:** Có codebase chạy được + data sạch.

1. **Viết lại source code** trong `Code/`:
   ```
   Code/
   ├── requirements.txt
   ├── configs/default.yaml
   ├── src/
   │   ├── landmarks.py      # MediaPipe FaceMesh
   │   ├── features.py       # EAR, MAR, PERCLOS, blink, head pose
   │   ├── dataset.py
   │   ├── models/           # LR, RF, XGB, temporal
   │   ├── train.py
   │   ├── evaluate.py
   │   └── realtime.py       # webcam demo + alarm
   ├── notebooks/eda.ipynb
   └── scripts/extract_features.py
   ```
2. **Audit nhãn CSV:**
   - Recompute EAR/MAR từ ảnh gốc (nếu còn) bằng cùng công thức.
   - So khớp với `Drowsy`; nếu nhãn đảo → flip + document.
   - Stratified train/val/test (70/15/15), seed cố định.
3. **Baseline chính thức:**
   - Threshold rules (đúng chiều sinh lý).
   - Logistic Regression, Random Forest trên (EAR, MAR).
   - Báo cáo metrics đầy đủ + confusion matrix.
4. **Unit tests** cho EAR/MAR (synthetic eye open/closed).

**Deliverable:** PR `phase-0-repro-baseline`, README chạy demo.

---

### Phase 1 — Feature engineering & classical ML mạnh (1 tuần) — **ROI cao nhất**

**Mục tiêu:** Vượt baseline rõ ràng **không cần deep model nặng**.

#### 1.1. Feature set mới (per frame + sliding window 30–90s)

| Feature | Ý nghĩa | Công thức / nguồn |
|---------|---------|-------------------|
| EAR_L, EAR_R, EAR_mean | Độ mở mắt | MediaPipe eye landmarks |
| MAR | Há miệng / ngáp | Mouth landmarks |
| **PERCLOS** | % thời gian mắt đóng trong cửa sổ T | ratio EAR < thr over window |
| **Blink rate / duration / amplitude** | Tần suất chớp mắt | peak detection trên EAR |
| **Microsleep count** | EAR thấp liên tục ≥ 0.5–1s | temporal state machine |
| **Yawn count / duration** | MAR > thr liên tục | temporal |
| **Head pose** pitch/yaw/roll | Gục đầu | solvePnP / MediaPipe |
| EAR velocity, MAR velocity | Động học | Δfeature/Δt |
| Rolling stats | mean/std/min EAR, MAR | window |

#### 1.2. Models so sánh

- Logistic Regression (baseline)
- Random Forest / ExtraTrees
- **XGBoost / LightGBM** (thường win trên tabular)
- Optional: small MLP

#### 1.3. Training hygiene

- Class weight / SMOTE **chỉ trên train**
- Nested CV hoặc fixed split + bootstrap CI
- Feature importance (SHAP) cho report

**Deliverable:** bảng so sánh models + notebook EDA + realtime dùng model tốt nhất.

---

### Phase 2 — Temporal deep learning (1.5–2 tuần)

**Mục tiêu:** Bắt pattern theo thời gian (microsleep, nhịp chớp).

#### 2.1. Kiến trúc đề xuất (từ nhẹ → nặng)

```
Option A (khuyến nghị đồ án): Feature sequence → TCN / BiLSTM
  [EAR, MAR, pose, ...]_t=1..T  →  BiLSTM → Dense → Softmax

Option B: CNN backbone (face crop) → BiLSTM/GRU
  frame sequence  → MobileNetV3/EfficientNet-Lite → temporal head

Option C (SOTA paper-style): Hybrid CNN + Transformer encoder
  chỉ khi A/B chưa đủ accuracy
```

#### 2.2. Training tips

- Sequence length: 30–90 frames (~1–3s @ 30fps) cho microsleep; window dài hơn cho PERCLOS state.
- Label strategy: **majority vote** hoặc **end-of-window state**; tránh label leakage.
- Augmentation video: brightness, blur, horizontal flip (cẩn thận landmarks), synthetic night.
- Export **ONNX** cho realtime.

**Deliverable:** model temporal + so sánh vs Phase 1; latency report.

---

### Phase 3 — Detection backbone hiện đại (song song / sau Phase 1)

**Mục tiêu:** Robust khi MediaPipe fail (góc nghiêng, che, low-res).

1. **YOLOv8n/s** fine-tune:
   - Classes: `face`, `eye_open`, `eye_closed`, `yawn`, `no_yawn` (tùy schema).
   - Hoặc detect face → crop → classifier mắt/miệng.
2. Fallback cascade:
   ```
   if MediaPipe confidence high → geometry features
   else → YOLO eye-state branch
   ```
3. Night path: CLAHE + gamma; optional low-light enhance trước detect.

**Deliverable:** dual-path pipeline + ablation “geometry only / YOLO only / fusion”.

---

### Phase 4 — Hệ thống real-time & UX an toàn (3–5 ngày)

1. **State machine cảnh báo** (giảm false alarm):
   ```
   NORMAL → SUSPICIOUS (EAR low N frames) → DROWSY (confirm M frames / PERCLOS)
            → ALERT (sound + visual) → cooldown
   ```
2. **Adaptive calibration 30–60s** lúc bắt đầu lái: ước lượng EAR baseline cá nhân.
3. Alarm: beep + overlay; optional gesture dismiss (theo paper adaptable EAR).
4. Logging session (CSV/JSON) để offline analyze.
5. Config YAML: thresholds, camera id, model path.

**Deliverable:** `python -m src.realtime --config configs/default.yaml`

---

### Phase 5 — Evaluation chuẩn academic + polish (3–5 ngày)

1. Benchmark trên **NTHU-DDD / YawDD** (ít nhất 1 public set) + custom set.
2. Cross-condition table: day/night, glasses, head pose.
3. Latency: CPU vs GPU, model size.
4. Update Report Part2 + Slide: related work 2024–25, ablation, limitations.
5. Demo video 2–3 phút cho defense.

---

## 4. Kiến trúc hệ thống đề xuất (target)

```
                    Webcam / Video
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
     Light preprocess            (optional)
     CLAHE / resize              YOLO face/eye
              │                       │
              └───────────┬───────────┘
                          ▼
              MediaPipe FaceMesh
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
        EAR/MAR      Head pose       Face crop
          │               │               │
          └───────┬───────┘               │
                  ▼                       ▼
         Temporal buffer (window)    CNN encoder
                  │                       │
                  ▼                       ▼
         Tabular / sequence feats → Fusion head
                  │
                  ▼
         State machine (NORMAL/DROWSY)
                  │
          ┌───────┴───────┐
          ▼               ▼
        Alarm UI      Session log
```

**Nguyên tắc:** giữ **geometry path** làm default (nhẹ, explainable); deep path là upgrade có đo lường.

---

## 5. Ưu tiên implement (checklist thực dụng)

### Tuần 1
- [ ] Khôi phục / viết `Code/` đầy đủ
- [ ] Fix audit nhãn EAR↔Drowsy
- [ ] Baseline LR + RF + metrics
- [ ] Realtime demo EAR/MAR overlay

### Tuần 2
- [ ] PERCLOS, blink, head pose, window features
- [ ] XGBoost/LightGBM + SHAP
- [ ] Adaptive threshold
- [ ] State machine alarm

### Tuần 3
- [ ] BiLSTM/TCN trên feature sequence
- [ ] ONNX export + FPS benchmark
- [ ] Low-light preprocess

### Tuần 4
- [ ] YOLO path (optional nếu còn bandwidth)
- [ ] Eval public dataset
- [ ] Report + slide + demo video

---

## 6. Rủi ro & cách giảm

| Rủi ro | Impact | Mitigation |
|--------|--------|------------|
| Nhãn sai / đảo | Model “học ngược” | Phase 0 audit; re-label sample; rule-based sanity |
| Overfit Kaggle | Acc ảo | Public benchmark + real webcam test |
| False alarm cao | UX tệ, user tắt hệ thống | State machine + PERCLOS, không alert 1 frame |
| MediaPipe fail (đêm/góc) | Miss detection | CLAHE + YOLO fallback |
| Scope creep (ViT, multimodal) | Không kịp deadline | Giữ P0–P1; P2 chỉ khi P1 xong |
| Privacy camera | Ethics | Consent, no cloud upload default, local process |

---

## 7. Stack đề xuất

| Thành phần | Tool |
|------------|------|
| Language | Python 3.10+ |
| Vision | OpenCV, MediaPipe |
| ML | scikit-learn, XGBoost/LightGBM |
| DL (phase 2+) | PyTorch |
| Detect (phase 3) | Ultralytics YOLOv8n |
| Config | YAML + pydantic |
| Export | ONNX Runtime |
| Logging | loguru / CSV session |
| UI demo | OpenCV imshow (+ optional Streamlit dashboard) |

---

## 8. Related work gợi ý cite (cập nhật report)

**Baseline / classical (đã có trong folder Papers):**
1. Elidrissi et al., IJECE 2023 — RF + single-channel EEG (98.5%, intrusive).
2. Dong et al., Appl. Sci. 2022 — S3FD + FAN + RF fatigue; CNN distraction.
3. Chandiwala & Agarwal, ICACCS 2021 — Adaptable EAR + smart alarm.
4. Sathasivam et al., SCOReD 2020 — EAR + Raspberry Pi (~90%).

**Nên bổ sung (2023–2025):**
5. Surveys on vision-based driver monitoring (Sensors / IEEE Access 2024–2025) — taxonomy visual vs bio vs vehicle.
6. Hybrid **CNN–LSTM / BiLSTM** drowsiness systems (nhiều paper 2024–25, acc thường >95% trên lab sets).
7. **YOLO-family** real-time eye/yawn detection for DDS.
8. **PERCLOS** as gold behavioral metric for drowsiness (classic + modern re-use).
9. Edge deployment papers (TFLite / Jetson) for in-cabin systems.
10. Multimodal driver state (vision + vehicle signals) for future work section.

---

## 9. Định hướng “câu chuyện” cho báo cáo / defense

1. **Problem:** drowsiness → tai nạn; cần non-intrusive, real-time, cheap hardware.
2. **Baseline:** MediaPipe + EAR/MAR + LR — lightweight nhưng feature/temporal yếu, data audit issue.
3. **Insight từ paper mới:** temporal + multi-cue (PERCLOS, head pose) + robust detect quan trọng hơn “model lớn”.
4. **Contribution nhóm:**
   - Pipeline tái lập + data audit
   - Multi-feature + temporal ML/DL
   - Real-time state machine + adaptive threshold
   - Ablation & public-data eval
5. **Limitation:** camera only; chưa multimodal; lab/Kaggle ≠ highway night VN.
6. **Future:** YOLO fusion, night enhance, CAN-bus / PPG, federated personalization.

---

## 10. Kết luận ưu tiên

| Hạng | Việc | Lý do |
|------|------|-------|
| **#1** | Audit nhãn + viết lại Code | Không làm thì mọi model đều vô nghĩa |
| **#2** | PERCLOS + blink + head pose + XGBoost | Gain lớn, effort vừa, đúng SOTA “practical” |
| **#3** | Temporal BiLSTM/TCN + state machine | Microsleep & giảm false alarm |
| **#4** | YOLO + low-light + ONNX | Robust & deploy |
| **#5** | ViT / multimodal | Research stretch, sau khi core ổn |

> **Tóm tắt một câu:** Đừng nhảy ngay sang ViT; **sửa data → làm giàu feature temporal → model tabular/sequence nhẹ → realtime state machine**, rồi mới cân YOLO/deep fusion — đúng cả paper mới và ràng buộc đồ án.
