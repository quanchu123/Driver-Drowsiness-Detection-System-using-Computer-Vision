# Related Work 2025–2026 — Driver Drowsiness Detection

Tài liệu cite cho report / slide. Baseline nhóm: MediaPipe + EAR/MAR + Logistic Regression.

---

## Papers 2026

| # | Citation | Đóng góp chính | Ảnh hưởng project |
|---|----------|----------------|-------------------|
| 1 | Ajudiya, C. G., & Panchal, S. S. (2026). *A review on AI-driven drowsiness detection systems using deep-learning.* EPJ Web of Conferences, 348, 04001 (ICMSI 2026). https://doi.org/10.1051/epjconf/202634804001 | Review CNN/LSTM/Transformer; EAR/PERCLOS cổ điển yếu light/occlusion; hybrid multimodal; cần realtime + interpretable | Related work structure; justification temporal + XAI |
| 2 | Hassan, O. F., et al. (2026). *Intelligent driver monitoring systems: a survey of drowsiness detection technologies for road safety.* Artificial Intelligence Review, 59, 137. https://doi.org/10.1007/s10462-026-11505-w | Taxonomy: vision · physio · vehicle · multimodal | Định vị baseline vision-only lightweight |
| 3 | Saxena, S., et al. (2026). *Low-light driver drowsiness detection for real-time safety…* Scientific Reports (attention + XAI). | Realtime low-light + attention + explainability | CLAHE/night path + SHAP/overlay |

## Papers 2025

| # | Citation | Đóng góp chính | Ảnh hưởng project |
|---|----------|----------------|-------------------|
| 4 | Hassan, O. F., Ibrahim, A., & Gomaa, A. (2025). *Real-time driver drowsiness detection using transformer architectures.* Scientific Reports. https://doi.org/10.1038/s41598-025-02111-x | ViT ~99.15% MRL open/close eyes; Swin; CAM | Future work / optional stretch path |
| 5 | Herath, D., Abeyrathne, C., & Jayaweera, P. (2025). *Vision-Based Driver Drowsiness Monitoring: Comparative Analysis of YOLOv5–v11 Models.* arXiv:2509.17498. | YOLOv9c mAP@0.5=0.986; **YOLOv11n** best edge balance; EAR-Dlib yếu pose/occlusion; UTA-RLDD | Prefer YOLOv11n optional detect path |
| 6 | Fonseca, T., et al. (2025). *Drowsiness Detection in Drivers: A Systematic Review…* Applied Sciences. | Systematic review DL contexts & challenges | Related work breadth |
| 7 | Owen, V., et al. (2025). *Computer Vision-Based Drowsiness Detection…* Applied Sciences 15(2), 638. | Higher accuracy + smaller model size | Compact CNN option |

## Papers baseline trong repo (2020–2023)

| File | Nội dung |
|------|----------|
| `Papers/2.pdf` | Elidrissi et al. IJECE 2023 — RF + single-channel EEG |
| `Papers/3.pdf` | Dong et al. Appl. Sci. 2022 — FAN + RF fatigue; CNN distraction |
| `Papers/6.pdf` | Chandiwala & Agarwal ICACCS 2021 — adaptable EAR + smart alarm |
| `Papers/7.pdf` | Sathasivam et al. SCOReD 2020 — EAR + Raspberry Pi ~90% |

## Narrative cho report (copy-ready)

> Traditional geometric cues such as EAR, MAR, and PERCLOS remain widely used for non-intrusive monitoring, but recent surveys (Ajudiya & Panchal, 2026; Hassan et al., 2026) show that fixed thresholds degrade under lighting changes, pose variation, and occlusion. State-of-the-art vision systems combine temporal deep models (CNN–LSTM / Transformer) or one-stage detectors (YOLO family). Herath et al. (2025) report that YOLOv9c reaches mAP@0.5 ≈ 0.986 on UTA-RLDD frames while YOLOv11n offers the best accuracy–latency trade-off for embedded deployment; classical EAR alone is less robust. Transformer approaches (Hassan et al., 2025) achieve very high open/closed-eye accuracy with attention-based explanations, and 2026 work emphasizes low-light robustness and XAI. Our system keeps a lightweight MediaPipe geometry path for real-time CPU use, enriches it with temporal features (PERCLOS, blink, microsleep) and strong tabular ML, and leaves YOLO/Transformer/multimodal fusion as optional upgrades aligned with 2025–2026 literature.
