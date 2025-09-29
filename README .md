# Smart Baby Monitor using Visual Language Models (VLM)

## Motivation
Traditional baby monitors stream audio/video but offer little semantic understanding. Parents must constantly watch or listen.

A lightweight VLM system can recognize common infant states (awake/asleep, crying, playing) and summarize them in natural language with simple alerts, reducing caregiver cognitive load.

**Research relevance:** applies VLMs to fine-grained activity recognition on short, low-resolution clips; explores multi-modal fusion (vision + audio); evaluates on-device vs cloud tradeoffs.

---

## Design Goals
- **Accurate core states:** classify at least 4 states—awake, asleep, crying, caregiver present.  
- **Low latency:** <1.5 s end-to-end on 720p @ 1–2 fps analysis.  
- **Privacy-aware:** optional on-prem inference; no raw video leaves device when “local mode” is on.  
- **Explainable output:** short NL summaries (“The baby is asleep; eyes closed, minimal motion for 2 min.”).  
- **Robustness:** works day/night (IR), handles occlusion, and background noise.  
- **Easy deploy:** one-command container + dashboard; fallback to Colab if no CUDA GPU.  

---

## Deliverables
- **Code:** modular pipeline (capture → preprocessing → VLM inference → rule/ML head → summary/alerts), with CLI and a minimal web dashboard.  
- **Trained/finetuned heads:** small classifier on top of VLM embeddings for the 4 target states.  
- **Demo video:** 2–3 min live run with scripted scenarios.  
- **Short report (4–6 pages):** problem, methods, design decisions, experiments, ablations, limitations, ethics.  
- **Reproducibility:** Dockerfile + Colab notebook; config files; sample test clips and evaluation scripts.  

---

## System Blocks (Dataflow)

[IP/USB Camera] --RTSP/USB--> [Capture Node]
| |
v v
[Frame Sampler (1-2 fps)] --> [Preprocess: resize, normalize, face/pose optional]
|
v
[VLM Encoder (e.g., LLaVA/BLIP-2)]
|
+-----------------+-----------------+
| |
[Linear/GRU Head] [Prompted NL Head]
(state logits) (“Describe baby state”)
| |
+-----------[Fusion + Rules]--------+
|
[Event Logic & Smoothing]
|
+---------------+------------------+
| |
[Alerts (MQTT/email/push)] [Dashboard + Log/CSV]


---

## Hardware / Software Requirements
**Hardware (any one path):**
- Local: Laptop with CUDA-enabled GPU (≥6 GB VRAM preferred) + USB webcam or RTSP baby-cam.  
- Cloud fallback: Google Colab Pro (T4/A100 as available).  
- Optional: ESP32-S3 mic module for cry-audio capture (stretch).  

**Software stack:**
- Python 3.10+, PyTorch 2.x, torchvision  
- VLM: LLaVA-1.5 or BLIP-2/Flan-T5 (open-weights options), Transformers + accelerate  
- Optional audio: open-source VAD + simple cry classifier (logistic/GRU on MFCCs)  
- OpenCV (capture), FFmpeg (RTSP), FastAPI, Uvicorn, WebSockets  
- MQTT broker (Mosquitto) or SMTP for alerts  
- Frontend: simple HTML/JS dashboard (charts + thumbnails)  
- DevOps: Docker, docker-compose; Colab notebook; Weights & Biases or TensorBoard  

---

## Dataset & Ground Truth
- Use short, ethically sourced baby-like clips (dolls/YouTube infant proxies) plus staged clips.  
- Label 5–10 s windows with one state. Aim for ~30–60 min labeled total.  
- Split 70/15/15 train/val/test. Keep a held-out “night mode” subset.  

---

## Methods
- **Vision:** Extract VLM image embeddings per frame; prompt VLM with structured template.  
- **Head:** Train a small linear/GRU head using VLM embeddings + motion features.  
- **Fusion:** Combine vision logits with audio cry probability; apply hysteresis & smoothing.  
- **Summaries:** Template + VLM rephrase for a 1-sentence explanation.  
- **Alerts:** Rule engine (e.g., crying >5s → push).  

---

## Evaluation
- Accuracy/F1 per class; latency (p95), false-alert rate/hour.  
- Ablations: VLM-only vs VLM+audio; with/without motion features; prompt-only vs learned head.  

---

## Team Members & Responsibilities
**Quan Minh (Q), Franck Gabriel (F), Jefferey Kidwell (J).**  
Everyone contributes to testing and the final demo.

| Workstream         | Lead (Role)             | Support |
|--------------------|-------------------------|---------|
| Setup/Infra        | F (Setup)               | Q       |
| Software Platform  | Q (Software)            | J       |
| Networking         | J (Networking)          | F       |
| Research           | Q (Research)            | F       |
| Algorithm Design   | F (Algorithm Design)    | J       |
| Writing            | J (Writing)             | Q       |
| Data/Labeling      | All (Quan coord.)       | —       |
| QA & Demo          | All                     | —       |

---

## Project Timeline (6 Weeks)
- **Week 1 – Planning & Setup**: Repo, Docker, Colab, literature skim.  
- **Week 2 – Prototype Inference**: Frame sampler, baseline VLM, minimal dashboard.  
- **Week 3 – Data & Head Training**: Collect/label clips, train head, first demo.  
- **Week 4 – Fusion & Alerts**: Add audio, fusion, rule engine, alerts.  
- **Week 5 – Hardening & Ablations**: Optimize latency, ablations, finalize dashboard.  
- **Week 6 – Polish & Deliver**: Record demo, finalize report, reproducibility test.  

---

## Risk & Mitigation
- **Latency too high:** lower fps, quantization (INT8/FP16), smaller VLM.  
- **Class confusion:** add motion/eye heuristics, more data.  
- **Low-light failures:** IR preprocessing, histogram equalization.  
- **Audio privacy:** local-only processing, no raw audio logs.  

---

## Ethics & Privacy
- Use consented/staged data, blur faces if publishing.  
- Default to local-only mode.  
- Provide “privacy pause” button.  

---

## Stretch Goals
- Caregiver detection + distance estimation.  
- Timeline view with daily summaries.  
- Deployment on Jetson Nano / Raspberry Pi 5.  

---

## References
- Hugging Face Transformers – LLaVA/BLIP-2 docs  
- VLMs & multimodal reasoning survey papers  
- Cry-detection literature (MFCC + shallow classifiers)  
- OpenCV + FFmpeg docs; FastAPI docs  

---

## Milestones
- **M1 (End W2):** Live pipeline runs, dashboard online.  
- **M2 (End W3):** Head improves F1 by ≥10%.  
- **M3 (End W4):** Alerts with <1 false alert/hour.  
- **Final (W6):** Demo + report + reproducible environment. 
 
---