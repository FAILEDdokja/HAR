# SIH26174 – AI Human Activity Recognition for On-board BAS Experiments

**Smart India Hackathon 2026 | ISRO / Department of Space**

Edge-deployed AI system that monitors an astronaut performing a predefined scientific experiment on the future **Bharatiya Antariksh Station (BAS)**, validates the procedural sequence in real time, suggests the next step, issues offline voice alerts on errors, and generates a lightweight timestamped log — all fully standalone under microgravity conditions.

---

## 1. Problem Statement (What ISRO Actually Wants)

The system must:

- Continuously process video from fixed-payload camera(s)
- Recognize and validate a predefined experiment sequence using Human Activity Recognition + sequence logic
- Suggest the next expected step after each completed action
- Issue a voice-based alert when a step is skipped or performed out of order
- Generate a timestamped, structured, lightweight text log of conducted steps and outcomes
- Stream experiment video to a specified IP and store video locally (when connectivity exists)
- Provide a GUI for monitoring
- Operate entirely at the edge (standalone) under severe bandwidth constraints
- Handle the unique visual challenges of **microgravity** (no fixed up/down orientation)

### Sample Experiment Sequence

```
Open Main Container
  → Grasp & Place Red Box
  → Grasp & Place Green Box
  → Grasp & Place Blue Box
  → Close Main Container
```

---

## 2. Core Technical Challenges & How We Solved Them

| # | Challenge | Why It Is Hard | Our Solution |
|---|-----------|----------------|--------------|
| 1 | **Orientation / No Up-Down** | Standard pose models (MediaPipe etc.) assume upright humans and fail when the astronaut is sideways or inverted | **Kinematic Normalization** (root-centering + torso-scale + relative bone vectors) + heavy rotational data augmentation (–180° to +180°) + training on inverted data |
| 2 | **Ordinary HAR is Insufficient** | Frame-level classifiers cannot enforce order, detect skipped steps, or distinguish similar actions | **Deterministic Finite State Machine (FSM)** – AI only perceives; FSM decides validity and next step |
| 3 | **No Public Dataset** | No existing dataset contains the exact colored-box protocol + microgravity orientations | **Custom dataset** mandatory (12–15 participants, normal + inverted views, intentional errors) |
| 4 | **Hand-Object Interaction (HOI)** | Skeletal pose alone cannot tell which object is being manipulated | Lightweight object detector (YOLO) + geometric HOI features (wrist-to-object distance, relative velocity, contact probability) |
| 5 | **Real-time FPS on Edge** | Must sustain ≥25–30 FPS end-to-end | YOLO nano/s models + `imgsz=640` + TensorRT FP16 + frame skip if needed |
| 6 | **Sequence Validation & Next-Step Suggestion** | Must reject out-of-order / incomplete actions and clearly tell the astronaut what is expected | FSM tracks current state and outputs “Expected X, Observed Y” |
| 7 | **Error Explanation** | Vague alerts are useless | FSM produces exact expected vs observed action |
| 8 | **Offline Voice Alerts** | No ground-link dependency for core functions | **Vosk** (CPU-based, fully offline TTS) – keeps GPU free for vision |
| 9 | **Timestamped Logging** | Lightweight log that can be synchronized later | JSON/CSV event log with timestamps |
| 10 | **Fully Standalone / Edge** | Bandwidth is restricted; system must work with network disabled | Entire pipeline runs locally on Jetson |
| 11 | **Partial Actions** (drop box, hover, etc.) | Actions that never complete | FSM + dwell-time thresholds + “unknown / incomplete” class |
| 12 | **Occlusion by Body** | Single camera; astronaut can block the view | Confidence threshold + abstention (do not force state change) |

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FIXED PAYLOAD CAMERA(S)                              │
│                     (1080p @ 30 FPS preferred)                              │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     VIDEO INGESTION (GStreamer / OpenCV)                    │
│                              CPU                                            │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
           ┌───────────────────────┴───────────────────────┐
           ▼                                               ▼
┌──────────────────────────────┐             ┌──────────────────────────────┐
│     YOLO POSE MODEL          │             │   YOLO OBJECT DETECTOR       │
│  (YOLO26s-pose / YOLO11s)    │             │  (Fine-tuned YOLO26s/11s)    │
│  TensorRT FP16 · imgsz=640   │             │  Colored boxes + Container   │
│           GPU                │             │           GPU                │
└──────────────┬───────────────┘             └──────────────┬───────────────┘
               │                                            │
               ▼                                            ▼
┌──────────────────────────────┐             ┌──────────────────────────────┐
│  KINEMATIC NORMALIZATION     │             │     GEOMETRIC HOI FEATURES   │
│  • Root-centering (hips)     │             │  • Wrist ↔ Object distance   │
│  • Torso-scale normalization │             │  • Relative velocity         │
│  • Relative bone vectors     │             │  • Contact / grasp score     │
│  • Inter-frame velocities    │             │                              │
└──────────────┬───────────────┘             └──────────────┬───────────────┘
               │                                            │
               └──────────────────┬─────────────────────────┘
                                  ▼
               ┌──────────────────────────────────────────┐
               │     TEMPORAL ACTION CLASSIFIER           │
               │  (Lightweight MLP / TCN / MS-TCN++)      │
               │  Sliding window of normalized HOI feats  │
               └──────────────────┬───────────────────────┘
                                  ▼
               ┌──────────────────────────────────────────┐
               │     DETERMINISTIC FINITE STATE MACHINE   │
               │  • Accepts only valid transitions        │
               │  • Rejects out-of-order / skipped steps  │
               │  • Outputs: Current State, Next Expected │
               │  • Error: “Expected X, Observed Y”       │
               │                 CPU                      │
               └──────┬───────────────────┬───────────────┘
                      │                   │
                      ▼                   ▼
        ┌─────────────────────┐   ┌─────────────────────┐
        │   OFFLINE VOICE     │   │  GUI + LOGGING      │
        │   (Vosk – CPU)      │   │  Live feed + overlays│
        │   Immediate alerts  │   │  Timestamped JSON/CSV│
        └─────────────────────┘   └─────────────────────┘
```

### Key Design Principle

> **Perception is probabilistic. Decision-making is deterministic.**

The neural networks only observe (pose, objects, actions).  
The Finite State Machine is the single source of truth for experiment progress and error reporting.

---

## 4. Technology Stack Explained

### 4.1 Pose Estimation – YOLO Pose (YOLO26s-pose / YOLO11s-pose)

- **Why YOLO-pose instead of MediaPipe?**  
  MediaPipe (and most classic pose estimators) are trained with a strong upright bias. They collapse under full inversion or extreme sideways views common in microgravity.  
  YOLO-pose models can be fine-tuned with aggressive rotational augmentation and inverted data, making them orientation-robust.

- **Recommended model**: YOLO26s-pose (or YOLO11s-pose if YOLO26 ecosystem maturity is a concern).  
  “s” size gives significantly better keypoint quality than nano while remaining real-time on AGX Orin 64 GB.

- **Optimization**: TensorRT FP16, input size 640.

### 4.2 Object Detection – Fine-tuned YOLO

- Detects the **Main Container** and the three colored boxes (Red, Green, Blue).  
- These objects do not exist in COCO, so a custom fine-tuned model is mandatory.  
- Same family (YOLO26s / YOLO11s) for consistency and easy TensorRT export.

### 4.3 Orientation Handling (Critical Innovation)

Two complementary layers:

1. **Mathematical Layer (always on, < 0.2 ms)**  
   - Root-centering: subtract hip center → translation invariance  
   - Scale normalization: divide by torso length → distance invariance  
   - Relative features: bone vectors + inter-frame velocities → more robust to mild tilt

2. **Data-Driven Layer**  
   - 30–40 % of training data recorded with sideways / fully inverted subjects (or camera rotated)  
   - Aggressive rotational augmentation (–180° to +180°) during training of both pose and action models  
   - Action classifier trained only on the normalized features

### 4.4 Hand-Object Interaction (HOI) Features

Pure skeletal pose cannot distinguish “grasping red box” from “grasping green box”.  
We compute lightweight geometric features between wrist keypoints and detected object centroids:

- Euclidean distance
- Relative velocity
- Temporal contact score

These features feed the action classifier.

### 4.5 Temporal Action Classifier

- Small custom model (MLP or lightweight TCN / MS-TCN++)  
- Operates on a sliding window of normalized pose + HOI features  
- Outputs clean action segments with confidence scores  
- Generic 9-class public models are useless for this specific protocol

### 4.6 Deterministic Finite State Machine (FSM)

The heart of sequence validation.

- Maintains current experiment state
- Accepts only legal transitions
- On invalid transition → raises clear error with exact explanation
- Continuously publishes the **next expected step** to the GUI and voice system
- Confidence below threshold τ → abstain (no forced state change)

### 4.7 Offline Voice Alerts – Vosk

- Fully offline, CPU-based speech synthesis  
- Keeps the GPU free for vision models  
- Issues immediate spoken alerts on errors

### 4.8 Logging & GUI

- Timestamped JSON or CSV event log (step, outcome, confidence, error message)
- GUI shows live camera feed, pose/object overlays, current state, next expected action, and confidence
- Video can be stored locally and optionally streamed when bandwidth is available

### 4.9 Hardware Target

| Platform | Role | Notes |
|----------|------|-------|
| **NVIDIA Jetson AGX Orin 64 GB** | Primary target | 275 sparse TOPS, 15–60 W, excellent headroom for YOLO-s models at 640 + full pipeline at 30 FPS |
| Jetson Orin Nano Super | Minimum viable | Possible with nano models + heavier optimization |

**Why AGX Orin 64 GB is acceptable**  
Power (40–60 W) and cooling are manageable engineering tasks for a space-station payload, not fundamental blockers. The extra compute enables stronger models, higher resolution, and future upgrades without redesign.

---

## 5. Dataset Strategy

Custom dataset is **mandatory**.

### What to Record

- 12–15 participants performing the full sequence
- Normal + sideways + fully inverted views
- Intentional errors (wrong order, drop box, skip step, hover, incomplete grasp)
- Varied lighting, clothing, and distances
- Target: 150–250 clips or continuous videos

### Useful Public Datasets (Pre-training only)

- **Assembly101** – best procedural / HOI match
- **Ego-Exo4D** – multi-view skilled activities
- **HICO-DET** – static human-object interaction

Final models must be fine-tuned on the custom colored-box data.

---

## 6. Implementation Roadmap

| Phase | Focus | Key Output |
|-------|-------|------------|
| 0 | Setup | Repo, environment, architecture freeze |
| 1 | Dataset | Record + annotate (include inverted views) |
| 2 | Models | Train object detector + action classifier + pose fine-tune |
| 3 | Core Pipeline | Pose → Normalize → HOI → Classifier → FSM |
| 4 | System Features | Voice, GUI, logging, TensorRT optimization |
| 5 | Test & Demo | Robustness tests (inverted, occlusion, partial actions) + polished demo |

---

## 7. Recommended Team Division (5 Persons)

| Person | Role | Main Ownership |
|--------|------|----------------|
| P1 | Dataset & Annotation Lead | Recording, labeling, inverted data, augmentation |
| P2 | Perception Lead | Pose + Object Detection + HOI features + TensorRT |
| P3 | Temporal & Classifier Lead | Feature extraction + action classifier |
| P4 | Logic & Feedback Lead | FSM + Voice alerts + error explanation |
| P5 | Systems & Integration Lead | Full pipeline, GUI, logging, edge optimization, demo |

---

## 8. Success Criteria (Definition of Done)

- [ ] Correctly tracks and validates the full colored-box sequence
- [ ] Detects and clearly explains out-of-order / skipped / incomplete actions
- [ ] Continues working when subject is sideways or inverted
- [ ] Issues offline voice alerts
- [ ] Generates timestamped JSON/CSV log
- [ ] Runs fully standalone (network disabled)
- [ ] Achieves ≥25–30 FPS on target hardware (AGX Orin 64 GB)
- [ ] GUI shows live feed, current state, next expected step, and confidence

---

## 9. Critical Warnings

- Quality of **inverted training data** will decide orientation performance — do not skip this.
- The action classifier **must** be trained on experiment-specific classes; generic models will fail.
- Long occlusions remain difficult — rely on confidence threshold and abstention.
- Measure real end-to-end FPS on the actual hardware early.
- Keep the FSM completely deterministic and separate from the neural network.

---

## 10. Quick Start (High-Level)

```bash
# 1. Environment
# JetPack 6.x + TensorRT + Ultralytics

# 2. Export models
yolo export model=yolo26s-pose.pt format=engine half=True imgsz=640
yolo export model=custom_boxes.pt format=engine half=True imgsz=640

# 3. Run pipeline
python main.py --pose engine/yolo26s-pose.engine \
               --det engine/custom_boxes.engine \
               --camera 0 \
               --fsm configs/experiment_fsm.yaml
```

---

## 11. Project Documents

| Document | Description |
|----------|-------------|
| `SIH26174_Full_Project_Context_Document.pdf` | Complete knowledge base & locked decisions |
| `SIH26174_Steps_Challenges_and_Requirements.pdf` | Consolidated requirements & architecture |
| `SIH26174_Orientation_Problem_Decision_Document.docx` | Detailed orientation solution |
| `SIH26174_Action_and_Sequence_Validation_Decision_Document.docx` | FSM & sequence logic |
| `Deep Technical Engineering Dossier- SIH26174...docx` | Deep technical analysis |

---

## 12. License & Acknowledgement

Developed for **Smart India Hackathon 2026** under the problem statement issued by **ISRO / Department of Space**.

This repository and documentation are intended for the participating team’s internal use and submission.

---

**Single source of truth**: Keep this README and the Full Project Context Document synchronized.  
Whenever a design question appears, return to these documents before changing direction.
