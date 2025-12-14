# Data-Engineering-for-YOLOv8-High-Jump-Motion-Classification

## Overview ##
This project builds a data-centric computer-vision pipeline for recognizing the three phases of a high-jump attempt (Run, Jump, Land) using a YOLOv8 model. We created a custom dataset by collecting high-jump videos, extracting frames, and performing all labeling, augmentation, and dataset management through Roboflow.

## Pipeline Overview (Data-Centric) ##
Our focus is data quality first: create a consistent dataset for a temporal action task (high jump phases) using frame-based training.

Stages:

1.Data Source Selection

2.Frame extraction + sampling strategy

3.Phase labeling policy (Run / Jump / Land)

4.Dataset versioning + quality control (Roboflow)

5.Augmentation (train-only) + class balance handling

6.YOLOv8 training + evaluation


## Stage 1 — Data Source Selection

To keep the problem **well-scoped and consistent**, we decided to train the classifier **only on Olympic Games high-jump broadcasts**. This was a deliberate constraint: the goal of this project is **not** to build a fully general “any video, any camera” high-jump phase model, but a **controlled, data-centric pipeline** where dataset consistency is prioritized over maximum domain coverage.

### Why Olympic videos?
Olympic broadcasts are relatively uniform compared to random internet footage:

- **Consistent camera language:** similar angles, zoom patterns, and shot types across attempts.
- **Stable scene layout:** bar, uprights, and landing mat appear in predictable positions.
- **High visual quality:** good resolution, stable lighting, and minimal compression artifacts.
- **Reduced domain noise:** fewer extreme variations in background, camera shake, and framing.

### What this choice gives us
By narrowing the domain, we achieve:
- cleaner and more reliable phase labels
- reduced risk of the model learning camera artifacts instead of motion phases
- faster iteration on dataset versions due to fewer edge cases

### What we explicitly did *not* aim for
This classifier is **domain-specific** and does not claim generalization to:

- handheld or phone recordings
- training-session footage
- non-broadcast camera angles
- competitions with significantly different filming styles

This design choice keeps the project focused and honest: the resulting model is a **high-jump phase classifier specialized for Olympic broadcast footage**, not a general-purpose action-recognition system.

### Video Sample:

https://github.com/user-attachments/assets/58f6c53d-71b6-4159-84fd-dfae97554a79

## Stage 2 — Frame Extraction & Sampling Strategy

After constraining the data source to Olympic broadcasts, the next critical step was deciding **how frames should be extracted and sampled** from each video. The objective was **not only class balance**, but also **semantic balance within each phase**.

High-jump phases are not uniform in duration or visual importance. Some of the most informative moments (especially during the jump) occur over very few frames, while other moments (e.g., the run-up) can span hundreds of near-duplicate frames. A naïve uniform sampling strategy would therefore bias the dataset toward visually redundant content.

### Core Principle: Semantic Balance Over Time Balance

Our sampling strategy was designed around the following idea:

> The dataset should be balanced by **informational content**, not just by frame count or elapsed time.

This means ensuring that **key moments within each phase** are sufficiently represented, even if they are short in duration.

---

### Phase-Aware Sampling Strategy

#### Run Phase
- The run-up is the longest phase in time.
- Frames were sampled **sparsely** to avoid excessive redundancy.
- Sampling focused on:
  - early approach
  - mid-approach acceleration
  - final steps before takeoff

This preserves variation in posture, speed, and camera distance without flooding the dataset.

#### Jump Phase
- The jump phase is **short but critical**.
- Uniform sampling would result in very few frames for this class.
- We therefore sampled **densely** around:
  - the takeoff step
  - initial airborne frames
  - bar clearance

This ensures the classifier sees enough examples of the most discriminative postures.

#### Land Phase
- The landing phase is moderate in duration but visually diverse.
- Sampling included:
  - descent after clearance
  - first mat contact
  - immediate post-impact recovery

This captures both motion dynamics and pose variability.

---

### Why This Strategy Matters

Without phase-aware sampling:
- the model would overfit to the run-up due to sheer volume
- the jump phase would be underrepresented despite being the most important
- decision boundaries between **Jump** and **Land** would be poorly learned

By explicitly accounting for phase duration and visual importance, we created a dataset that better reflects **what matters for classification**, not just how long each phase lasts.

---

### Key Moments per Phase (Examples)

To illustrate what we consider *semantically important*, the following key moments were prioritized during sampling.

#### Run — Key Moments
- early approach with upright posture
- mid-run acceleration and curve
- final strides immediately before takeoff

<div style="display: flex; gap: 12px;">
  <img src="https://github.com/user-attachments/assets/8be56e73-3d6c-4f11-b3f1-77be727e9b7f" width="32%" />
  <img src="https://github.com/user-attachments/assets/db0b8611-a7a8-4558-ad2a-63441b21083d" width="32%" />
  <img src="https://github.com/user-attachments/assets/3a029111-3d58-46b7-9f30-9ca54d16c60e" width="32%" />
</div>
<p align="center">
  <em>Left to right: early approach posture, mid-run acceleration and curve, final strides immediately before takeoff</em>
</p>


#### Jump — Key Moments
- takeoff plant and upward drive
- early airborne posture
- bar clearance / peak height

<div style="display: flex; gap: 12px;">
  <img src="https://github.com/user-attachments/assets/cf7a77ab-d442-4575-85e6-4c3dda2b3e79" width="32%" />
  <img src="https://github.com/user-attachments/assets/26848c29-7cb4-4c56-87f9-b52198145cc3" width="32%" />
  <img src="https://github.com/user-attachments/assets/ae9b8715-13f0-4a66-beb2-156aee578fec" width="32%" />
</div>

<p align="center">
  <em>Left to right: takeoff initiation, early airborne phase, bar clearance at peak height</em>
</p>


#### Land — Key Moments
- descent toward the mat
- first contact with the landing mat
- post-impact stabilization or roll

<div style="display: flex; gap: 12px;">
  <img src="https://github.com/user-attachments/assets/b10ea3ef-4d14-454e-9a47-cc744dffb6b2" width="32%" />
  <img src="https://github.com/user-attachments/assets/ca47397b-cabd-4ff9-9d06-266dd23ac8b2" width="32%" />
  <img src="https://github.com/user-attachments/assets/d35a0e31-3b68-44d5-b426-64bbf7353980" width="32%" />
</div>

<p align="center">
  <em>Left to right: descent phase after bar clearance, initial mat contact, post-impact stabilization and recovery</em>

These examples highlight why certain frames were intentionally oversampled: they contain the most phase-specific visual cues and are critical for learning reliable class boundaries.


## Stage 3 — Phase Labeling Policy (Run / Jump / Land)

Accurate phase classification depends heavily on **clear and consistent labeling rules**, especially at phase boundaries where ambiguity is highest. To minimize subjectivity, we defined **explicit visual criteria** for transitioning between phases, based on observable athlete posture and interaction with the bar.

Our labeling policy focuses on **key biomechanical moments** that can be reliably identified across Olympic broadcast footage.

---

### Run → Jump Transition Rule

A frame is labeled as **Jump** when the athlete transitions from the run-up into takeoff.

**Rule:**
- The athlete is supported by **only one leg**, and
- the body is **nearly airborne**, indicating initiation of the takeoff phase.

In practice, this corresponds to the **plant and drive step**, where forward momentum is converted into vertical lift.

**Why this rule works:**
- it aligns with the biomechanical definition of takeoff
- it is visually distinguishable across camera angles
- it avoids labeling ambiguity during the final running strides

Frames prior to this moment remain labeled as **Run**.

---

### Jump → Land Transition Rule

A frame is labeled as **Land** only after the athlete has fully cleared the bar.

**Rule:**
- both legs are **completely above the bar**, and
- the pixels corresponding to the athlete’s shoes are **visually above the bar**.

This ensures that the entire jump and clearance sequence—including takeoff, flight, and peak height—is consistently assigned to the **Jump** class.

**Why this rule works:**
- it prevents early transition into the Land class
- it keeps bar-clearance frames within the Jump phase
- it provides a concrete, pixel-level visual criterion

Frames before this condition is met remain labeled as **Jump**.

### Visual Examples of Phase Transitions

To make the labeling policy fully transparent and reproducible, we provide visual examples of the exact frames where phase transitions occur. These examples illustrate how the defined rules are applied in practice.

---

#### Run → Jump Transition (Takeoff Initiation)

The transition from **Run** to **Jump** is defined at the moment when:
- the athlete is supported by **only one leg**, and
- the body is **nearly airborne**, indicating takeoff initiation.

<div style="display: flex; gap: 12px; justify-content: center;">
  <img src="https://github.com/user-attachments/assets/575884dd-8113-42cf-ac3d-63aeb2a36319" width="45%" />
  <img src="https://github.com/user-attachments/assets/7bcdc0d4-4aa2-4142-bb79-9860f389562e" width="45%" />
</div>

<p align="center">
  <em>Left: final running stride (Run). Right: takeoff plant with single-leg support marking the Run → Jump boundary.</em>
</p>

This transition ensures that the final approach strides are not confused with the takeoff phase, even when motion appears continuous.

---

#### Jump → Land Transition (Bar Clearance Completion)

The transition from **Jump** to **Land** is defined only after:
- both legs are **completely above the bar**, and
- the shoe pixels are visually **above the bar**.

<div style="display: flex; gap: 12px; justify-content: center;">
  <img src="https://github.com/user-attachments/assets/50baca77-f046-4ff3-97ab-c66554db7039" width="45%" />
  <img src="https://github.com/user-attachments/assets/e95e8811-c0ef-4f15-9698-6a48fee55c1f" width="45%" />
</div>

<p align="center">
  <em>Left: late jump before full clearance (Jump). Right: complete bar clearance with both feet above the bar, marking the Jump → Land boundary.</em>
</p>

This rule guarantees that all bar-clearance and peak-height frames remain within the **Jump** class, preventing premature labeling of landing behavior.

### Summary of Labeling Boundaries

| Transition      | Visual Criterion |
|-----------------|-----------------|
| Run → Jump      | Single-leg support with near-airborne posture |
| Jump → Land     | Both feet (shoe pixels) fully above the bar |

---

### Design Rationale

This labeling policy was chosen to:
- reduce subjectivity across annotators
- ensure consistency across videos and dataset versions
- align phase labels with biomechanically meaningful events

By grounding phase transitions in **clear visual cues**, we ensure that the classifier learns phase distinctions based on athlete motion rather than background or timing heuristics.

## Stage 4 — Dataset Versioning & Quality Control (Roboflow)

To ensure reproducibility, consistency, and high label quality, we managed the dataset using **explicit versioning and systematic quality control**. All dataset iterations were handled through **Roboflow**, allowing us to treat data updates with the same rigor as code changes.

---

### Dataset Versioning Strategy

Each meaningful modification to the dataset resulted in a **new dataset version**. These modifications included:
- addition of newly labeled frames
- refinement of phase boundaries based on updated labeling rules
- removal of ambiguous or mislabeled frames
- reduction of near-duplicate frames caused by dense video sampling

Versioning allowed us to:
- track the impact of data changes independently of model changes
- roll back to earlier dataset states if needed
- maintain a clear history of annotation decisions

---

### Fixed Train / Validation / Test Splits

Dataset splits were defined **before any augmentation** and remained fixed across experiments.

- training, validation, and test sets were created at the dataset level
- the same splits were reused for all model runs
- validation and test sets were kept free of augmentation

This prevents data leakage and ensures that performance comparisons reflect real model improvements rather than changes in evaluation data.

---

### Quality Control Procedures

Before exporting each dataset version, we performed targeted quality checks:

- manual inspection of **phase transition frames** (Run → Jump, Jump → Land)
- verification that labeling rules were applied consistently across videos
- removal of:
  - frames with unclear athlete posture
  - frames where the bar or athlete was occluded
  - near-identical consecutive frames offering no new information

These checks ensured that the dataset emphasized **clear, discriminative examples** rather than noisy or ambiguous ones.

---

## Stage 5 — Augmentation Strategy & Class Balance Handling

Data augmentation was treated as an **experimental variable**, not an automatic preprocessing step. To understand its effect on model performance, we created **multiple dataset versions**, some without augmentation (baseline) and others with controlled, train-only augmentation.

This allowed us to isolate the impact of augmentation on generalization and class balance.

---

### Baseline Datasets (No Augmentation)

Initial dataset versions were trained **without any augmentation**.

**Purpose of the baseline:**
- establish a clean performance reference
- verify that the labeling policy and sampling strategy were sound
- identify failure modes caused by data scarcity rather than model design

These baseline versions revealed that:
- the model could overfit to broadcast-specific cues
- short-duration phases (especially **Jump**) were more difficult to learn robustly

---

### Augmented Dataset Versions

After establishing baseline behavior, augmentation was introduced in **subsequent dataset versions**.

**Key principles:**
- augmentation was applied **only to the training set**
- validation and test sets remained untouched
- dataset splits were preserved across versions

This ensured that performance differences could be attributed to augmentation rather than changes in evaluation data.

---

### Augmentation Goals

Augmentation was designed to:
- improve robustness to lighting and minor camera variations
- reduce reliance on background and broadcast-specific artifacts
- compensate for limited visual diversity in short phases

Typical augmentations included:
- brightness and contrast variation
- mild blur and noise
- scale, crop, and translation
- small rotations within realistic bounds

All transformations were chosen to **preserve phase semantics** and avoid altering the meaning of the motion.

---

### Design Rationale

By evaluating both non-augmented and augmented dataset versions, we were able to:
- quantify the effect of augmentation on generalization
- avoid masking data issues with synthetic variation
- make informed decisions about dataset composition

This approach reinforces the data-centric philosophy of the project: **improving the dataset itself before increasing model complexity**.





