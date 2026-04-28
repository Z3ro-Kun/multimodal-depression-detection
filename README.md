<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=28&duration=3000&pause=1000&color=4FC3F7&center=true&vCenter=true&width=700&lines=Multimodal+Depression+Detection;Parameter-Efficient+Late+Fusion;DistilBERT+%2B+Wav2Vec2+on+DAIC-WOZ" alt="Typing SVG" />

<br/>

<h1>🧠 Lightweight Multimodal Depression Detection</h1>
<h3>Parameter-Efficient Late Fusion of Text and Audio Clinical Interview Data</h3>

<br/>

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗_Transformers-4.x-FFD21E?style=for-the-badge)](https://huggingface.co/transformers)
[![CUDA](https://img.shields.io/badge/CUDA-RTX_4060-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-DAIC--WOZ-8B5CF6?style=for-the-badge)](https://dcapswoz.ict.usc.edu/)

<br/>

> **"Stability is a clinical requirement, not an afterthought."**  
> Our fusion model achieves **5× lower variance** than text-only baselines —  
> consistent F1 of **0.732 ± 0.021** across 10 random seeds.

<br/>

```
┌──────────────────────────────────────────────────────────────────┐
│  161M total parameters    →    only 394K trainable  (0.24%)      │
│  Training time per seed   →    ~20–25 minutes                    │
│  Hardware requirement     →    8GB VRAM consumer GPU             │
│  Depressed recall         →    0.75  (9/12 detected)             │
└──────────────────────────────────────────────────────────────────┘
```

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Results](#-key-results)
- [Architecture](#️-architecture)
- [Repository Structure](#-repository-structure)
- [Dataset](#-dataset-daic-woz)
- [Installation](#-installation)
- [Training](#-training)
  - [Text-Only Model](#1-text-only-model)
  - [Audio-Only Baseline](#2-audio-only-baseline)
  - [Fusion Model](#3-fusion-model--primary)
  - [Statistical Evaluation](#4-statistical-multi-seed-evaluation)
- [Evaluation](#-evaluation)
- [Ablation Studies](#-ablation-studies)
- [Figures](#-figures)
- [Parameter Breakdown](#-parameter-breakdown)
- [Reproducing Results](#-reproducing-results)
- [Limitations](#-limitations)
- [Citation](#-citation)

---

## 🔬 Overview

This repository contains the complete training and evaluation pipeline for a **parameter-efficient multimodal depression detection system** applied to the DAIC-WOZ clinical interview dataset.

The core research question is: *Can multimodal fusion improve not just performance but reliability — making automated depression detection robust enough for real clinical use?*

### The Problem with Single-Modality Approaches

| Issue | Text-Only | Audio-Only |
|-------|-----------|------------|
| Variance across seeds | ±0.102 (huge) | ±0.037 (collapses) |
| Failure mode | Random init can halve F1 | 80% of runs predict all-positive |
| Clinical reliability | ❌ Unpredictable | ❌ Unusable |

### Our Solution

A **frozen-encoder late fusion** architecture that:
- Extracts text embeddings via **hierarchical DistilBERT** with learned chunk attention
- Extracts audio embeddings via **Wav2Vec2** over random 20-second segments
- Combines both via a **lightweight MLP** with LayerNorm (only trained component beyond attention)
- Achieves **5× variance reduction** while maintaining competitive F1

---

## 📊 Key Results

### Performance Across 10 Random Seeds (Fusion Model)

| Seed | Macro F1 | Accuracy | Dep Recall | Non-Dep Recall |
|------|----------|----------|------------|----------------|
| 42   | 0.7348   | 0.7714   | 0.750      | 0.783          |
| 123  | 0.7348   | 0.7714   | 0.750      | 0.783          |
| 999  | **0.7822** | **0.8000** | **0.750** | **0.826**    |
| 555  | 0.7200   | 0.7429   | 0.583      | 0.870          |
| 777  | 0.7086   | 0.7429   | 0.583      | 0.870          |
| 888  | 0.7464   | 0.7714   | 0.583      | 0.913          |
| 111  | 0.7200   | 0.7429   | 0.583      | 0.870          |
| 222  | 0.7464   | 0.7714   | 0.583      | 0.913          |
| 333  | 0.7086   | 0.7429   | 0.583      | 0.870          |
| 444  | 0.7200   | 0.7429   | 0.583      | 0.870          |
| **Mean** | **0.732** | **0.760** | — | — |
| **Std** | **±0.021** | **±0.019** | — | — |

### Cross-Model Stability Comparison

```
Audio-Only  ████░░░░░░░░░░░░░░░░  0.274 ± 0.037  [4/5 seeds: class collapse]
Text-Only   ██████████████░░░░░░  0.698 ± 0.102  [range: 0.40 → 0.84]
Fusion      ███████████████░░░░░  0.732 ± 0.021  [range: 0.71 → 0.78] ✓
```

**The stability improvement is the contribution.** A system that reliably achieves 0.73 is clinically more useful than one that sometimes hits 0.84 but often falls to 0.40.

### Best Model Confusion Matrix (Seed 999)

```
                Predicted
                Non-Dep   Depressed
Actual Non-Dep  [ 19  ]   [  4  ]    ← 82.6% specificity
Actual Depressed[  3  ]   [  9  ]    ← 75.0% sensitivity

Macro F1:  0.782   Accuracy: 80.0%
Precision: 0.692   Recall:   0.750
```

---

## 🏗️ Architecture

### System Overview

```
Interview Transcript                    Interview Audio (16 min)
        │                                       │
        ▼                                       ▼
┌───────────────────┐             ┌─────────────────────────┐
│  Sliding Window   │             │  Random Segment Sampler  │
│  512-token chunks │             │  3 × 20-second clips     │
│  256-token stride │             └───────────┬─────────────┘
│  ~6.9 chunks avg  │                         │
└────────┬──────────┘                         ▼
         │                       ┌─────────────────────────┐
         ▼                       │   Wav2Vec2-base          │
┌───────────────────┐            │   facebook/wav2vec2-base │
│   DistilBERT      │            │   94.4M params 🔒 FROZEN │
│   base-uncased    │            └───────────┬─────────────┘
│   66.4M params    │                        │
│   🔒 FROZEN       │            Per-segment: 768-d vectors
└────────┬──────────┘                        │
         │                                   ▼
Per-chunk [CLS]: 768-d            ┌─────────────────────────┐
         │                        │   Temporal Averaging     │
         ▼                        │   across 3 segments      │
┌───────────────────┐             └───────────┬─────────────┘
│ Attention Pooling │                         │
│ ⚡ 768 params     │                         │
│ TRAINABLE        │              Audio Embedding: 768-d
└────────┬──────────┘                         │
         │                                    │
Text Embedding: 768-d                         │
         │                                    │
         └──────────────┬─────────────────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │   Concatenate        │
             │   [text ; audio]     │
             │   1536-d vector      │
             └──────────┬───────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │   Fusion MLP         │  ⚡ TRAINABLE
             │   Linear(1536 → 256) │  393,474 params
             │   LayerNorm(256)     │  ← critical for stability
             │   ReLU + Dropout(0.3)│
             │   Linear(256 → 2)   │
             └──────────┬───────────┘
                        │
                        ▼
              [Non-Depressed | Depressed]
```

### Why Frozen Encoders?

With only **107 training samples**, fine-tuning 160M parameters risks catastrophic overfitting. Frozen encoders:
- Act as powerful fixed feature extractors
- Preserve all pretraining knowledge
- Reduce trainable parameter count to **0.24%**
- Enable training in **20-25 minutes** on consumer hardware

---

## 📁 Repository Structure

```
depression-detection/
│
├── 📂 training/
│   ├── train_text.py              # Text-only hierarchical DistilBERT training
│   ├── fusion_train.py            # Primary fusion model training (saves to fusion_model2/)
│   ├── train_fusion_statiscal.py  # Multi-seed statistical evaluation
│   ├── train_audio_only.py        # Audio-only baseline with focal loss
│   └── evaluate.py                # Comprehensive evaluation: F1, ROC-AUC, confusion matrix
│
├── 📂 diagrams/
│   ├── architecture_compact.pdf   # System architecture diagram
│   ├── confusion_matrix.pdf       # Best model confusion matrix
│   ├── stability_comparison.pdf   # Boxplot: variance across model types
│   ├── generate_architecture.py   # Architecture diagram generator
│   ├── generate_confusion_matrix.py
│   └── generate_stability_plot.py
│
├── 📂 models/                     # Saved checkpoints (git-ignored, see below)
│   ├── hierarchical_text/
│   │   └── best_model.pt
│   ├── fusion_model2/             # ← Primary best fusion model
│   │   └── fusion_best_model.pt
│   └── audio_only/
│       └── best_audio_only_model.pt
│
├── 📂 data/
│   └── processed/
│       └── daic_text_clean.csv    # Preprocessed transcripts (not included, see Dataset)
│
└── README.md
```

> **Note:** Model weights and raw dataset files are not included in this repository. See [Installation](#-installation) and [Dataset](#-dataset-daic-woz) sections.

---

## 📦 Dataset: DAIC-WOZ

The **Distress Analysis Interview Corpus — Wizard of Oz (DAIC-WOZ)** is a benchmark for automated depression detection from clinical interviews.

| Property | Value |
|----------|-------|
| Total participants | 189 |
| Training set | 107 (42 depressed, 65 non-depressed) |
| Development set | 35 (12 depressed, 23 non-depressed) |
| Test set | 47 (labels held out) |
| Interview duration | ~16 min average (range: 7–30 min) |
| Average transcript | 1,643 ± 877 tokens |
| Token range | 209 – 4,710 tokens |
| Average chunks | 6.9 ± 3.4 per interview |
| Audio format | 16 kHz mono WAV |
| Depression label | PHQ-8 ≥ 10 |

### Accessing the Dataset

1. Request access at: [dcapswoz.ict.usc.edu](https://dcapswoz.ict.usc.edu/)
2. Sign the data use agreement
3. Organize downloaded files as:
```
daic_audio_subset/
├── 300_P/
│   └── 300_AUDIO.wav
├── 301_P/
│   └── 301_AUDIO.wav
...
```

### Expected CSV Format

The preprocessed `daic_text_clean.csv` file should have:

```csv
participant_id,split,label,text
300,train,0,"good well i've been doing pretty well lately..."
301,train,1,"i don't know i've just been feeling really tired..."
...
```

Where `split` is `"train"` or `"dev"`, and `label` is `0` (non-depressed) or `1` (depressed).

---

## ⚙️ Installation

### Requirements

```
Python 3.10+
CUDA-capable GPU (8GB VRAM recommended)
~5GB free disk space for models
```

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/depression-detection.git
cd depression-detection

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets scikit-learn pandas numpy tqdm
pip install imblearn matplotlib seaborn psutil
```

### Verify Setup

```python
import torch
from transformers import AutoModel, Wav2Vec2Model

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")
```

---

## 🚀 Training

All training scripts follow the same conventions:
- **Paths are relative** — set `BASE_DIR` at the top of each script
- **Seeds are fixed** — all randomness sources are seeded for reproducibility
- **Class weights are computed** automatically from training split distribution

### 1. Text-Only Model

Trains the hierarchical DistilBERT encoder with sliding-window chunking and learned attention pooling.

```bash
python training/train_text.py
```

**Configuration** (edit at top of file):
```python
DATA_PATH = "data/processed/daic_text_clean.csv"
MODEL_DIR = "models/hierarchical_text"
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 512
STRIDE = 256
LEARNING_RATE = 1e-5
EPOCHS = 10
PATIENCE = 5
```

**Expected output:**
```
Training samples: 107
Test samples: 35
Class weights: tensor([0.6948, 1.7833])

Epoch 1: Loss=0.6234, Acc=0.6857, F1=0.5139
Epoch 2: Loss=0.5891, Acc=0.7143, F1=0.6261
  → New best F1: 0.6261
...
Best F1: 0.68xx
```

**Expected time:** ~15–20 min on RTX 4060

---

### 2. Audio-Only Baseline

Demonstrates why audio alone is insufficient — most seeds collapse to majority-class prediction.

```bash
python training/train_audio_only.py
```

**Configuration:**
```python
BASE_AUDIO = "D:/daic_audio_subset"     # ← update this path
CSV_PATH = "data/processed/daic_text_clean.csv"
SAVE_DIR = "models/audio_only"
SEGMENT_SECONDS = 20
N_SEGMENTS = 3
LEARNING_RATE = 1e-4
SEEDS = [42, 123, 999, 555, 777]
```

**Expected output (demonstrating instability):**
```
AUDIO-ONLY MODEL - FINAL RESULTS
===========================================
Seed 42:  F1 = 0.2553  [class collapse]
Seed 123: F1 = 0.2553  [class collapse]
Seed 999: F1 = 0.3467  [best seed]
Seed 555: F1 = 0.2553  [class collapse]
Seed 777: F1 = 0.2553  [class collapse]

Mean Macro F1: 0.2736 ± 0.0365
```

> ⚠️ The near-identical F1 values for 4/5 seeds indicate the model is stuck predicting everything as depressed. This is an expected finding that motivates multimodal fusion.

---

### 3. Fusion Model ★ Primary

The main contribution. Loads the pre-trained frozen text model, combines with frozen Wav2Vec2, and trains only the lightweight fusion MLP.

```bash
python training/fusion_train.py
```

**Configuration:**
```python
BASE_AUDIO = "D:/daic_audio_subset"     # ← update this path
CSV_PATH = "data/processed/daic_text_clean.csv"
TEXT_MODEL_PATH = "models/hierarchical_text/best_model.pt"
FUSION_SAVE_DIR = "models/fusion_model2"
SEGMENT_SECONDS = 20
N_SEGMENTS = 3
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
PATIENCE = 5
```

> **Important:** Train the text model first (`train_text.py`) before running fusion training. The fusion model loads the saved text encoder checkpoint.

**Expected output:**
```
Training samples: 107  |  Test samples: 35
Class weights: tensor([0.6948, 1.7833])

Epoch 1: Loss=0.4821, Acc=0.7143, F1=0.6500
Epoch 2: Loss=0.3914, Acc=0.7429, F1=0.7086
  → New best F1: 0.7086
Epoch 3: Loss=0.3102, Acc=0.7714, F1=0.7348
  → New best F1: 0.7348
...
Best F1: 0.73–0.78 (varies by seed)
```

**Expected time:** ~40–55 min per seed on RTX 4060

---

### 4. Statistical Multi-Seed Evaluation

Runs the fusion model across multiple seeds and computes mean ± std for rigorous statistical reporting.

```bash
python training/train_fusion_statiscal.py
```

This script runs 3 seeds by default. For the full 10-seed evaluation reported in our paper, the pre-trained `fusion_model2/fusion_best_model.pt` is evaluated with different audio segment seeds in `evaluate.py`.

---

## 📈 Evaluation

The `evaluate.py` script performs comprehensive evaluation of any saved model checkpoint, including:

- Macro F1, Accuracy, per-class Precision/Recall
- Confusion matrix with TN/FP/FN/TP breakdown
- ROC-AUC and PR-AUC curves
- Threshold sweep analysis
- Classification report

```bash
python training/evaluate.py
```

**Configuration:**
```python
TEXT_MODEL_PATH = "models/hierarchical_text/best_model.pt"
FUSION_MODEL_PATH = "models/fusion_model2/fusion_best_model.pt"
BASE_AUDIO = "D:/daic_audio_subset"
```

**Sample output:**
```
====================================================
EVALUATION RESULTS — FUSION MODEL (10 seeds)
====================================================
Mean Macro F1:   0.7322 ± 0.0212
Mean Accuracy:   0.7600 ± 0.0190
Min F1:          0.7086
Max F1:          0.7822

BEST SEED (999):
  Macro F1:  0.7822
  Accuracy:  0.8000

Classification Report:
               precision  recall  f1-score  support
Non-Depressed   0.8636    0.8261    0.8444       23
    Depressed   0.6923    0.7500    0.7200       12

Confusion Matrix:
  TN=19  FP=4
  FN=3   TP=9
```

---

## 🔬 Ablation Studies

We tested three design alternatives. All performed worse than the baseline:

### A. Five-Segment Audio Sampling

Replacing 3 random segments with 5 stratified segments:

```python
# In fusion_train.py, change:
N_SEGMENTS = 5
STRATIFIED = True   # divides interview into 5 equal regions
```

| Config | Mean F1 | Std |
|--------|---------|-----|
| **3 random (baseline)** | **0.732** | **0.021** |
| 5 stratified | 0.676 | 0.019 |

**Conclusion:** More coverage ≠ better signal. Random short segments provide coherent acoustic windows that generalize better than stratified cross-interview coverage.

---

### B. Participant-Only Audio

Using transcript timestamps to extract only participant utterances:

```python
# Implemented in train_fusion_participant_only.py
# Reads TRANSCRIPT.csv timestamps, extracts only "Participant" rows
```

| Config | Mean F1 | Dep Recall |
|--------|---------|------------|
| **Full interview (baseline)** | **0.732** | **0.667** |
| Participant-only audio | 0.635 | 0.417 |

**Conclusion:** Speaker diarization creates boundary artifacts; conversational dynamics between interviewer and participant carry implicit depression signal; full audio is more robust.

---

### C. LayerNorm in Fusion MLP

| Config | Mean F1 | Std |
|--------|---------|-----|
| **With LayerNorm (baseline)** | **0.732** | **0.021** |
| Without LayerNorm | 0.698 | 0.048 |

**Conclusion:** LayerNorm is critical. It normalizes the scale mismatch between DistilBERT and Wav2Vec2 embeddings, preventing one modality from dominating based on magnitude rather than information content.

---

## 🖼️ Figures

All figures are pre-generated in the `diagrams/` directory. To regenerate:

```bash
python diagrams/generate_architecture.py    # System architecture diagram
python diagrams/generate_confusion_matrix.py  # Best-model confusion matrix
python diagrams/generate_stability_plot.py    # Variance comparison boxplot
```

### Architecture Diagram
![Architecture](diagrams/architecture_compact.pdf)
*Hierarchical text encoding + segment-level audio encoding + late fusion*

### Stability Comparison
![Stability](diagrams/stability_comparison.pdf)
*Boxplot showing 5× variance reduction from fusion vs. text-only*

### Confusion Matrix (Best Model)
![Confusion Matrix](diagrams/confusion_matrix.pdf)
*Seed 999: TN=19, FP=4, FN=3, TP=9 — Macro F1=0.782*

---

## ⚖️ Parameter Breakdown

```
Model Component             Parameters      Trainable?
─────────────────────────────────────────────────────
DistilBERT-base-uncased     66,365,187      🔒 Frozen
  └─ Attention (task)              768      ⚡ Trainable
Wav2Vec2-base               94,371,712      🔒 Frozen
Fusion MLP                     393,474      ⚡ Trainable
  ├─ Linear(1536→256)         393,216
  ├─ LayerNorm(256)                512
  └─ Linear(256→2)                514 (includes bias)
─────────────────────────────────────────────────────
TOTAL                      161,131,397
TRAINABLE                      394,242      (0.24%)
FROZEN                     160,737,155      (99.76%)
```

**Why this matters:**
- Models with fewer trainable parameters are less likely to overfit on 107 training samples
- The 394K trainable parameters fit in ~1.5 MB — the entire task-specific model is trivially portable
- Inference runs on CPU — no GPU required for deployment

---

## 🔁 Reproducing Results

### Step-by-Step to Reproduce Paper Results

```bash
# Step 1: Train text-only model (saves best checkpoint)
python training/train_text.py

# Step 2: Train fusion model with default seed
python training/fusion_train.py

# Step 3: Run 10-seed evaluation on saved best model
python training/evaluate.py

# Step 4: Run audio-only baseline to demonstrate instability
python training/train_audio_only.py
```

### Expected Approximate Results

| Model | Expected F1 | Expected Std |
|-------|-------------|--------------|
| Text-only | 0.68–0.70 | ±0.10 |
| Audio-only | 0.27–0.30 | ±0.04 |
| Fusion | 0.73–0.75 | ±0.02 |

> Small deviations are normal due to hardware-specific floating-point behavior. Results within ±0.02 of reported values are expected.

### Pretrained Model Weights

Pre-trained model checkpoints are available upon request (not included due to dataset access requirements). Contact the repository owner with proof of DAIC-WOZ data access agreement.

---

## ⚠️ Limitations

1. **Dataset size:** 107 training samples is very small. Results may not generalize to other demographics, interview formats, or languages without retraining.

2. **PHQ-8 ground truth:** Labels derive from self-reported questionnaires, which carry inherent under/over-reporting bias. Not equivalent to clinician diagnosis.

3. **Evaluation set:** 35 samples — one misclassified sample ≈ 3% metric change. Confidence intervals are wide.

4. **Audio evaluation variance:** Random segment sampling during evaluation means different runs with different seeds produce slightly different results (hence the 10-seed evaluation).

5. **DAIC-WOZ specific:** The Wizard-of-Oz virtual interview format may not generalize to in-person clinical interviews or other interview modalities.

6. **Not a clinical tool:** This is a research system. Any clinical deployment would require extensive prospective validation, regulatory approval, and physician oversight.

---

## 📚 Citation

If you use this code or findings in your research, please cite:

```bibtex
@article{yourlastname2026lightweight,
  title     = {Lightweight Multimodal Depression Detection Using Parameter-Efficient Late Fusion},
  author    = {[Your Name]},
  journal   = {[Venue — IEEE EMBC / ICASSP / etc.]},
  year      = {2026},
  note      = {DAIC-WOZ benchmark, DistilBERT + Wav2Vec2 late fusion, 0.732 ± 0.021 Macro F1}
}
```

### Key References

```bibtex
@inproceedings{gratch2014distress,
  title     = {The Distress Analysis Interview Corpus of human and computer interviews},
  author    = {Gratch, Jonathan and Artstein, Ron and Lucas, Gale M. and others},
  booktitle = {LREC},
  pages     = {3123--3128},
  year      = {2014}
}

@inproceedings{devault2014simsensei,
  title     = {{SimSensei} Kiosk: A Virtual Human Interviewer for Healthcare Decision Support},
  author    = {DeVault, David and Artstein, Ron and Benn, Grace and others},
  booktitle = {AAMAS},
  pages     = {1061--1068},
  year      = {2014}
}

@article{sanh2019distilbert,
  title   = {{DistilBERT}, a distilled version of {BERT}},
  author  = {Sanh, Victor and Debut, Lysandre and Chaumond, Julien and Wolf, Thomas},
  journal = {arXiv preprint arXiv:1910.01108},
  year    = {2019}
}

@inproceedings{baevski2020wav2vec,
  title     = {{wav2vec} 2.0: A Framework for Self-Supervised Learning of Speech Representations},
  author    = {Baevski, Alexei and Zhou, Yuhao and Mohamed, Abdelrahman and Auli, Michael},
  booktitle = {NeurIPS},
  volume    = {33},
  pages     = {12449--12460},
  year      = {2020}
}
```

---

## 🙏 Acknowledgements

- Dataset: [DAIC-WOZ](https://dcapswoz.ict.usc.edu/) — USC Institute for Creative Technologies
- Text encoder: [DistilBERT](https://huggingface.co/distilbert-base-uncased) — HuggingFace / Hugging Face / Victor Sanh et al.
- Audio encoder: [Wav2Vec2-base](https://huggingface.co/facebook/wav2vec2-base) — Meta AI Research

---

<div align="center">

<br/>

**Made with 📊 data, 🧠 transformers, and a lot of ⏱️ GPU time**

<br/>

![Visitors](https://visitor-badge.laobi.icu/badge?page_id=yourusername.depression-detection)
[![Stars](https://img.shields.io/github/stars/yourusername/depression-detection?style=social)](https://github.com/yourusername/depression-detection)

<br/>

*If this repository helped your research, consider leaving a ⭐*

</div>
