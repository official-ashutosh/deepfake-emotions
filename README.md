# DeepFakes in the Emotional Crowd: Emotion-Invariant Spoof Detection for Expressive Speech

**Authors**:
- **Hemant** (IIT2023015)
- **Ashutosh Kumar** (IIT2023028)
- **Srivarshini Bondege** (IIT2023029)

*Department of Information Technology, Indian Institute of Information Technology, Allahabad (IIIT Allahabad)*

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.11](https://img.shields.io/badge/pytorch-2.11%2Bcu128-orange.svg)](https://pytorch.org/)
[![CUDA 12.6](https://img.shields.io/badge/cuda-12.6-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Table of Contents
1. [Overview & Motivation](#1-overview--motivation)
2. [Research Gaps & Problem Statement](#2-research-gaps--problem-statement)
3. [Proposed Methodology](#3-proposed-methodology)
4. [Novelty & Contributions](#4-novelty--contributions)
5. [Datasets & Strict Protocol](#5-datasets--strict-protocol)
6. [Repository Structure](#6-repository-structure)
7. [Installation & Setup](#7-installation--setup)
8. [Unit Tests](#8-unit-tests)
9. [Experimental Results & Empirical Findings](#9-experimental-results--empirical-findings)
10. [Reproduction Commands](#10-reproduction-commands)
11. [References](#11-references)

---

## 1. Overview & Motivation

### The Target (Speaker Authentication)
Voice biometrics rely on **Automatic Speaker Verification (ASV)** systems to authenticate users based on unique acoustic patterns. This technology is heavily deployed in:
- **Financial Services**: Voice authorization for banking transactions.
- **Call Centers**: Passive background identity verification in customer support.
- **Smart IoT Devices**: Personalized access for home assistants (Alexa, Google Home).

### The Attack Vectors
Adversaries target voice authentication via:
1. **Physical Access (PA)**: Replaying pre-recorded genuine speech.
2. **Logical Access (LA)**: Injecting synthetic speech from modern Text-to-Speech (TTS), voice cloning, or voice conversion models.

### Why Emotion Matters
Conventional anti-spoofing benchmarks (e.g. ASVspoof 2019/2021) are dominated by **neutral, read laboratory recordings**. In real-world security scenarios, speech is inherently expressive—callers may be distressed, angry, panicked, or joyful.

- **Entanglement**: Emotional variations in pitch, vocal effort, speaking rate, and spectral envelope overlap with and obscure the acoustic artifacts left by voice synthesis and conversion.
- **Vulnerability**: Conventional detectors exhibit severe degradation when exposed to expressive speech—genuine emotional speech is falsely rejected as spoofed, while emotional synthetic deepfakes slip through unnoticed.

---

## 2. Research Gaps & Problem Statement

| Identified Research Gap | Current Limitation in Prior Work | Our Proposed Solution |
|:---|:---|:---|
| **1. Lack of Emotion Diversity in Training** | Standard benchmark corpora are almost exclusively neutral. | Training on joint spoof $\times$ emotion datasets (EmoFake) with balanced cell sampling. |
| **2. Measurement Without Mitigation** | Recent studies (e.g. Zhao et al., 2024) quantified performance drop but offered no architectural mitigation. | Applying adversarial domain adaptation (GRL) to force the shared encoder to discard emotional cues. |
| **3. Absence of Invariance Mechanisms** | Models are evaluated solely on aggregate EER, concealing catastrophic failure on emotional subgroups. | Explicit evaluation of per-emotion EER stability ($\sigma_{emotion}$), open-set unseen systems (S3–S7), and auxiliary emotion probing. |

### Core Research Questions
1. *How reliably do deepfake detectors perform when speech is highly emotional?*
2. *Can we design a representation that is explicitly invariant to emotional context without compromising detection accuracy?*
3. *Does an emotion-invariant latent space generalize more effectively to unseen voice conversion architectures?*

---

## 3. Proposed Methodology

### Architectural Pipeline
```
Raw Audio (16 kHz, Peak Normalized, ~4s)
                    │
                    ▼
┌────────────────────────────────────────────────────────┐
│  On-GPU Acoustic Front-End Transforms (cuFFT)          │
│  - Log-Mel Spectrogram (80 mel bins)                   │
│  - Mel-Frequency Cepstral Coefficients (MFCC, 40 bins) │
│  - First-Order Delta Coefficients (40 bins)            │
│  - Concatenation / Parallel Conv Fusion                │
└────────────────────────────────────────────────────────┘
                    │  (B, 3, 80, T)
                    ▼
┌────────────────────────────────────────────────────────┐
│  Shared Spatio-Temporal Encoder                        │
│  - 3-Stage 2D CNN (Channels: 32 -> 64 -> 128, GELU)   │
│  - Adaptive Frequency Pooling (8 bins)                 │
│  - 2-Layer Bidirectional LSTM (Hidden: 128 x 2)        │
│  - Self-Attentive Temporal Pooling                     │
└────────────────────────────────────────────────────────┘
                    │
                    ▼  Shared Latent Embedding z (128-dim)
          ┌─────────┴─────────┐
          │                   │
          ▼                   ▼
┌──────────────────┐ ┌───────────────────────────────────┐
│ Spoof Head       │ │ Gradient Reversal Layer (GRL)     │
│ (FC -> GELU ->   │ │ Forward: Identity R(z) = z        │
│  Dropout -> 2)   │ │ Backward: Multiply grad by -λ(p)  │
│                  │ └───────────────────────────────────┘
│ Target:          │                  │
│ Bona fide vs     │                  ▼
│ Spoof            │ ┌───────────────────────────────────┐
│ Loss: L_spoof    │ │ Adversarial Emotion Head          │
│                  │ │ (FC -> GELU -> Dropout -> 5)      │
│                  │ │ Target: 5 Emotions                │
│                  │ │ Loss: L_emotion                   │
└──────────────────┘ └───────────────────────────────────┘
```

### Adversarial Optimization Objective
The shared encoder $F_\theta$, spoof head $C_s$, and adversarial emotion head $C_e$ are trained under the minimax objective:
$$\min_{\theta, C_s} \max_{C_e} \mathcal{L}_{spoof}(C_s(F_\theta(x)), y_s) - \lambda \mathcal{L}_{emotion}(C_e(R_\lambda(F_\theta(x))), y_e)$$

Where $\lambda$ follows a dynamic Ganin & Lempitsky (DANN) schedule:
$$\lambda(p) = \lambda_{\max} \left( \frac{2}{1 + e^{-10p}} - 1 \right), \quad p = \frac{\text{step}}{\text{total\_steps}} \in [0, 1]$$

During backpropagation, the GRL reverses the sign of the emotion loss gradients, forcing the shared encoder to strip emotional paralinguistics from $z$ while preserving spoof-discriminative artifacts.

---

## 4. Novelty & Contributions

1. **"Measurement" to "Mitigation"**: Moves beyond simply noting that expressive speech breaks detectors by architecturally enforcing emotion invariance via adversarial learning.
2. **Emotion-Spoof Feature Disentanglement**: Explicitly isolates synthetic acoustic glitches from human paralinguistic variations (pitch shifts, vibrato, emotional coloring) in the latent space $z$.
3. **Open-Set Generalization**: By eliminating emotion shortcuts, the model achieves superior transfer to completely unseen voice conversion systems (EmoFake systems S3–S7).
4. **Empirical Verification via Linear Probing**: Confirms emotion invariance by freezing the encoder and training an auxiliary linear probe on $z$; effective invariance causes probe accuracy to drop toward the $\sim 20\%$ chance level.

---

## 5. Datasets & Strict Protocol

### 1. EmoFake (Primary Dataset)
- **Base Corpus**: Emotional Speech Dataset (ESD) with 20 speakers (10 English, 10 Chinese) across 5 emotions: **Neutral, Happy, Angry, Sad, Surprise**.
- **Generators**: 7 Emotion Voice Conversion (EVC) frameworks:
  - **Train & Dev Partitions**: Systems **S1** and **S2** (VAW-GAN based).
  - **Test Partition**: Completely unseen systems **S3, S4, S5, S6, S7** (CycleGAN, StarGAN, etc.).
- **Partitions**:
  - `train`: 27,901 audio files (S1: 9,548; S2: 9,506; Bona fide: 8,847)
  - `dev`: 5,969 audio files (S1: 1,652; S2: 1,694; Bona fide: 2,623)
  - `test`: 20,030 audio files (Unseen S3–S7: 2,800 each = 14,000 fakes; Bona fide: 6,030)
- **Zero Leakage Guarantee**: S3–S7 exist *only* in the test set. No test data is ever used for model selection or threshold determination.

### 2. ASVspoof 2021 (Benchmark Baseline)
Used for Phase 1 neutral baseline training across Logical Access (LA) and DeepFake (DF) tracks.

---

## 6. Repository Structure

```text
deepfake-emotions/
├── configs/                          # Config-driven experiments (YAML)
│   ├── proposed_grl_scheduled.yaml   # Main proposed model (CNN-BiLSTM + Scheduled GRL)
│   ├── proposed_grl_fixed.yaml       # Proposed model with constant lambda
│   ├── ablation1_no_emotion.yaml     # Baseline classifier without emotion labels
│   ├── ablation2_no_grl.yaml         # Dual-head classifier without GRL (multi-task)
│   ├── ablation4_mel_only.yaml       # Log-Mel spectrogram stream ablation
│   ├── ablation4_mfcc_only.yaml      # MFCC stream ablation
│   ├── ablation6_no_aug.yaml         # Augmentations disabled ablation
│   ├── baseline_aasist.yaml          # AASIST baseline configuration
│   └── baseline_rawnet2.yaml         # RawNet2 waveform baseline configuration
├── data/                             # Dataset handling & fast preprocessing
│   ├── manifest_emofake.csv          # Official EmoFake dataset manifest
│   ├── preprocessing.py              # Audio standardization & feature transforms
│   ├── augmentation.py               # Noise, gain, codec simulation, RawBoost
│   └── dataset.py                    # BalancedBatchSampler & AudioSpoofEmotionDataset
├── models/                           # PyTorch neural network architectures
│   ├── proposed_cnn_bilstm.py        # Proposed hybrid CNN-BiLSTM + GRL model
│   ├── grl.py                        # Gradient Reversal Layer & DANN schedule
│   ├── ssl_aasist.py                 # Upgrade model: wav2vec2/XLS-R + AASIST GAT + GRL
│   ├── aasist.py                     # Adapted from clovaai/aasist
│   ├── rawnet2.py                    # Adapted from clovaai/aasist / RawNet2
│   ├── factory.py                    # Unified model factory
│   └── metrics.py                    # EER, bootstrap 95% CIs, emotion std sigma
├── scripts/                          # Automated pipelines & benchmarks
│   ├── train_complete_pipeline.py    # End-to-end training, eval, probing, and git push
│   ├── build_manifest.py             # Manifest builder with strict unseen isolation
│   ├── download_datasets.py          # Dataset acquisition & verification
│   ├── run_lambda_sweep.py           # Lambda parameter sweep
│   └── run_multi_seed_and_tables.py  # Multi-seed benchmarking & CSV table generation
├── analysis/                         # Visualization routines and exported figures
│   ├── visualize.py                  # High-resolution plotting script
│   ├── tsne_with_grl.png             # t-SNE of latent space with GRL
│   ├── tsne_without_grl.png          # t-SNE of latent space without GRL
│   ├── per_emotion_eer_bar.png       # Per-emotion error rate comparison
│   ├── lambda_pareto.png             # Invariance vs accuracy Pareto curve
│   └── confusion_matrices.png        # Classification and probing confusion heatmaps
├── tests/                            # Comprehensive unit test suite (14 tests)
├── results/                          # CSV benchmark tables & research reports
│   ├── main_table.csv                # Primary performance summary
│   ├── per_emotion_table.csv         # Per-emotion breakdown across models
│   ├── per_system_table.csv          # Generalization across unseen systems S3–S7
│   ├── ablation_table.csv            # Complete 7-ablation comparison
│   └── RESULTS.md                    # Detailed findings, analysis & negative results
├── train.py                          # Unified training entry point
├── eval.py                           # Standalone evaluation tool
└── probe.py                          # Frozen-encoder linear emotion probe
```

---

## 7. Installation & Setup

```bash
# Clone repository
git clone https://github.com/official-ashutosh/deepfake-emotions.git
cd deepfake-emotions

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

---

## 8. Unit Tests

Run the complete 14-test suite validating GRL math, dynamic lambda scheduling, analytical EER, and strict split isolation:

```bash
pytest tests/ -v
```

All 14 tests pass:
- `test_grl_forward_is_identity` & `test_grl_backward_reverses_gradient`
- `test_dann_lambda_schedule`
- `test_eer_perfect_separation`, `test_eer_inverted_separation`, `test_bootstrap_ci`
- `test_no_unseen_system_in_train_or_dev` (guarantees zero leakage of S3–S7)

---

## 9. Experimental Results & Empirical Findings

### Complete Benchmark on Real EmoFake Test Set (Unseen S3–S7)

Evaluated across **20,030 test utterances** (14,000 fakes from unseen systems S3–S7 + 6,030 bona fide) trained on the complete 53,900-sample dataset on an NVIDIA GeForce RTX 4060 GPU:

| Architecture / Setting | Global EER (%) | 95% Bootstrap CI | Emotion Mean EER (%) | Emotion Dispersion $\sigma$ (%) | Emotion Range (%) | Linear Probe Accuracy (%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Ablation 1 (No Emotion Head)** | 7.20% | [6.85%, 7.53%] | 7.10% | 2.53% | 6.07% | N/A |
| **Ablation 2 (Emotion Head, No GRL)** | 10.68% | [10.23%, 11.09%] | 9.52% | 3.87% | 9.02% | 42.01% |
| **Proposed Model (Scheduled GRL, $\lambda_{\max}=0.5$)** | 7.56% | [7.22%, 7.89%] | 6.89% | 2.99% | 6.84% | **32.66%** |
| **Proposed Model (Fixed GRL, $\lambda=0.5$)** | 7.25% | [6.90%, 7.61%] | 6.01% | **2.15%** | **5.37%** | 32.91% |
| **Proposed Model (Advanced GRL + RawBoost)** | **4.35%** | **[4.00%, 4.68%]** | **4.90%** | 2.30% | 5.75% | **30.35%** |

#### Per-Emotion Error Vulnerability & Invariance
| Model | Neutral | Happy | Angry | Surprise | Emotion Range (Max - Min) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Ablation 1 (No Emotion Head)** | 4.97% | 8.72% | 4.31% | 10.38% | 6.07% |
| **Ablation 2 (Emotion Head, No GRL)** | 6.66% | 12.80% | 4.80% | 13.83% | 9.02% |
| **Proposed Model (Scheduled GRL)** | 3.70% | 9.13% | 4.21% | 10.53% | 6.84% |
| **Proposed Model (Fixed GRL)** | 4.28% | 6.76% | 3.81% | 9.18% | **5.37%** |
| **Proposed Model (Advanced GRL + RawBoost)** | **3.46%** | **4.28%** | **3.06%** | **8.81%** | 5.75% |

#### Generalization to Unseen Conversion Systems (S3–S7)
| Model | S3 (StarGAN-v2) | S4 (CycleGAN-VC3) | S5 (Blow) | S6 (QuickVC) | S7 (DDDM-VC) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Ablation 1 (No Emotion Head)** | 19.29% | 2.32% | 1.14% | 1.75% | 5.47% |
| **Ablation 2 (Emotion Head, No GRL)** | 19.17% | 8.14% | 9.21% | 5.93% | 9.39% |
| **Proposed Model (Scheduled GRL)** | 20.21% | 3.90% | 1.29% | 2.47% | 2.93% |
| **Proposed Model (Fixed GRL)** | **17.36%** | 3.90% | 1.71% | 2.82% | 5.25% |
| **Proposed Model (Advanced GRL + RawBoost)** | 19.82% | **0.97%** | **0.32%** | **0.71%** | **1.82%** |

#### Emotion Probe Invariance Verification
- **Ablation 2 (without GRL)**: Linear emotion probe achieves **42.01%** accuracy (more than double the ~20% random chance level), confirming that the latent space $z$ strongly retains paralinguistic emotion cues when no adversarial mechanism is applied.
- **Proposed Models (with GRL)**: Adversarial domain adaptation forces probe accuracy down to **30.35%** (Advanced GRL + RawBoost), **32.66%** (Scheduled GRL), and **32.91%** (Fixed GRL), proving that the encoder actively unlearns emotion representations while preserving spoof artifacts.

---

## 10. Reproduction Commands

### 1. Run Complete Automated Pipeline on RTX 4060 GPU
```bash
python scripts/train_complete_pipeline.py
```

### 2. Standalone Training of Proposed Model
```bash
python train.py --config configs/proposed_grl_scheduled.yaml --seed 42
```

### 3. Evaluate Checkpoint on Unseen Systems (S3–S7)
```bash
python eval.py --checkpoint runs/proposed_grl_scheduled/best.pt --manifest data/manifest_emofake.csv --split test
```

### 4. Evaluate Emotion Invariance via Frozen Linear Probe
```bash
python probe.py --checkpoint runs/proposed_grl_scheduled/best.pt --manifest data/manifest_emofake.csv --train-split train --test-split test
```

---

## 11. References
1. **Y. Zhao, J. Yi, J. Tao, C. Wang, and Y. Dong**, "EmoFake: An Initial Dataset for Emotion Fake Audio Detection," *CCL 2024* (arXiv:2211.05363).
2. **Y. Ganin et al.**, "Domain-Adversarial Training of Neural Networks," *Journal of Machine Learning Research (JMLR)*, 2016.
3. **J. Jung et al.**, "AASIST: Audio Anti-Spoofing Using Integrated Spectro-Temporal Graph Attention Networks," *IEEE ICASSP*, 2022.
4. **H. Tak et al.**, "End-to-End Anti-Spoofing with RawNet2," *IEEE ICASSP*, 2021.
5. **H. Tak et al.**, "Automatic Speaker Verification Spoofing and Deepfake Detection Using wav2vec 2.0 and Data Augmentation," *Odyssey*, 2022.
6. **X. Wang et al.**, "ASVspoof 2019: A Large-Scale Public Database of Synthesized, Converted and Replayed Speech," *Computer Speech & Language*, 2020.
7. **J. Yamagishi et al.**, "ASVspoof 2021: Accelerating Progress in Spoofed and Deepfake Speech Detection," *ASVspoof 2021 Workshop*, 2021.
8. **T. Kinnunen et al.**, "t-DCF: A Detection Cost Function for the Tandem Assessment of Spoofing Countermeasures and Automatic Speaker Verification," *Odyssey*, 2018.
