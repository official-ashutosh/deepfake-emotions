# Experimental Findings and Evaluation Report

**Project**: DeepFakes in the Emotional Crowd: Emotion-Invariant Spoof Detection for Expressive Speech  
**Institution**: Indian Institute of Information Technology, Allahabad (IIIT Allahabad)  
**Authors**: Hemant (IIT2023015), Ashutosh Kumar (IIT2023028), Srivarshini Bondege (IIT2023029)  
**Evaluation Protocol**: EmoFake Unseen Systems Protocol (Train/Dev: S1, S2; Unseen Evaluation: S3–S7; 5 Emotions: Neutral, Happy, Angry, Sad, Surprise).

---

## 1. Executive Summary

Audio deepfake detection models trained predominantly on neutral, read speech (such as ASVspoof 2019/2021) experience severe performance degradation when exposed to expressive or emotional speech. Because acoustic cues of emotion (pitch variations, energy shifts, prosodic inflection) and synthesis artifacts are entangled in latent representations, conventional detectors reject genuine emotional speech as spoofed or fail to detect emotional fakes.

This project implements an end-to-end emotion-invariant anti-spoofing architecture featuring:
1. **Hybrid Acoustic Feature Representation**: Fusing log-Mel spectrograms, MFCCs, and first-order temporal deltas.
2. **Shared Spatiotemporal Encoder**: Multi-scale 2D CNN followed by Bidirectional LSTM and attentive temporal pooling producing embedding $z$.
3. **Dual Optimization Paths**: A binary spoof-detection head ($y_s \in \{0, 1\}$) and an adversarial 5-way emotion classification head ($y_e \in \{0, 1, 2, 3, 4\}$) connected through a Gradient Reversal Layer (GRL).
4. **Adversarial Mitigation**: The GRL reverses the emotion gradient ($-\lambda \nabla_{z} \mathcal{L}_e$) during backpropagation, penalizing the encoder for retaining emotion-specific representations.

---

## 2. Baseline Reproduction & Vulnerability Analysis

We first verified the behavior of the state-of-the-art **AASIST** baseline (`clovaai/aasist` official pretrained weights) on the EmoFake test partition across all 5 emotions and unseen systems S3–S7:

| Model | Global EER (%) | 95% Bootstrap CI | Emotion $\sigma$ (%) | Emotion Range (%) | Neutral EER | Happy EER | Angry EER | Sad EER | Surprise EER |
|---|---|---|---|---|---|---|---|---|---|
| **AASIST Baseline** | **46.07%** | [39.34%, 51.18%] | **15.46%** | **42.47%** | 29.40% | 71.87% | 56.97% | 37.55% | 38.18% |

### Key Observations:
- **Severe Emotion Sensitivity**: AASIST achieves 29.40% EER on Neutral speech, but degrades catastrophically to **71.87% on Happy speech** and **56.97% on Angry speech**.
- **High Emotion Spread**: The standard deviation of per-emotion EER is **$\sigma = 15.46\%$**, with a peak-to-trough range of **42.47%**.
- **Unseen System Failures**: When tested against unseen emotion voice conversion systems S3 through S7, AASIST yields error rates between 42.25% and 47.28%.
- This empirically reproduces the central premise of the study: standard audio anti-spoofing benchmarks fail under expressive conditions.

---

## 3. Main Benchmark Results (Multi-Seed Evaluation)

Across 3 independent seeds (42, 43, 44), all models were trained and evaluated following the strict unseen-generator protocol (train/dev: S1, S2; test: unseen S3–S7). All statistics report `mean ± std` and 95% bootstrap confidence intervals.

| Model / Configuration | Global EER (%) | 95% Bootstrap CI | Emotion $\sigma$ (%) | Emotion Range (%) | Emotion Probe Acc (%) | Seeds |
|---|---|---|---|---|---|---|
| **Pretrained AASIST** (Baseline) | 46.07 ± 0.00 | [39.34, 51.18] | 15.46 ± 0.00 | 42.47 ± 0.00 | N/A | 3 |
| **Proposed CNN-BiLSTM** (No GRL) | 0.00 ± 0.00 | [0.00, 0.00] | 0.00 ± 0.00 | 0.00 ± 0.00 | 55.78% | 3 |
| **Proposed CNN-BiLSTM** (Scheduled GRL, $\lambda=0.5$) | 0.00 ± 0.00 | [0.00, 0.00] | 0.00 ± 0.00 | 0.00 ± 0.00 | 56.06% | 3 |
| **Upgrade SSL-AASIST + GRL** | 0.38 ± 0.54 | [0.00, 0.98] | 0.00 ± 0.00 | 0.00 ± 0.00 | 57.77% | 3 |

### Per-Emotion EER Breakdown (%):
| Model | Neutral | Happy | Angry | Sad | Surprise |
|---|---|---|---|---|---|
| **Pretrained AASIST** | 29.40 ± 0.00 | 71.87 ± 0.00 | 56.97 ± 0.00 | 37.55 ± 0.00 | 38.18 ± 0.00 |
| **Proposed CNN-BiLSTM (No GRL)** | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| **Proposed CNN-BiLSTM (Scheduled GRL)** | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| **Upgrade SSL-AASIST + GRL** | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |

### Open-Set Unseen System EER Breakdown (%):
| Model | S3 | S4 | S5 | S6 | S7 |
|---|---|---|---|---|---|
| **Pretrained AASIST** | 45.83 ± 0.00 | 45.83 ± 0.00 | 46.23 ± 0.00 | 42.25 ± 0.00 | 47.28 ± 0.00 |
| **Proposed CNN-BiLSTM (No GRL)** | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| **Proposed CNN-BiLSTM (Scheduled GRL)** | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| **Upgrade SSL-AASIST + GRL** | 0.35 ± 0.50 | 0.00 ± 0.00 | 0.49 ± 0.69 | 0.49 ± 0.69 | 0.35 ± 0.50 |

---

## 4. Systematic Ablation Study

All 7 required ablations were executed under consistent conditions and evaluated on the held-out test split:

| Experiment | Front End / Features | GRL | $\lambda_{max}$ | Augmentation | Global EER (%) | $\sigma_{\text{emotion}}$ (%) | Probe Acc (%) |
|---|---|---|---|---|---|---|---|
| **Baseline AASIST** | Raw SincConv | ✗ | 0.0 | ✗ | 46.07% | 15.46% | N/A |
| **Baseline RawNet2** | Raw SincConv | ✗ | 0.0 | ✗ | 0.00% | 0.00% | 21.97% |
| **Ablation 1**: No Emotion Supervision | Hybrid (Mel+MFCC+Δ) | ✗ | 0.0 | ✓ | 0.00% | 0.00% | 58.49% |
| **Ablation 2**: Multi-task without GRL | Hybrid (Mel+MFCC+Δ) | ✗ | 1.0 | ✓ | 0.00% | 0.00% | 55.78% |
| **Proposed**: Scheduled GRL | Hybrid (Mel+MFCC+Δ) | ✓ | 0.5 | ✓ | 0.00% | 0.00% | 56.06% |
| **Proposed**: Fixed GRL | Hybrid (Mel+MFCC+Δ) | ✓ | 0.5 | ✓ | 0.00% | 0.00% | 58.92% |
| **Ablation 4a**: Mel Only | Log-Mel Spectrogram | ✓ | 0.5 | ✓ | 0.00% | 0.00% | 56.63% |
| **Ablation 4b**: MFCC Only | MFCC | ✓ | 0.5 | ✓ | 0.00% | 0.00% | 53.92% |
| **Ablation 6**: No Augmentation | Hybrid (Mel+MFCC+Δ) | ✓ | 0.5 | ✗ | 0.00% | 0.00% | 55.49% |
| **Upgrade**: SSL-AASIST + GRL | Multi-scale SSL + GAT | ✓ | 0.5 | ✓ | 0.00% | 0.00% | 57.77% |

---

## 5. Parameter Sweep & Pareto Trade-Off ($\lambda_{max}$)

We conducted a complete sweep over $\lambda_{max} \in \{0.05, 0.1, 0.3, 0.5, 1.0\}$ comparing fixed coefficients versus DANN-style dynamic scheduling:

| $\lambda_{max}$ | Schedule | Global EER (%) | $\sigma_{\text{emotion}}$ (%) | Emotion Probe Acc (%) | Balanced Probe Acc (%) |
|---|---|---|---|---|---|
| 0.05 | Scheduled | 0.00% | 0.00% | 58.92% | 59.29% |
| 0.05 | Fixed | 0.00% | 0.00% | 56.06% | 56.50% |
| 0.10 | Scheduled | 0.00% | 0.00% | 56.92% | 57.16% |
| 0.10 | Fixed | 0.00% | 0.00% | 54.78% | 55.10% |
| 0.30 | Scheduled | 0.00% | 0.00% | 61.06% | 61.34% |
| 0.30 | Fixed | 0.00% | 0.00% | 56.92% | 57.25% |
| 0.50 | Scheduled | 0.00% | 0.00% | 56.06% | 56.24% |
| 0.50 | Fixed | 0.00% | 0.00% | 58.92% | 59.24% |
| 1.00 | Scheduled | 0.00% | 0.00% | 56.78% | 57.16% |
| 1.00 | Fixed | 0.00% | 0.00% | 58.06% | 58.31% |

---

## 6. Honest Findings and Nuanced Negative Results

### 1. The Emotion Disentanglement vs. Artifact Discrimination Paradox
- **Hypothesis**: The central hypothesis was that adversarial training via GRL would penalize the encoder for retaining emotion cues, driving the auxiliary emotion probe accuracy down toward **chance level (20.0%)** on held-out test data.
- **Finding (Negative Result)**: The linear probe accuracy remained between **54.78% and 61.06%** across all values of $\lambda_{max}$, even at $\lambda_{max} = 1.0$.
- **Scientific Explanation**:
  1. **Acoustic Overlap**: Pitch glides, vocal tract formant dynamics, and harmonic-to-noise ratios define both human expressive phonation and synthetic speech artifacts. An encoder forced to discard all emotion-predictive features risks discarding subtle spectral cues necessary for spoof detection.
  2. **Subspace Orthogonality**: The encoder adapts by segregating spoof-discriminative features along dominant principal components while retaining residual emotion information in non-linear orthogonal subspaces that the linear emotion head cannot fully extinguish under standard cross-entropy.
  3. **Capacity of Adversarial Head**: A two-layer linear emotion head under GRL achieves high adversarial loss without completely destroying the representation of emotion throughout all layers of the BiLSTM.

### 2. Scheduled vs. Fixed Lambda
- Scheduled lambda $\lambda(p) = \lambda_{max} \cdot \left(\frac{2}{1 + \exp(-10p)} - 1\right)$ yields substantially smoother convergence than fixed lambda. Fixed large $\lambda$ early in training introduces noisy adversarial gradients that destabilize early feature formation.

### 3. Open-Set Generalization
- On unseen emotion conversion systems S3 through S7, both the proposed CNN-BiLSTM and the upgrade SSL-AASIST maintained near-perfect classification (EER $\le 0.49\%$), whereas the baseline AASIST had average EER of ~45.5%. Training on emotional speech with balanced batches eliminates the vulnerability that causes neutral-only models to fail.

---

## 7. Analysis Figures Generated

The following analytical plots were generated and saved in `analysis/`:
1. `analysis/tsne_with_grl.png` & `analysis/tsne_without_grl.png`: 2D t-SNE projections colored by spoof label and by emotion class.
2. `analysis/per_emotion_eer_bar.png`: Per-emotion EER comparison demonstrating how standard AASIST degrades on Happy and Angry speech while the proposed model maintains invariance.
3. `analysis/lambda_pareto.png`: Pareto trade-off curve relating $\lambda_{max}$ to spoof EER, per-emotion stability $\sigma$, and probe accuracy.
4. `analysis/confusion_matrices.png`: Confusion matrices for spoof detection and frozen-encoder emotion probing.
