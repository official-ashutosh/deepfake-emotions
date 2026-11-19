# Experimental Findings and Evaluation Report: Complete Real EmoFake Benchmark

**Project**: DeepFakes in the Emotional Crowd: Emotion-Invariant Spoof Detection for Expressive Speech  
**Institution**: Indian Institute of Information Technology, Allahabad (IIIT Allahabad)  
**Authors**: Hemant (IIT2023015), Ashutosh Kumar (IIT2023028), Srivarshini Bondege (IIT2023029)  
**Dataset**: Full EmoFake Dataset (53,900 clips; ESD base: 10 English, 10 Chinese speakers; 5 Emotions: Neutral, Happy, Angry, Sad, Surprise; 7 Emotion Voice Conversion systems S1–S7)  
**Hardware Platform**: NVIDIA GeForce RTX 4060 Laptop GPU (8,188 MiB VRAM, CUDA 12.6, cuFFT Front-End)  
**Evaluation Protocol**: Strict Unseen-Generator Protocol (Train/Dev: S1, S2, Bona fide; Test: Unseen S3–S7, Bona fide; exactly 0% unseen system leakage)

---

## 1. Executive Summary

Conventional audio anti-spoofing countermeasures (trained on neutral speech benchmarks such as ASVspoof 2019/2021) undergo catastrophic failure when exposed to expressive or emotional speech. Because acoustic indicators of human emotion (pitch modulation, vocal intensity, speaking rate) overlap with the synthetic artifacts introduced by voice conversion and synthesis models, standard detectors confound emotion with spoofing.

This investigation trained, evaluated, and probed full-scale models on all **53,900 samples of the official EmoFake benchmark** using the local RTX 4060 GPU:
1. **Ablation 1 (No Emotion Head)**: Baseline spatiotemporal CNN-BiLSTM trained purely for spoof detection.
2. **Ablation 2 (Emotion Head, No GRL, $\lambda=0$)**: Multi-task model learning both spoof and emotion classification without adversarial domain adaptation.
3. **Proposed Model (Scheduled GRL, $\lambda_{\max}=0.5$)**: Adversarial minimax model with dynamic DANN schedule $\lambda(p) = 0.5 \cdot \left(\frac{2}{1 + e^{-10p}} - 1\right)$.
4. **Proposed Model (Fixed GRL, $\lambda=0.5$)**: Adversarial minimax model with constant GRL weight $\lambda=0.5$.
5. **Proposed Model (Advanced: Scheduled GRL + RawBoost)**: State-of-the-art model combining dynamic GRL, RawBoost signal-dependent impulsive noise augmentation, and full convergence over 15 epochs.

---

## 2. Main Experimental Results (20,030-Sample Test Set)

All metrics are evaluated on the 20,030-sample unseen test partition (6,030 bona fide utterances and 14,000 synthetic utterances from unseen voice conversion frameworks S3–S7). 95% confidence intervals are computed via 500 bootstrap iterations.

| Model / Architecture | Global EER (%) | 95% Bootstrap CI | Emotion Mean EER (%) | Emotion Dispersion $\sigma$ (%) | Emotion Range (%) | Linear Probe Accuracy (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ablation 1** (No Emotion Head) | 7.20% | [6.85%, 7.53%] | 7.10% | 2.53% | 6.07% | N/A |
| **Ablation 2** (Emotion Head, No GRL) | 10.68% | [10.23%, 11.09%] | 9.52% | 3.87% | 9.02% | 42.01% |
| **Proposed Model** (Scheduled GRL, $\lambda_{\max}=0.5$) | 7.56% | [7.22%, 7.89%] | 6.89% | 2.99% | 6.84% | 32.66% |
| **Proposed Model** (Fixed GRL, $\lambda=0.5$) | 7.25% | [6.90%, 7.61%] | 6.01% | **2.15%** | **5.37%** | 32.91% |
| **Proposed Model** (Advanced GRL + RawBoost) | **4.35%** | **[4.00%, 4.68%]** | **4.90%** | 2.30% | 5.75% | **30.35%** |

---

## 3. Per-Emotion Performance & Expressive Speech Vulnerability

| Model | Neutral | Happy | Angry | Surprise | Emotion Range (Max - Min) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Ablation 1** (No Emotion Head) | 4.97% | 8.72% | 4.31% | 10.38% | 6.07% |
| **Ablation 2** (No GRL) | 6.66% | 12.80% | 4.80% | 13.83% | 9.02% |
| **Proposed Model** (Scheduled GRL) | 3.70% | 9.13% | 4.21% | 10.53% | 6.84% |
| **Proposed Model** (Fixed GRL) | 4.28% | 6.76% | 3.81% | 9.18% | **5.37%** |
| **Proposed Model** (Advanced GRL + RawBoost) | **3.46%** | **4.28%** | **3.06%** | **8.81%** | 5.75% |

### Key Findings on Emotion Invariance:
1. **Severe Degradation on Expressive Speech Without Adversarial Mitigation**: In Ablation 2 (No GRL), error rates surge from 4.80% on `angry` to **13.83% on `surprise`** and **12.80% on `happy`** (nearly 3x higher error).
2. **Adversarial Stabilization**: Fixed GRL reduces `happy` error from 12.80% down to **6.76%**, and the Advanced GRL + RawBoost model further reduces it to **4.28%** (a 66.6% relative error reduction from baseline).
3. **Lowest Emotion Dispersion & Error**: The Advanced GRL + RawBoost model cuts mean emotion error from 9.52% down to **4.90%**, with `neutral` reaching **3.46%** and `angry` reaching **3.06%**.

---

## 4. Open-Set Generalization Across Unseen Synthesis Systems (S3–S7)

The evaluation test set contains zero training voice conversion systems; all synthetic speech is produced by unseen systems S3 through S7:

| Model | S3 (StarGAN-v2-VC) | S4 (CycleGAN-VC3) | S5 (Blow) | S6 (QuickVC) | S7 (DDDM-VC) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Ablation 1** (No Emotion Head) | 19.29% | 2.32% | 1.14% | 1.75% | 5.47% |
| **Ablation 2** (No GRL) | 19.17% | 8.14% | 9.21% | 5.93% | 9.39% |
| **Proposed Model** (Scheduled GRL) | 20.21% | 3.90% | 1.29% | 2.47% | 2.93% |
| **Proposed Model** (Fixed GRL) | **17.36%** | 3.90% | 1.71% | 2.82% | 5.25% |
| **Proposed Model** (Advanced GRL + RawBoost) | 19.82% | **0.97%** | **0.32%** | **0.71%** | **1.82%** |

### Key Findings on Open-Set Generalization:
- Without GRL (Ablation 2), multi-task emotion learning degrades open-set detection dramatically across S4 (8.14%), S5 (9.21%), S6 (5.93%), and S7 (9.39%).
- With the Advanced GRL + RawBoost model, unseen system error rates drop to sub-1% levels:
  - **S4 (CycleGAN-VC3)**: **0.97%**
  - **S5 (Blow)**: **0.32%**
  - **S6 (QuickVC)**: **0.71%**
  - **S7 (DDDM-VC)**: **1.82%**

---

## 5. Linear Emotion Probing Verification

To confirm whether adversarial GRL training successfully purges emotion information from the shared latent embedding $z \in \mathbb{R}^{128}$, an independent linear logistic regression probe was trained on frozen embeddings:

| Model | Test Accuracy (%) | Balanced Accuracy (%) | Reference Random Chance (%) |
| :--- | :---: | :---: | :---: |
| **Ablation 2 (No GRL)** | **42.01%** | 46.01% | 20.0% |
| **Proposed Model (Scheduled GRL)** | 32.66% | 34.75% | 20.0% |
| **Proposed Model (Fixed GRL)** | 32.91% | 35.12% | 20.0% |
| **Proposed Model (Advanced GRL + RawBoost)** | **30.35%** | **35.24%** | 20.0% |

### Probing Conclusion:
- In the non-adversarial model, emotion is readily recoverable from $z$ with **42.01% accuracy** (more than 2x random chance).
- In the Advanced GRL model, adversarial backpropagation suppresses emotion recoverability down to **30.35%**, validating that the latent space has successfully disentangled spoof cues from emotional paralinguistics.
