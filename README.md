# SpineVQA: Localization-Aware Multimodal Learning for Spinal Disease Question Answering

This repository contains the implementation, experimental framework, and evaluation code for **SpineVQA**, a deep learning project that investigates multimodal learning techniques for spinal disease question answering using the **SpineBench** dataset.

The primary objective of this project is to improve spinal disease diagnosis and vertebral-level lesion localization from spine X-ray images by integrating visual and textual information through multimodal representation learning. Unlike conventional approaches that rely solely on global image representations, SpineVQA explores localization-aware learning strategies, including patch-level visual representation learning, vertebral attention mechanisms, disease-aware and localization-aware contrastive learning, hierarchical vertebral reasoning, and vision-language feature fusion.

The project provides a unified and modular experimental framework for systematically evaluating multiple deep learning architectures and learning strategies. A series of experiments are conducted to analyze the impact of global visual representations, patch-token modeling, vertebral attention, contrastive learning, hierarchical evidence modeling, and different vision-language encoders on spinal medical visual question answering. The framework is designed to support reproducible experimentation and facilitate future research on medical vision-language models for spine disease understanding.


---
## Project Overview

Medical VQA for spine images requires the model to answer two major types of questions:

1. **Spinal disease classification**

   * Predict the disease/pathology visible in the spine image.

2. **Spinal lesion localization**

   * Predict the affected vertebral level, such as `L1/L2`, `L2/L3`, `L3/L4`, `L4/L5`, or `L5/S1`.

This project investigates whether anatomical modeling of the spine improves VQA performance compared with global image feature baselines.

---

## Dataset

This project uses the **SpineBench** dataset.

The dataset is not included in this repository due to size and licensing considerations.

Expected local dataset structure:

```text
data/SpineBench/
├── all/
│   ├── train_split.json
│   ├── val_split.json
│   └── images...
└── evaluation/
    ├── test.json
    └── images...
```

Expected paths used in scripts:

```text
/home/dsia-st125985/SpineVQA/data/SpineBench/all/train_split.json
/home/dsia-st125985/SpineVQA/data/SpineBench/all/val_split.json
/home/dsia-st125985/SpineVQA/data/SpineBench/evaluation/test.json
```

---

## Disease Classes

The model predicts 12 disease classes:

```python
DISEASES = [
    "Subarticular Stenosis",
    "Foraminal stenosis",
    "Healthy",
    "Osteophytes",
    "Spinal Canal Stenosis",
    "cervical Lordosis",
    "Straight cervical vertebrae",
    "sigmoid cervical vertebrae",
    "cervical Kyphosis",
    "Disc space narrowing",
    "Spondylolisthesis",
    "Vertebral collapse"
]
```

---

## Spinal Levels

The model predicts 5 spinal levels:

```python
LEVELS = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
```

---

## Main Research Direction

The main idea is that global image features are not enough for fine-grained spinal VQA. Instead, the model should learn anatomical, vertebral-level visual representations.

The strongest architecture so far is:

```text
Frozen SigLIP2
→ patch tokens
→ vertebral attention
→ question fusion
→ disease classification + lesion localization
```

The best current model is:

```text
E4b: SigLIP2 + Vertebral Attention
Overall test accuracy: 58.32%
```

---

## Experiment Summary

| Exp. ID | Model / Method                          | Main Idea                                                             | Trainable Params | Disease Acc |    Loc Acc |  Precision |     Recall |    Overall |
| ------- | --------------------------------------- | --------------------------------------------------------------------- | ---------------: | ----------: | ---------: | ---------: | ---------: | ---------: |
| **E1a** | SigLIP2 CLS + BERT Baseline             | Uses global image CLS feature + BERT question feature                 |           ~110M+ |      47.87% |     16.80% |     38.37% |     47.29% |     33.27% |
| **E2a** | Disease Contrastive Learning            | Adds disease-level contrastive learning to baseline                   |           ~110M+ |      47.70% |     18.00% |     39.42% |     49.18% |     33.74% |
| **E3a** | Localization-Aware Contrastive Learning | Pulls samples close if they share disease and spinal level pattern    |           110.8M |      56.03% |     49.10% |     66.92% |     73.04% |     52.77% |
| **E4a** | Patch Mean Pooling                      | Uses mean-pooled SigLIP2 patch tokens instead of global CLS           |                — |      50.98% |     37.20% |     58.74% |     68.96% |     44.50% |
| **E4b** | Vertebral Attention                     | Learns 5 vertebral-level features from patch tokens                   |                — |  **63.92%** |     52.00% |     71.56% |     78.39% | **58.32%** |
| **E4c** | Question-Guided Vertebral Attention     | Adds question-guided fusion to vertebral attention                    |                — |      61.44% |     52.60% |     73.19% |     80.39% |     57.28% |
| **E5a** | E4b + Unfreeze Last 2 SigLIP2 Layers    | Partially fine-tunes SigLIP2 vision encoder                           |           127.3M |      59.40% |     54.00% |     73.40% |     76.29% |     56.86% |
| **E5b** | E4b + Unfreeze Last 4 SigLIP2 Layers    | Fine-tunes more SigLIP2 vision layers                                 |           141.5M |      57.36% |     52.30% |     73.00% |     79.24% |     54.98% |
| **E6a** | E4b + PubMedBERT                        | Replaces BERT-base with PubMedBERT                                    |           113.2M |      59.57% |     53.80% |     72.53% |     78.52% |     56.86% |
| **E7a** | HVA-Net Evidence Matrix                 | Disease-location evidence matrix `[12 × 5]`; evidence-only prediction |             6.5M |      60.20% | **54.50%** | **73.50%** | **83.34%** |     57.52% |
| **E7b** | Hybrid HVA-Net                          | Evidence matrix + auxiliary disease/location heads                    |              ~7M |      59.93% |     52.70% |     72.98% |     79.37% |     56.53% |
| **E8a** | E4b + CLIP ViT-B/16                     | Replaces frozen SigLIP2 with frozen CLIP ViT-B/16 visual encoder      |                 — |      57.27% |     52.70% |     71.62% |     76.59% |     55.12% |

---

## Ranking by Overall Test Accuracy

| Rank | Experiment | Model                                   |    Overall |
| ---: | ---------- | --------------------------------------- | ---------: |
|    1 | **E4b**    | Vertebral Attention                     | **58.32%** |
|    2 | **E7a**    | HVA-Net Evidence Matrix                 | **57.52%** |
|    3 | E4c        | Question-Guided Vertebral Attention     |     57.28% |
|    4 | E5a        | E4b + Last 2 SigLIP2 Layers Fine-tuned  |     56.86% |
|    4 | E6a        | E4b + PubMedBERT                        |     56.86% |
|    6 | E7b        | Hybrid HVA-Net                          |     56.53% |
|    7 | E8a        | E4b + CLIP ViT-B/16                     |     55.12% |
|    8 | E5b        | E4b + Last 4 SigLIP2 Layers Fine-tuned  |     54.98% |
|    9 | E3a        | Localization-Aware Contrastive Learning |     52.77% |
|   10 | E4a        | Patch Mean Pooling                      |     44.50% |
|   11 | E2a        | Disease Contrastive Learning            |     33.74% |
|   12 | E1a        | Baseline CLS Fusion                     |     33.27% |

---

## Key Findings

### 1. Vertebral attention is the strongest contribution so far

E4b improves overall accuracy from **33.27%** in the baseline to **58.32%**.

This shows that using patch-level anatomical information is much more effective than using only global image features.

---

### 2. Disease-only contrastive learning is not enough

E2a only slightly improves over the baseline:

```text
E1a: 33.27%
E2a: 33.74%
```

This suggests that disease-level contrastive learning alone is not sufficient for fine-grained spinal VQA.

---

### 3. Localization-aware contrastive learning is useful

E3a improves overall accuracy to **52.77%**, showing that disease-location-aware representation learning is more effective than disease-only contrastive learning.

---

### 4. Patch-level modeling is important

E4a uses patch mean pooling and improves over earlier baselines, but simple mean pooling is not enough.

This motivated E4b, where vertebral attention learns five level-specific spine representations.

---

### 5. Unfreezing SigLIP2 does not improve generalization

E5a and E5b fine-tuned the last 2 and last 4 SigLIP2 vision layers.

However, both performed worse than E4b:

```text
E4b: 58.32%
E5a: 56.86%
E5b: 54.98%
```

This suggests that partial SigLIP2 fine-tuning may overfit or reduce test generalization.

---

### 6. PubMedBERT does not improve overall performance

E6a replaces BERT-base with PubMedBERT. It slightly improves some localization-related metrics but reduces disease accuracy.

This suggests that the main bottleneck is visual anatomical reasoning rather than medical text understanding.

---

### 7. HVA-Net is parameter-efficient and interpretable

E7a introduces a disease-location evidence matrix of shape `[12 × 5]`.

It achieves **57.52% overall accuracy** using only **6.5M trainable parameters**.

Although it does not beat E4b, it improves localization accuracy, precision, and recall:

```text
Loc Acc:   52.00% → 54.50%
Precision: 71.56% → 73.50%
Recall:    78.39% → 83.34%
```

This shows that explicit disease-location evidence modeling is useful for localization and interpretability.

---

### 8. Hybrid HVA-Net does not improve over evidence-only HVA-Net

E7b adds auxiliary prediction heads to E7a, but it performs lower than E7a.

This suggests that naive auxiliary-head fusion may interfere with joint evidence learning.

---

### 9. SigLIP2 outperforms CLIP as the frozen visual encoder

E8a replaces E4b's frozen SigLIP2 encoder with a frozen CLIP ViT-B/16 encoder, holding the vertebral-attention architecture, text encoder, and training protocol fixed:

```text
E4b (SigLIP2): 58.32%
E8a (CLIP):    55.12%  (↓3.20)
```

The drop is broad-based rather than concentrated in one task — disease accuracy falls from 63.92% to 57.27% and precision/recall both decline modestly alongside it, while localization exact-match accuracy is essentially unchanged (52.00% vs. 52.70%). This suggests SigLIP2's pretraining (sigmoid loss, web-scale image-text pairs at higher native resolution) produces patch-level features that are a measurably better starting point for this architecture's disease reasoning specifically, while both encoders supply roughly comparable raw spatial signal for the vertebral-attention module to work with. Combined with Finding 6 (PubMedBERT does not help), this reinforces that **visual encoder choice still matters at the margin, but architectural changes (vertebral attention, evidence matrix) remain the dominant driver of performance** — encoder swaps move the needle by a few points, while the anatomical-attention mechanism itself moved it by twenty-five.

---

## Current Conclusion

The best model so far is:

```text
E4b: SigLIP2 + Vertebral Attention
Overall test accuracy: 58.32%
```

However, E7a is also important because it is:

* parameter-efficient
* interpretable
* close to E4b
* better in localization accuracy, precision, and recall

Overall, the experiments suggest that **anatomical visual modeling is more important than simply fine-tuning larger encoders or replacing the text or visual encoder**.

---

## Planned Experiments

| Planned Exp.                | Model                            | Purpose                                        |
| --------------------------- | -------------------------------- | ---------------------------------------------- |
| **E8b**                     | E4b + RadDINO                    | Test radiology-pretrained visual encoder       |
| **E8c**                     | E4b + BioMedCLIP                 | Test biomedical image-text encoder             |
| **E9a**                     | HVA-Net + GNN                    | Model spinal levels as an anatomical graph     |
| **Seed Study**              | E4b / E7a with multiple seeds    | Test stability and report mean ± std           |
| **Attention Visualization** | Vertebral attention maps         | Provide interpretability evidence              |
| **Error Analysis**          | Per-class and per-level analysis | Identify strengths and weaknesses              |

---
## W&B Report

A summarized Weights & Biases report for the SpineVQA experiments is available here:

[SpineVQA W&B Report](https://api.wandb.ai/links/st125985-asian-institute-of-technology/eo0j1qxq)

Note: Access may depend on the W&B report/project visibility settings.

## Installation

Create and activate environment:

```bash
conda create -n spinevqa python=3.10 -y
conda activate spinevqa
```

Install dependencies:

```bash
pip install torch torchvision transformers pillow numpy tqdm scikit-learn wandb open_clip_torch
```

---

## Running Experiments

Example:

```bash
cd /home/dsia-st125985/SpineVQA
python scripts/E4b_vertebral_attention.py
```

CLIP visual-encoder comparison:

```bash
cd /home/dsia-st125985/SpineVQA
python scripts/E8a_clip_e4b.py
```

For Slurm-based runs, submit the corresponding job script if available (e.g. `sbatch scripts/run_e8a.sh`).

---

## Repository Notes

The following are intentionally excluded from GitHub:

```text
data/
wandb/
models/*.pth
models/*.pt
models/*.ckpt
__pycache__/
env/
.conda/
```

Model checkpoints and datasets should be stored separately.

---

## 📊 Weights & Biases Report

All experiments were tracked using **Weights & Biases (W&B)**.

The report includes:

- Training and validation loss
- Disease classification accuracy
- Lesion localization accuracy
- Precision, Recall, and F1-score
- Learning rate schedules
- Experiment comparison across all models

🔗 **W&B Report**

https://api.wandb.ai/links/st125985-asian-institute-of-technology/eo0j1qxq



## Author

**Nabin Lama**
Master's in Data Science and AI
Asian Institute of Technology
