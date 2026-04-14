# GLOCAL-FSL-OOD

**Few-Shot Medical Image Classification & Out-of-Distribution Detection**
based on *"Global and Local Vision-Language Alignment for Few-Shot Learning and Few-Shot OOD Detection"*

## Overview

This repository implements a CLIP-based vision-language framework for:
- **Few-shot medical image classification** (1-shot, 4-shot, 8-shot)
- **Few-shot out-of-distribution (OOD) detection**

The core idea: enhance CLIP representations with LLM-generated disease descriptions, text refinement using disease-relevant visual regions, supervised local contrastive learning, and global-local alignment.

## System Pipeline

```
OFFLINE:                    ONLINE/TRAIN:
Class Names                Medical Image
    |                           |
    v                           v
LLM (Qwen)               CLIP Image Encoder
Disease Descriptions         |
(M=5 per class)             /   \
"A photo of X"           Global  Local (196 patches)
    |                    Feature    |
    v                       |       v
CLIP Text Encoder    Text Refiner   LesionRegionSelector
    |                      |       /     \
    +----------------------+------/       \
                                    Top-k   Bottom-k
                                     |        |
                                     v        v
                               LocalContrastiveLearner
                                     |
       Refined Text <----------------+----------------> Global+Local Features
                                           |
                                     GlobalLocalAligner
                                     (alpha_global + alpha_local)
                                           |
                                     Classification Logits
```

## Project Documentation

| File | Purpose |
|------|---------|
| **README.md** | Overview, installation, usage |
| **SKILL.md** | Technical pipeline, module descriptions, config mapping |
| **TASKS.md** | Progress tracker (done, in progress, todo, bugs) |
| **CHANGELOG.md** | Version history and changelog |

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies:
- `torch`, `torchvision` (PyTorch 2.0+)
- `open_clip_torch` (CLIP implementation)
- `transformers` (LLM wrapper for description generation)
- `pyyaml`, `numpy`, `pandas`, `scikit-learn`, `tqdm`, `matplotlib`

## Quick Start

### 1. Generate LLM Disease Descriptions (OFFLINE)

```bash
python src/scripts/generate_llm_descriptions.py \
    --config configs/experiment/exp_full_model.yaml \
    --class_names "Fracture" "Osteoarthritis" "Bone Lesion" ...
```

This generates `data/prompts/class_descriptions.yaml` with M=5 descriptions per class.

### 2. Build Few-Shot Splits (OFFLINE)

```bash
python src/scripts/build_fewshot_split.py --config configs/experiment/exp_full_model.yaml
```

### 3. Train Few-Shot Model

```bash
# Full model, 1-shot
python src/scripts/train_fsl.py --config configs/experiment/exp_full_model.yaml

# Override with 4-shot settings
python src/scripts/train_fsl.py --config configs/experiment/exp_full_model.yaml \
    --override configs/train/fewshot_4shot.yaml

# Full model, 8-shot
python src/scripts/train_fsl.py --config configs/experiment/exp_full_model.yaml \
    --override configs/train/fewshot_8shot.yaml
```

### 4. Evaluate Classification

```bash
python src/scripts/train_fsl.py --config configs/eval/cls.yaml \
    --checkpoint outputs/checkpoints/best.pt
```

### 5. Evaluate OOD Detection

```bash
python src/scripts/eval_ood.py --config configs/eval/ood.yaml \
    --checkpoint outputs/checkpoints/best.pt
```

## Experiment Configs

Switch between experiments by changing the config file:

| Experiment | Command |
|------------|---------|
| Baseline CLIP | `--config configs/experiment/exp_baseline_clip.yaml` |
| + Text Refinement | `--config configs/experiment/exp_text_refine.yaml` |
| + Local Contrastive | `--config configs/experiment/exp_local_contrast.yaml` |
| Full Model | `--config configs/experiment/exp_full_model.yaml` |

## Config-Driven Design

All parameters are in YAML. Key parameters you can change without touching code:

| Parameter | Config File | Key |
|-----------|-------------|-----|
| CLIP backbone | `configs/model/clip_vit_b16.yaml` | `clip.backbone` |
| LLM model | `configs/model/llm_qwen.yaml` | `llm.model_name` |
| # Descriptions M | `configs/default.yaml` | `llm_descriptions.num_descriptions_per_class` |
| Top-k / Bottom-k | `configs/default.yaml` | `model.local_contrastive.top_k / bottom_k` |
| alpha_global/alpha_local | `configs/model/alignment.yaml` | `alignment.alpha_global / alpha_local` |
| Loss weights | `configs/default.yaml` | `loss.*.weight` |
| Shots per class | `configs/train/fewshot_*.yaml` | `fewshot.k_shot` |
| Learning rate | `configs/default.yaml` | `train.optimizer.lr` |
| OOD method | `configs/eval/ood.yaml` | `eval.ood.method` |

## Directory Structure

```
.
├── configs/                 # YAML configs (default, model, data, train, eval, experiment)
│   ├── default.yaml
│   ├── model/
│   │   ├── clip_vit_b16.yaml
│   │   ├── llm_qwen.yaml
│   │   ├── text_refiner.yaml
│   │   ├── local_contrastive.yaml
│   │   ├── local_region_selector.yaml
│   │   └── alignment.yaml
│   ├── data/bone_xray.yaml
│   ├── train/fewshot_1shot.yaml, fewshot_4shot.yaml, fewshot_8shot.yaml
│   ├── eval/cls.yaml, ood.yaml
│   └── experiment/exp_*.yaml
├── data/                   # Data: raw, processed, splits, prompts
├── src/
│   ├── models/
│   │   ├── encoders/       # CLIPImageEncoder, CLIPTextEncoder, LLMWrapper
│   │   ├── modules/       # DiseaseTextRefiner, LesionRegionSelector,
│   │   │                   # LocalContrastiveLearner, GlobalLocalAligner
│   │   └── framework/      # GLocalFSLOODModel (full model integration)
│   ├── datasets/           # BaseDataset, BoneXRayDataset, FewShotSampler
│   ├── losses/             # ClassificationLoss, AlignmentLoss, TotalLoss
│   ├── trainer/            # train(), validate(), test()
│   ├── evaluation/         # ClassificationMetrics, OODMetrics, Evaluator
│   ├── utils/              # config, logger, seed, checkpoint, registry
│   └── scripts/            # generate_llm_descriptions, build_fewshot_split,
│                           # train_fsl, eval_ood
├── outputs/                # logs/, checkpoints/, predictions/, figures/
├── notebooks/              # Jupyter notebooks for analysis
├── README.md               # This file
├── SKILL.md                # Technical pipeline & module descriptions
├── TASKS.md                # Progress tracker
├── CHANGELOG.md            # Version history
└── requirements.txt
```

## Key Design Decisions

1. **Config-driven**: All hyperparameters in YAML. No hard-coded values.
2. **Modular**: Each component is a standalone PyTorch module.
3. **Registry pattern**: Components registered via `src/utils/registry.py`.
4. **Frozen vs Trainable**: CLIP encoder and LLM are frozen. Only Text Refiner, Region Selector, Local Contrastive, and Aligner are trainable.
5. **Ablation-friendly**: Any component can be disabled via config (`trainable: false`).
6. **Easy backbone swap**: Change `clip.backbone` from `ViT-B/16` to `ViT-L/14`.

## Citation

If you use this code, please cite:

```bibtex
@article{glocal2024,
  title={Global and Local Vision-Language Alignment for Few-Shot Learning
         and Few-Shot OOD Detection},
  author={},
  journal={},
  year={2024}
}
```
