# GLOCAL-FSL-OOD

## Overview
This repo implements few-shot medical image classification and OOD detection based on
"Global and Local Vision-Language Alignment for Few-Shot Learning and Few-Shot OOD Detection".

Key: CLIP backbone + LLM disease descriptions + Text Refinement + Local Contrastive + Global-Local Alignment.

## Docs
| File | Purpose |
|------|---------|
| README.md | Overview, usage |
| SKILL.md | Technical pipeline and design |
| TASKS.md | Progress tracker |
| CHANGELOG.md | Version history |

## Quick Start
```bash
pip install -r requirements.txt
python src/scripts/generate_llm_descriptions.py --config configs/experiment/exp_full_model.yaml
python src/scripts/train_fsl.py --config configs/experiment/exp_full_model.yaml
python src/scripts/eval_cls.py --config configs/eval/cls.yaml --checkpoint outputs/checkpoints/best.pt
python src/scripts/eval_ood.py --config configs/eval/ood.yaml --checkpoint outputs/checkpoints/best.pt
```

## Running Experiments
```bash
# Baseline CLIP
python src/scripts/train_fsl.py --config configs/experiment/exp_baseline_clip.yaml
# Full model 1/4/8-shot
python src/scripts/train_fsl.py --config configs/experiment/exp_full_model.yaml --override configs/train/fewshot_1shot.yaml
```

## Config-Driven Design
All params via YAML. Key paths:
- backbone: model.clip.backbone
- LLM: model.llm.model_name
- descriptions M: llm_descriptions.num_descriptions_per_class
- top/bottom-k: model.local_contrastive.top_k / bottom_k
- alpha: model.alignment.alpha_global / alpha_local
- shots: fewshot.k_shot
