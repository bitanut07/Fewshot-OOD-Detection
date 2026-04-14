# CHANGELOG.md — Version History

> Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
> For progress tracking, see `TASKS.md`.
> For technical design, see `SKILL.md`.

---

## [Unreleased]

### Added
- Initial project scaffolding with full directory structure
- 18 YAML configuration files:
  - `configs/default.yaml` — Default settings for all components
  - `configs/model/clip_vit_b16.yaml` — CLIP ViT-B/16 backbone config
  - `configs/model/llm_qwen.yaml` — Qwen LLM config for disease descriptions
  - `configs/model/text_refiner.yaml` — Text refinement module config
  - `configs/model/local_contrast.yaml` — Local contrastive config (short)
  - `configs/model/local_contrastive.yaml` — Local contrastive config (detailed)
  - `configs/model/local_region_selector.yaml` — Region selection config
  - `configs/model/alignment.yaml` — Global-local alignment config
  - `configs/data/bone_xray.yaml` — Bone X-Ray dataset config
  - `configs/train/fewshot_1shot.yaml` — 1-shot training config
  - `configs/train/fewshot_4shot.yaml` — 4-shot training config
  - `configs/train/fewshot_8shot.yaml` — 8-shot training config
  - `configs/eval/cls.yaml` — Classification evaluation config
  - `configs/eval/ood.yaml` — OOD detection evaluation config
  - `configs/experiment/exp_baseline_clip.yaml` — Baseline CLIP experiment
  - `configs/experiment/exp_text_refine.yaml` — Text refinement ablation
  - `configs/experiment/exp_local_contrast.yaml` — Local contrastive ablation
  - `configs/experiment/exp_full_model.yaml` — Full model experiment
- 14 Python skeleton files:
  - `src/utils/`: config.py, logger.py, seed.py, checkpoint.py, registry.py
  - `src/models/encoders/`: clip_image_encoder.py, clip_text_encoder.py, llm_wrapper.py
  - `src/models/modules/`: text_refinement.py, local_region_selector.py, local_contrastive.py, global_local_alignment.py
- Documentation: README.md, SKILL.md, TASKS.md, CHANGELOG.md

### Planned
- Full model integration in `src/models/framework/glocal_fsl_ood_model.py`
- Dataset implementations and few-shot samplers
- Training and evaluation scripts
- LLM description generation script
- Loss implementations
- OOD detection methods (MSP, Energy, Cosine, Mahalanobis)

---

## [v0.1.0] — 2025-04-14 — Initial Setup

### Added
- Project directory structure with configs/, data/, src/, outputs/, notebooks/
- YAML-based config system with inheritance via `includes` field
- Registry pattern for modular component registration
- Config-driven design (all hyperparameters in YAML, no hard-coded values)
- Documentation files for project management (README, SKILL, TASKS, CHANGELOG)

### Architecture Components Designed
- CLIP Image Encoder with global + local feature extraction
- CLIP Text Encoder for class name and description encoding
- LLM Wrapper for disease description generation
- Disease Text Refiner (self-attention + cross-attention + FFN)
- Lesion Region Selector (top-k / bottom-k selection)
- Local Contrastive Learner (supervised contrastive)
- Global-Local Aligner (alpha-weighted combination)
- Config schema for 1-shot, 4-shot, 8-shot settings
- Config schema for OOD detection (MSP, Energy, Cosine)
