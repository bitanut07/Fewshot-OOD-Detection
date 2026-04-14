# TASKS.md — Project Progress Tracker

> **This file tracks project progress.**
> For technical details, see `SKILL.md`.
> For version history, see `CHANGELOG.md`.

## Legend

- `- [ ]` = Not started / Not done
- `- [x]` = Completed

---

## Done

- [x] Create project directory structure (configs, data, src, outputs, notebooks)
- [x] Create 18 YAML config files (default, model, data, train, eval, experiment)
- [x] Create `src/utils/` modules: config loader, logger, seed, checkpoint, registry
- [x] Create `src/models/encoders/`: CLIPImageEncoder, CLIPTextEncoder, LLMWrapper
- [x] Create `src/models/modules/`: DiseaseTextRefiner, LesionRegionSelector, LocalContrastiveLearner, GlobalLocalAligner
- [x] Create `README.md`, `SKILL.md`, `TASKS.md`, `CHANGELOG.md` documentation

---

## In Progress

- [ ] Implement `src/models/framework/glocal_fsl_ood_model.py` — integrate all modules into full model
- [ ] Implement `src/datasets/`: BaseDataset, BoneXRayDataset, FewShotSampler
- [ ] Implement `src/losses/`: ClassificationLoss, LocalContrastiveLoss, AlignmentLoss, TotalLoss
- [ ] Implement `src/trainer/`: train.py, validate.py, test.py

---

## TODO

### High Priority

- [ ] Implement full training loop with episodic few-shot sampling
- [ ] Implement `src/evaluation/`: metrics_cls.py, metrics_ood.py, evaluator.py
- [ ] Implement OOD detection: MSP, Energy, Cosine, Mahalanobis methods
- [ ] Implement `src/scripts/train_fsl.py` — main training script
- [ ] Implement `src/scripts/eval_cls.py` and `src/scripts/eval_ood.py`
- [ ] Implement `src/scripts/generate_llm_descriptions.py` — LLM description generation
- [ ] Implement `src/scripts/build_fewshot_split.py` — build few-shot splits

### Medium Priority

- [ ] Add `local_region_selector.yaml` config reference to experiment configs
- [ ] Implement weighted averaging of multiple descriptions per class
- [ ] Add warmup scheduler
- [ ] Add TensorBoard / WandB logging integration
- [ ] Implement model EMA (Exponential Moving Average)
- [ ] Add mixed precision training support

### Low Priority / Future Work

- [ ] Visualization: t-SNE of embeddings, attention maps, OOD histograms
- [ ] Implement Mahalanobis distance for OOD detection
- [ ] Implement energy-based OOD scoring
- [ ] Add support for ViT-L/14 backbone
- [ ] Add support for alternative LLMs (LLaMA, Phi-3)
- [ ] Ablation study scripts
- [ ] Jupyter notebooks for analysis and visualization
- [ ] Pre-generated descriptions cache for reproducibility

---

## Bugs / Issues

| # | Description | Severity | Status |
|---|-------------|----------|--------|
| 1 | `src/models/framework/` directory is empty — full model not yet integrated | High | Open |
| 2 | `src/datasets/`, `src/losses/`, `src/trainer/`, `src/evaluation/`, `src/scripts/` are empty | High | Open |
| 3 | `configs/model/local_region_selector.yaml` not referenced in experiment configs | Medium | Open |

---

## Next Priorities

1. **Immediate**: Implement `src/models/framework/glocal_fsl_ood_model.py` to integrate all modules
2. **Immediate**: Implement `src/datasets/bone_xray_dataset.py` and `src/datasets/sampler_fewshot.py`
3. **Next**: Implement `src/losses/total_loss.py` and training loop
4. **Next**: Implement `src/scripts/train_fsl.py` and run first training experiment
5. **Then**: Implement evaluation and OOD detection
