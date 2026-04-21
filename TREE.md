Fewshot-OOD-Detection/
├── README.md                          ← Project overview, usage, commands
├── SKILL.md                          ← Technical pipeline & module design
├── TASKS.md                          ← Progress tracker (done/in-progress/todo)
├── CHANGELOG.md                      ← Version history
├── PLAN.md                           ← Data + generation implementation notes
├── requirements.txt                  ← Dependencies
├── configs/
│   ├── default.yaml                  ← All default params (seed, CLIP, LLM, losses...)
│   ├── model/
│   │   ├── clip_vit_b16.yaml         ← CLIP backbone dims
│   │   ├── llm_qwen.yaml             ← LLM model & generation params
│   │   ├── text_refiner.yaml         ← Refinement layers config
│   │   ├── local_contrast.yaml       ← Local contrastive (short alias)
│   │   ├── local_contrastive.yaml    ← Local contrastive (detailed)
│   │   ├── local_region_selector.yaml ← Top-k/bottom-k region selection
│   │   └── alignment.yaml             ← alpha_global/alpha_local
│   ├── data/
│   │   ├── bone_xray.yaml             ← Dataset config (legacy)
│   │   └── splits_data.yaml           ← BTXRD + FracAtlas split config (8 classes)
│   ├── train/
│   │   ├── fewshot_0shot.yaml
│   │   ├── fewshot_1shot.yaml
│   │   ├── fewshot_4shot.yaml
│   │   └── fewshot_8shot.yaml
│   ├── eval/
│   │   ├── cls.yaml                  ← Classification eval
│   │   └── ood.yaml                  ← OOD detection eval (MSP, Energy...)
│   └── experiment/
│       ├── exp_baseline_clip.yaml     ← Ablation: CLIP only
│       ├── exp_text_refine.yaml       ← Ablation: + text refinement
│       ├── exp_local_contrast.yaml    ← Ablation: + local contrastive
│       └── exp_full_model.yaml        ← Full model (all components)
├── src/
│   ├── utils/
│   │   ├── config.py                 ← YAML loader + Config class + includes
│   │   ├── logger.py                  ← setup_logging + TensorBoardLogger
│   │   ├── seed.py                    ← set_seed (numpy/torch/cuda)
│   │   ├── checkpoint.py               ← save/load checkpoint
│   │   └── registry.py                ← Registry pattern (MODEL/DATASET/LOSS/EVALUATOR)
│   ├── models/
│   │   ├── encoders/
│   │   │   ├── clip_image_encoder.py  ← CLIPImageEncoder (global + local feats)
│   │   │   ├── clip_text_encoder.py   ← CLIPTextEncoder (text -> embeddings)
│   │   │   ├── llm_wrapper.py         ← Backward-compatible re-export
│   │   │   └── text_generation/       ← Refactored LLM generation package
│   │   │       ├── __init__.py
│   │   │       ├── base_generator.py   ← BaseTextGenerator + GenerationConfig
│   │   │       ├── hf_local_generator.py ← HF local backend
│   │   │       ├── prompt_builder.py   ← Question/attribute/description/retry prompts
│   │   │       ├── output_cleaner.py   ← Cleaning/filtering/language checks
│   │   │       ├── description_scorer.py ← Class-aware scoring + diversity control
│   │   │       ├── cache_manager.py    ← Schema v3 cache + quality metadata
│   │   │       └── llm_wrapper.py      ← Multi-stage orchestration pipeline
│   │   ├── modules/
│   │   │   ├── text_refinement.py     ← DiseaseTextRefiner (SelfAttn + CrossAttn + FFN)
│   │   │   ├── local_region_selector.py ← LesionRegionSelector (top-k/bottom-k)
│   │   │   ├── local_contrastive.py   ← LocalContrastiveLearner (supervised contrastive)
│   │   │   └── global_local_alignment.py ← GlobalLocalAligner (alpha-weighted logits)
│   │   └── framework/
│   │       └── glocal_fsl_ood_model.py ← GLocalFSLOODModel (full integration)
│   ├── datasets/
│   │   ├── base_dataset.py            ← BaseDataset (image loading, transforms)
│   │   ├── bone_xray_dataset.py       ← BoneXRayDataset (few-shot/OOD splits)
│   │   └── sampler_fewshot.py         ← FewShotSampler (episodic n-way k-shot)
│   ├── losses/
│   │   ├── classification_loss.py     ← ClassificationLoss (cross-entropy)
│   │   ├── contrastive_loss.py        ← LocalContrastiveLoss (supervised contrastive)
│   │   ├── alignment_loss.py          ← GlobalAlignmentLoss + LocalAlignmentLoss
│   │   └── total_loss.py              ← TotalLoss (weighted sum of all terms)
│   ├── trainer/
│   │   ├── train.py                   ← train() one epoch
│   │   ├── validate.py                ← validate()
│   │   └── test.py                    ← test()
│   ├── evaluation/
│   │   ├── metrics_cls.py             ← ClassificationMetrics (Acc/P/R/F1/AUROC)
│   │   ├── metrics_ood.py             ← OODMetrics (AUROC/AUPR/FPR@95)
│   │   └── evaluator.py               ← Evaluator (unified eval interface)
│   └── scripts/
│       ├── generate_llm_descriptions.py ← OFFLINE: multi-stage ID-only descriptions + quality report
│       ├── build_fewshot_split.py     ← OFFLINE: build train/val/test splits
│       ├── parse_btxrd.py             ← OFFLINE: BTXRD parser
│       ├── parse_fracatlas.py         ← OFFLINE: FracAtlas parser
│       ├── splits_dataset.py          ← OFFLINE: merge/process split dataset
│       ├── train_fsl.py               ← TRAINING: main training script
│       └── eval_ood.py                ← EVALUATION: OOD detection
├── data/
│   ├── raw/, processed/, splits/, prompts/
├── class_questions.yaml               ← Latest generated questions (root-level run)
├── class_descriptions.yaml            ← Latest structured descriptions (schema v3)
├── class_descriptions.json            ← Latest class-centric JSON output
├── quality_report.json                ← Candidate-level debug report
└── outputs/
    └── logs/, checkpoints/, predictions/, figures/