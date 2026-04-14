# SKILL.md — Technical System Design Document

> **This file is a system skill / technical design document.**
> It describes the pipeline, modules, input/output, and config mapping.
> For progress tracking, see `TASKS.md`.
> For version history, see `CHANGELOG.md`.

## What This Repository Does

This repository implements the **GLOCAL-FSL-OOD** framework for:
1. **Few-shot medical image classification** — classifying images with very few (1, 4, or 8) labeled examples per class
2. **Few-shot out-of-distribution (OOD) detection** — identifying whether a query image belongs to a known class (seen during training) or an unknown class (never seen)

The core idea is to leverage **CLIP's vision-language representations** enhanced with:
- LLM-generated disease descriptions for richer text embeddings
- Text refinement to align text with disease-relevant visual regions
- Local contrastive learning on selected image regions
- Global-local alignment for classification

---

## System Pipeline (End-to-End)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           OFFLINE PHASE                                  │
│                                                                         │
│  Class Names ──────────► LLM (Qwen/LLaMA) ──► Disease Descriptions     │
│  (e.g., "Fracture")     (generate M=5 descs)     + Template Prompt      │
│                                          "A photo of Fracture"           │
│                                                                         │
│  CLIP Text Encoder ────────────────────────────────────► Text Embeddings │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           ONLINE / TRAIN PHASE                          │
│                                                                         │
│  Medical Image ──► CLIP Image Encoder ──► Global Feature (CLS token)    │
│                   (frozen backbone)        └──► Local Features (196)     │
│                                              per-patch embeddings         │
│                                                                         │
│  Text Embeddings ──► Text Refiner ◄─── Global Feature                   │
│  (M+1 per class)     (Self-Attn + Cross-Attn + FFN)                     │
│                      DiseaseTextRefiner                                  │
│                                                                         │
│  Local Features ──► LesionRegionSelector ──► Top-k Relevant Patches     │
│                    (similarity w/ class proto)  └──► Bottom-k Irrelevant│
│                                                                         │
│  Top-k Relevant ◄──► LocalContrastiveLearner ◄──► Bottom-k Irrelevant  │
│  (pull close)        Supervised Local Contrastive     (push away)        │
│                                                                         │
│  Global Feature ─┐                                                    │
│  Local Features ──┼──► GlobalLocalAligner ──► Classification Logits    │
│  Refined Text ───┘   (alpha_global + alpha_local)                     │
│                                                                         │
│  Classification Loss + Global Alignment Loss + Local Alignment Loss    │
│  + Local Contrastive Loss + Text Refinement Loss                        │
│                              │                                          │
│                              ▼                                          │
│                    Update trainable params only                          │
│                    (Text Refiner + Local modules + Aligner)             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           EVALUATION PHASE                               │
│                                                                         │
│  Query Image ──► Forward Pass ──► Logits ──► Predicted Class            │
│                   Global+Local                  Score = softmax(logits) │
│                   Alignment                                              │
│                                                                         │
│  OOD Detection: Score < threshold? ──► Known (ID) vs Unknown (OOD)      │
│  Methods: MSP, Energy, Cosine, Mahalanobis                              │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Module Descriptions

### 1. LLM Disease Description Generator
**File**: `src/models/encoders/llm_wrapper.py` (`LLMWrapper`)

**Purpose**: Generate M diverse, clinically accurate visual descriptions for each disease class.

**Input**: Class name (e.g., "Fracture")
**Output**: List of M description strings (e.g., ["Bone discontinuity visible on X-ray", "Sharp edge in cortical bone", ...])

**Config**: `configs/model/llm_qwen.yaml`
| Key | Description |
|-----|-------------|
| `llm.model_name` | HuggingFace model name |
| `llm.temperature` | Sampling temperature |
| `llm.max_new_tokens` | Max tokens per generation |
| `llm_descriptions.num_descriptions_per_class` | M = number of descriptions |
| `llm_descriptions.use_template_prompt` | Include "A photo of {class_name}" |

**Status**: Skeleton only. Needs actual LLM API integration.

---

### 2. CLIP Image Encoder
**File**: `src/models/encoders/clip_image_encoder.py` (`CLIPImageEncoder`)

**Purpose**: Extract global (pooled CLS token) and local (per-patch) visual features from medical images.

**Input**: Image tensor [B, 3, 224, 224]
**Output**:
- `global_feat`: [B, 512] (pooled feature)
- `local_feat`: [B, 196, 768] (per-patch features, ViT-B/16)

**Config**: `configs/model/clip_vit_b16.yaml`
| Key | Description |
|-----|-------------|
| `clip.backbone` | Model name (ViT-B/16, ViT-L/14) |
| `clip.freeze` | Freeze backbone weights |
| `clip.return_local_features` | Return per-patch embeddings |
| `clip.return_global_feature` | Return pooled feature |

**Status**: Implemented.

---

### 3. CLIP Text Encoder
**File**: `src/models/encoders/clip_text_encoder.py` (`CLIPTextEncoder`)

**Purpose**: Encode class names and disease descriptions into text embeddings using CLIP.

**Input**: List of text prompts
**Output**: Text embeddings [num_texts, 512]

**Config**: Inherits from `clip_vit_b16.yaml`

**Status**: Implemented.

---

### 4. Disease Text Refiner
**File**: `src/models/modules/text_refinement.py` (`DiseaseTextRefiner`)

**Purpose**: Refine text embeddings using disease-relevant visual embeddings. Uses self-attention, cross-attention (text attends to visual), and FFN.

**Input**:
- `text_embeddings`: [B, T, 512]
- `visual_embeddings`: [B, V, 512] (disease-relevant visual features)

**Output**: Refined text embeddings [B, T, 512]

**Config**: `configs/model/text_refiner.yaml`
| Key | Description |
|-----|-------------|
| `text_refiner.num_layers` | Number of refinement layers |
| `text_refiner.num_heads` | Attention heads per layer |
| `text_refiner.alpha` | Interpolation: alpha * text + (1-alpha) * refined |
| `text_refiner.trainable` | Enable/disable training |

**Architecture**:
```
Text Embeddings ──► Self-Attention ──► Cross-Attention (attend to visual) ──► FFN ──► Refined
```

**Status**: Implemented (skeleton + core logic).

---

### 5. Lesion Region Selector
**File**: `src/models/modules/local_region_selector.py` (`LesionRegionSelector`)

**Purpose**: Select disease-relevant (top-k) and disease-irrelevant (bottom-k) regions from patch embeddings based on similarity with class prototypes.

**Input**:
- `local_features`: [B, num_patches, dim]
- `prototypes`: [B, num_classes, dim] or [B, 1, dim]

**Output**:
- `relevant_features`: [B, top_k, dim]
- `irrelevant_features`: [B, bottom_k, dim]
- `relevant_indices`: [B, top_k]
- `irrelevant_indices`: [B, bottom_k]

**Config**: `configs/model/local_region_selector.yaml`
| Key | Description |
|-----|-------------|
| `region_selector.top_k` | Number of disease-relevant regions |
| `region_selector.bottom_k` | Number of disease-irrelevant regions |
| `region_selector.similarity_metric` | cosine, dot_product |
| `region_selector.normalize_before_sim` | L2-normalize before similarity |

**Status**: Implemented (skeleton + core logic).

---

### 6. Local Contrastive Learner
**File**: `src/models/modules/local_contrastive.py` (`LocalContrastiveLearner`)

**Purpose**: Perform supervised local contrastive learning:
- **Positive pairs**: Disease-relevant patches from the same class (pull closer)
- **Negative pairs**: Disease-irrelevant patches / cross-class disease-relevant patches (push apart)

**Input**:
- `relevant_features`: [B, top_k, dim]
- `irrelevant_features`: [B, bottom_k, dim]
- `class_prototypes`: [num_classes, dim]

**Output**: Contrastive loss scalar + metrics dict

**Config**: `configs/model/local_contrastive.yaml`
| Key | Description |
|-----|-------------|
| `local_contrastive.temperature` | Temperature for softmax |
| `local_contrastive.top_k` | Number of relevant patches |
| `local_contrastive.bottom_k` | Number of irrelevant patches |
| `local_contrastive.trainable` | Enable/disable training |

**Status**: Implemented (skeleton + core logic).

---

### 7. Global-Local Aligner
**File**: `src/models/modules/global_local_alignment.py` (`GlobalLocalAligner`)

**Purpose**: Compute final classification logits by combining global alignment (image global vs. text) and local alignment (image patches vs. text).

**Input**:
- `image_global`: [B, 512]
- `local_features`: [B, 196, 768] or None
- `text_embeddings`: [num_classes, 512]

**Output**: Classification logits [B, num_classes]

**Config**: `configs/model/alignment.yaml`
| Key | Description |
|-----|-------------|
| `alignment.alpha_global` | Weight for global alignment |
| `alignment.alpha_local` | Weight for local alignment |
| `alignment.logit_temperature` | Temperature for softmax |
| `alignment.learnable_weights` | Make alpha learnable |

**Formula**:
```
logits = alpha_global * global_logits + alpha_local * local_logits
```

**Status**: Implemented (skeleton + core logic).

---

### 8. Full Model Integration
**File**: `src/models/framework/glocal_fsl_ood_model.py` (`GLocalFSLOODModel`)

**Purpose**: Integrate all components into a single trainable model.

**Components integrated**:
1. CLIP Image Encoder (frozen)
2. CLIP Text Encoder (frozen)
3. Disease Text Refiner (trainable)
4. Lesion Region Selector (trainable)
5. Local Contrastive Learner (trainable)
6. Global-Local Aligner (trainable)

**Status**: NOT YET IMPLEMENTED. Needs integration of all modules.

---

### 9. Losses
**Files**: `src/losses/`

| File | Class | Purpose |
|------|-------|---------|
| `classification_loss.py` | `ClassificationLoss` | Cross-entropy on query set |
| `contrastive_loss.py` | `LocalContrastiveLoss` | Supervised local contrastive loss |
| `alignment_loss.py` | `GlobalAlignmentLoss`, `LocalAlignmentLoss` | Alignment between image and text |
| `total_loss.py` | `TotalLoss` | Weighted sum of all losses |

**Config**: `configs/default.yaml` → `loss` section
| Key | Description |
|-----|-------------|
| `loss.classification.weight` | Weight for classification loss |
| `loss.global_alignment.weight` | Weight for global alignment loss |
| `loss.local_alignment.weight` | Weight for local alignment loss |
| `loss.local_contrastive.weight` | Weight for local contrastive loss |
| `loss.text_refinement.weight` | Weight for text refinement loss |

---

### 10. Few-Shot Dataset & Sampler
**Files**: `src/datasets/`

| File | Class | Purpose |
|------|-------|---------|
| `base_dataset.py` | `BaseDataset` | Base dataset with image loading |
| `bone_xray_dataset.py` | `BoneXRayDataset` | Bone X-Ray dataset implementation |
| `sampler_fewshot.py` | `FewShotSampler` | Episodic sampler for few-shot episodes |

**Config**: `configs/data/bone_xray.yaml` + `configs/default.yaml` → `fewshot` section

---

### 11. Training Loop
**Files**: `src/trainer/`

| File | Function | Purpose |
|------|----------|---------|
| `train.py` | `train_one_epoch`, `train` | Main training loop |
| `validate.py` | `validate` | Validation loop |
| `test.py` | `test` | Test loop |

**Config**: `configs/train/fewshot_*.yaml` + `configs/default.yaml` → `train` section

---

### 12. Evaluation
**Files**: `src/evaluation/`

| File | Class | Purpose |
|------|-------|---------|
| `metrics_cls.py` | Classification metrics | Accuracy, Precision, Recall, F1, AUROC |
| `metrics_ood.py` | OOD metrics | AUROC, AUPR, FPR@95 |
| `evaluator.py` | `Evaluator` | Unified evaluation interface |

**Config**: `configs/eval/cls.yaml` + `configs/eval/ood.yaml`

---

## Config File to Module Mapping

| Config File | Controls |
|-------------|----------|
| `configs/default.yaml` | All default values (seed, paths, CLIP, LLM, text refiner, local contrastive, alignment, loss weights, fewshot, train, eval) |
| `configs/model/clip_vit_b16.yaml` | CLIP backbone dimensions |
| `configs/model/llm_qwen.yaml` | LLM model name, generation params |
| `configs/model/text_refiner.yaml` | Text refinement architecture |
| `configs/model/local_region_selector.yaml` | Region selection strategy |
| `configs/model/local_contrastive.yaml` | Local contrastive learning params |
| `configs/model/alignment.yaml` | Global-local alignment weights |
| `configs/data/bone_xray.yaml` | Dataset paths, class names, augmentation |
| `configs/train/fewshot_1shot.yaml` | 1-shot training settings |
| `configs/train/fewshot_4shot.yaml` | 4-shot training settings |
| `configs/train/fewshot_8shot.yaml` | 8-shot training settings |
| `configs/eval/cls.yaml` | Classification evaluation settings |
| `configs/eval/ood.yaml` | OOD detection evaluation settings |
| `configs/experiment/exp_*.yaml` | Pre-defined experiment configs |

---

## Script to Step Mapping

| Script | Step |
|--------|------|
| `src/scripts/generate_llm_descriptions.py` | OFFLINE: Generate disease descriptions |
| `src/scripts/build_fewshot_split.py` | OFFLINE: Build train/val/test splits |
| `src/scripts/train_fsl.py` | TRAINING: Train few-shot model |
| `src/scripts/eval_ood.py` | EVALUATION: OOD detection |
| `src/scripts/eval_cls.py` | EVALUATION: Classification (if separate) |

---

## Design Principles

1. **Config-driven**: All parameters from YAML. No hard-coded class names or hyperparameters.
2. **Modular**: Each component (encoder, refiner, selector, aligner) is a standalone module.
3. **Registry pattern**: Components are registered and instantiated via `src/utils/registry.py`.
4. **Frozen vs Trainable**: CLIP backbone and LLM are frozen. Only trainable modules are updated.
5. **Ablation-friendly**: Any component can be disabled via config (`trainable: false`).
6. **Extensible**: Easy to swap CLIP backbone (ViT-B/16 → ViT-L/14) or LLM (Qwen → LLaMA).

---

## Key Config Parameters to Know

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.clip.backbone` | ViT-B/16 | CLIP vision backbone |
| `model.llm.model_name` | Qwen2.5-7B | LLM for descriptions |
| `llm_descriptions.num_descriptions_per_class` | 5 | M = number of descriptions |
| `model.text_refiner.alpha` | 0.5 | Text refinement interpolation |
| `model.local_contrastive.top_k` | 4 | Disease-relevant regions |
| `model.local_contrastive.bottom_k` | 4 | Disease-irrelevant regions |
| `model.alignment.alpha_global` | 0.5 | Global alignment weight |
| `model.alignment.alpha_local` | 0.5 | Local alignment weight |
| `fewshot.k_shot` | 1 | Support samples per class |
| `fewshot.n_way` | 5 | Classes per episode |
| `train.optimizer.lr` | 0.0001 | Learning rate |
| `eval.ood.method` | msp | OOD detection method |
