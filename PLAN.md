# PLAN

## Data parsing
- Tạo 2 file parse riêng cho 2 bộ dữ liệu:
  - BTXRD
  - FracAtlas
- Kết quả parse được lưu tại:
  - `data/processed/data_processing/btxrd.csv`
  - `data/processed/data_processing/fracatlas.csv`

## Parse BTXRD
- Gán nhãn theo 2 danh sách:
  - `USED_LABELS`
  - `UNUSED_LABELS`
- Cấu hình:
```python
USED_LABELS = [
    "giant cell tumor",
    "osteochondroma",
    "osteofibroma",
    "osteosarcoma",
    "simple bone cyst",
    "synovial osteochondroma",
]

UNUSED_LABELS = [
    "multiple osteochondromas",
    "other bt",
    "other mt",
]
```
- Ảnh không thuộc cả `USED_LABELS` và `UNUSED_LABELS` được gán là `no_tumor`.

## Parse FracAtlas
- Xử lý tương tự BTXRD gồm:
USED_LABELS = [
    "fractured",
]

UNUSED_LABELS = [
    "non-fractured",
]
- Kết quả parse lưu vào `data/processed/data_processing/fracatlas.csv`.

## File `splits_data.py`
- Gộp 2 file CSV parse từ BTXRD và FracAtlas.
- Tạo `manifest.csv` từ dữ liệu đã gộp.
- `manifest.csv` được lưu tại:
  - `data/processed/data_processing/manifest.csv`

## Class ID / OOD
- Dùng danh sách `id_class` và `ood_class` để tạo cột `class` trong `manifest.csv`.
- Nhãn cuối cùng sử dụng gồm:
  - `giant cell tumor`
  - `osteochondroma`
  - `osteofibroma`
  - `osteosarcoma`
  - `simple bone cyst`
  - `synovial osteochondroma`
  - `no_tumor`
  - `fractured`
- Cột `class` gồm:
  - `id`
  - `ood`

## Xử lý ảnh
- Resize toàn bộ ảnh về `224x224`.
- Kiểm tra tỷ lệ vùng trắng của ảnh sau resize.
- Ảnh có vùng trắng quá nhiều sẽ bị bỏ qua.
- Ảnh hợp lệ sẽ được tăng cường dữ liệu bằng:
  - ảnh gốc
  - xoay 90 độ
  - xoay 180 độ
  - xoay 270 độ

## Quy tắc đặt `name_id`
- Định dạng:
```text
IMG<mã dữ liệu><số thứ tự ảnh 5 chữ số><mã đa dạng>
```
- Mã dữ liệu:
  - `01`: BTXRD
  - `02`: FracAtlas
- Mã đa dạng:
  - `00`: ảnh gốc
  - `01`: xoay 90 độ
  - `02`: xoay 180 độ
  - `03`: xoay 270 độ

## Đầu ra
- Ảnh sau xử lý được lưu tại:
  - `data/processed/image_processing`
- `manifest.csv` gồm các cột:
  - `name_id`
  - `data_name`
  - `path`
  - `label`
  - `class`

## Thứ tự chạy
1. Chạy file parse của BTXRD
2. Chạy file parse của FracAtlas
3. Chạy file `splits_data.py`

## LLM description generation
- Đăng nhập và Hugging-Face: chạy lệnh `huggingface-cli login`

#Liên hệ với Admin để lấy GPU hoặc tự cấu hình GPU Cloud của mình.

Chạy lệnh dưới để bật server GPU Cloud:
ssh -i ~/.ssh/id_ed25519 -p 33528 root@116.109.174.203

scp -i ~/.ssh/id_ed25519 -P 33528 root@116.109.174.203:/workspace/Fewshot-OOD-Detection/data/prompts/class_descriptions.json .



cd Fewshot-OOD-Detection

Lệnh generate descriptions:

python src/scripts/generate_llm_descriptions.py --config configs/experiment/exp_full_model.yaml --num_attributes 6 --num_descriptions 8 --force

# Descriptions Logic

## Những gì đã làm cho `generate_llm_descriptions`
- Refactor script sang pipeline multi-stage, chỉ generate cho ID classes:
  1. Generate câu hỏi phân biệt ở mức dataset
  2. Stage A: generate attributes theo từng class
  3. Stage B: generate descriptions từ attributes + cross-class awareness
  4. Filter/score/post-process + targeted retry
  5. Save YAML/JSON + quality report
- Thêm package `src/models/encoders/text_generation/` để tách module rõ ràng:
  - `hf_local_generator.py`: backend HF local
  - `prompt_builder.py`: prompt question/attribute/description/retry
  - `output_cleaner.py`: lọc generic, length, domain keyword, language
  - `description_scorer.py`: scoring class-aware + diversity selection
  - `cache_manager.py`: cache-first load/save + schema v3
  - `llm_wrapper.py`: orchestration pipeline
- Quality report chi tiết theo candidate:
  - kept/rejected
  - reason reject
  - score breakdown
  - class_rule_hits
  - stage + attempt
  - regeneration_triggered/class_min_satisfied/failure_reason

## Rule và kiểm soát chất lượng hiện tại
- Chỉ ID classes, không có OOD generation path.
- Class-aware validation (preferred/suspicious/forbidden) cho:
  - giant cell tumor
  - osteochondroma
  - osteofibroma
  - osteosarcoma
  - simple bone cyst
  - synovial osteochondroma
- Lọc non-English/mixed-language fragment.
- Lọc linh hoạt hơn cho CLIP-style ngắn:
  - word range hiện tại: `6..22`
- Diversity control theo feature category:
  - cortical / trabecular / periosteal / lesion_pattern / matrix / location / shape_growth / soft_tissue
- Class survival guarantee:
  - tối thiểu `5` descriptions/class (`--min_final_descriptions`)
  - nếu thiếu sẽ targeted retry tự động

## Cấu hình/lệnh chạy có thể dùng

### 1) Generate cơ bản (cache-first)
```bash
python src/scripts/generate_llm_descriptions.py --config configs/experiment/exp_full_model.yaml
```

### 2) Force generate lại từ đầu
```bash
python src/scripts/generate_llm_descriptions.py --config configs/experiment/exp_full_model.yaml --force
```

### 3) Chạy đầy đủ với quality report
```bash
python src/scripts/generate_llm_descriptions.py \
  --config configs/experiment/exp_full_model.yaml \
  --num_questions 10 \
  --num_attributes 6 \
  --num_descriptions 8 \
  --min_final_descriptions 5 \
  --quality_report quality_report.json \
  --force
```

### 4) Chạy deterministic để dễ reproducible
```bash
python src/scripts/generate_llm_descriptions.py \
  --config configs/experiment/exp_full_model.yaml \
  --seed 42 \
  --deterministic \
  --force
```

### 5) Reuse câu hỏi cũ
```bash
python src/scripts/generate_llm_descriptions.py \
  --config configs/experiment/exp_full_model.yaml \
  --skip_questions \
  --force
```

# Trainer (GLOCAL-FSL-OOD)

## Tổng quan các giai đoạn trainer
Pipeline trainer đã được hoàn thành, bám theo cấu trúc/logic của
`glali/trainers/locproto_supc.py` nhưng refactor về module hóa:

1. `src/trainer/train.py` - training epoch (multi-loss, AMP, grad clip)
2. `src/trainer/validate.py` - validation + optional OOD sanity check
3. `src/trainer/test.py` - test + OOD evaluation (MSP, GL-MCM, local-MCM)
4. `src/trainer/trainer.py` - orchestrator (optimizer, scheduler, AMP, checkpoint, resume)
5. `src/scripts/train_fsl.py` - entry point đọc manifest, build loader, chạy trainer

## Những gì đã làm cho trainer

### 1) Data layer
- Viết lại `src/datasets/bone_xray_dataset.py` để load trực tiếp từ
  `data/processed/data_processing/manifest.csv`.
- Hỗ trợ `mode = id | ood | all` để tách ID-only, OOD-only hoặc lấy cả.
- `class_names`, `id_classes`, `ood_classes` config-driven.
- Tự map `label -> class_idx` và bỏ qua ảnh có label không nằm trong list.
- Cung cấp `get_default_transform("train"|"test")` (Resize 224, flip,
  rotate, normalize ImageNet).
- Hỗ trợ optional `split_file` (one `name_id` per line) để cố định
  train/val/test.

### 2) Model framework (bổ sung để hỗ trợ trainer)
File: `src/models/framework/glocal_fsl_ood_model.py`
- Thêm **teacher image encoder** (deep-copy của student, frozen) giống
  `zs_img_encoder` của glali — dùng cho distillation loss.
- Forward pass trả về đầy đủ các trường cần cho trainer:
  - `logits`           — global + local kết hợp (cho classification chính)
  - `global_logits`    — chỉ global branch
  - `local_logits`     — per-patch `[B, P, C]` (cho OOD entropy reg)
  - `global_feat`, `global_feat_teacher` — cho distillation L1
  - `text_for_align`, `class_prototypes` — cho text distillation
  - `relevant_features`, `irrelevant_features` — cho local contrastive
  - `loss_contrastive` — SupCon-like loss khi `return_loss=True`
- `freeze_encoders()` đóng băng cả text encoder, image encoder và teacher.
- Config getter an toàn (hỗ trợ cả `config.model.*` và `config.*` top-level).

### 3) Loss tổng
File: `src/losses/total_loss.py`
- `TotalLoss` gom 8 loss hạng tử (optional, chỉ cộng khi weight > 0):
  - `loss_cls`                 — classification (cross-entropy)
  - `loss_global_alignment`    — InfoNCE global image↔text
  - `loss_local_alignment`     — cross-entropy trên local logits (glali `loss_id2`)
  - `loss_local_contrastive`   — SupCon local (glali `loss_supc`)
  - `loss_text_refinement`
  - `loss_ood_reg`             — entropy reg trên non-topK patch (glali `entropy_select_topk`)
  - `loss_distill_img`         — L1 teacher↔student image feature (glali `loss_distil_img`)
  - `loss_distill_text`        — L1 refined text ↔ raw prototype (glali `loss_distil_text`)
- Hàm `entropy_select_topk(local_logits, labels, top_k)` reimplement
  đúng logic của glali nhưng làm việc trên tensor `[B, P, C]` trực tiếp
  (không cần flatten thủ công trong trainer).

### 4) Stage `train` (`src/trainer/train.py`)
- Hỗ trợ AMP (`torch.amp.GradScaler`) + gradient clipping + AdamW/SGD.
- Tính toàn bộ 8 loss trong 1 forward, tổng hợp qua `TotalLoss`.
- Logging per-batch: loss, cls loss, accuracy; đẩy scalars ra TensorBoard
  khi có `tb_logger`.
- `global_step` được chuyển tiếp qua các epoch (resume-friendly).
- `freeze_encoders()` được gọi mỗi epoch để đảm bảo CLIP backbone luôn
  đóng băng (match glali `turn off gradients`).

### 5) Stage `validate` (`src/trainer/validate.py`)
- Chạy `model.eval()`, tính classification metrics trên **restricted
  ID logits** (index theo `data.id_classes`).
- Tính thêm MSP score trên toàn bộ logits để cung cấp OOD sanity check
  khi có `ood_loader` (AUROC / AUPR / FPR@95).
- Relabel `labels -> 0..num_id_classes-1` để tương thích metric.
- Đẩy scalars `val/*` ra TensorBoard.

### 6) Stage `test` (`src/trainer/test.py`)
- Classification metrics trên ID test loader (restricted ID logits).
- OOD detection với 3 phương pháp (giống glali):
  - `msp`       — `-max softmax(global_logits)`
  - `glmcm`     — `-max softmax(combined_logits)`
  - `local_mcm` — `-max softmax(local_logits_per_patch)` (amax dim=(1,2))
- Temperature scaling cho softmax (từ `ood.temperature`).
- Trả về dict gồm `accuracy, precision, recall, f1_score, auroc,
  predictions, labels, ood.{msp,glmcm,local_mcm}`.

### 7) Orchestrator `Trainer` (`src/trainer/trainer.py`)
- Build optimizer (AdamW/Adam/SGD) + scheduler (Cosine với linear warmup,
  Step, hoặc None) từ config.
- Quản lý AMP scaler, grad clip, seed, checkpoint dir.
- `train()`:
  - lặp epoch → train → optional validate → optional save checkpoint
  - theo dõi best model theo `val_accuracy`, save `best.pt`.
- `test(use_best=True)`: auto load `best.pt` nếu có rồi chạy `test()`.
- `resume(path)`: restore model + optimizer + scheduler + scaler + epoch
  + best metric + global_step.

### 8) Entry point `train_fsl.py` (`src/scripts/train_fsl.py`)
- Load `configs/default.yaml` + experiment config + optional `--override`.
- Auto-detect `manifest.csv` (thứ tự: `data.manifest_file` → 2 path mặc định).
- Build `BoneXRayDataset` ba lần (train-transform cho train subset,
  test-transform cho val/test subset, ood mode cho ood subset).
- Stratified split theo class (val_ratio / test_ratio từ config).
- Optional k-shot capping (`fewshot.k_shot > 0` → giữ k mẫu/class).
- Load LLM descriptions YAML (fallback JSON của glali).
- Log dir / TB dir tự tạo theo `experiment_name`.
- Flags:
  - `--config`      (bắt buộc) — experiment YAML
  - `--override`    — override YAML
  - `--resume`      — checkpoint file
  - `--do-test`     — chạy test sau khi train
  - `--eval-only`   — bỏ qua train, chỉ test

### 9) Config bổ sung (`configs/default.yaml`)
- Thêm `loss.ood_reg.weight` (mặc định `0.25`).
- Thêm `loss.distill_img.weight` (mặc định `10.0`, khớp glali).
- Thêm `loss.distill_text.weight` (mặc định `0.0`, bật thủ công).
- Thêm `ood.topk = 50`, `ood.temperature = 1.0`.
- Thêm block `experiment`, `paths`, `logging`, `data.manifest_file`,
  `data.val_ratio`, `data.test_ratio`, `data.num_workers`, `data.pin_memory`.

## Lệnh chạy

### Update mới đã làm (liên quan train pipeline)
- Bổ sung `src/scripts/download_dataset.py` hỗ trợ 2 nguồn:
  - `source: figshare` (download `.zip` từ DOI)
  - `source: kaggle` (qua `kagglehub.dataset_download`)
- Thêm cleanup tự động sau tải:
  - unzip xong thì tự flatten lớp thư mục bọc ngoài (nếu có),
  - tự xóa file `.zip` sau khi giải nén thành công.
- Chuẩn hóa config tải dataset để khớp pipeline parse/train:
  - `configs/data/download_btxrd.yaml` -> dữ liệu về `data/raw/BTXRD`.
  - `configs/data/download_fracatlas.yaml` -> dữ liệu về `data/raw/FracAtlas`.
  - Có `kaggle_subpath` để tránh lỗi lồng thư mục kiểu `data/raw/.../...`.
- Cải thiện thông báo lỗi config:
  - nếu YAML thiếu key top-level `dataset`, script báo lỗi rõ ràng thay vì `KeyError`.

### Train + test một lượt
```bash
python src/scripts/train_fsl.py \
  --config configs/experiment/exp_full_model.yaml \
  --do-test
```

### Train với override few-shot
```bash
python src/scripts/train_fsl.py \
  --config configs/experiment/exp_full_model.yaml \
  --override configs/train/fewshot_4shot.yaml
```

### Resume training
```bash
python src/scripts/train_fsl.py \
  --config configs/experiment/exp_full_model.yaml \
  --resume outputs/runs/full_model/checkpoints/epoch_20.pt
```

### Eval-only (dùng best checkpoint)
```bash
python src/scripts/train_fsl.py \
  --config configs/experiment/exp_full_model.yaml \
  --eval-only --do-test
```

### Câu lệnh chạy full pipeline (khuyến nghị hiện tại)
```bash
# 1) Download raw datasets (KaggleHub)
python src/scripts/download_dataset.py configs/data/download_btxrd.yaml
python src/scripts/download_dataset.py configs/data/download_fracatlas.yaml

# 2) Parse từng nguồn
python src/scripts/parse_btxrd.py
python src/scripts/parse_fracatlas.py

# 3) Merge + split + tạo manifest/split files
python src/scripts/splits_dataset.py
python src/scripts/build_fewshot_split.py --config configs/data/splits_data.yaml

# 4) Train + test
python src/scripts/train_fsl.py --config configs/experiment/exp_full_model.yaml --do-test
```

## Mapping logic với glali
| glali (`locproto_supc.py`)            | Dự án hiện tại                                                            |
| ------------------------------------- | ------------------------------------------------------------------------- |
| `zs_img_encoder` (frozen copy)        | `GLocalFSLOODModel.teacher_image_encoder`                                 |
| `loss_id = CE(output, label)`         | `loss_cls` trên `logits` (`TotalLoss.weight_cls`)                         |
| `loss_id2 = CE(output_local, label)`  | `loss_local_alignment` (mean-pool local_logits, `TotalLoss.weight_la`)    |
| `loss_distil_img`                     | `loss_distill_img` (L1 teacher↔student global feat)                       |
| `loss_distil_text`                    | `loss_distill_text` (L1 refined↔raw prototype)                            |
| `loss_supc` (SupCon)                  | `loss_local_contrastive` (SupCon-like trên relevant/irrelevant)           |
| `entropy_select_topk` (non-topK)      | `loss_ood_reg` qua `entropy_select_topk` trong `total_loss.py`            |
| `test_ood` (MCM/GL-MCM/local-MCM)     | `src/trainer/test.py` tính 3 scoring song song                            |
| `TrainerX` (dassl)                    | `src/trainer/trainer.Trainer` (orchestrator độc lập, không dùng dassl)    |

## Smoke test đã chạy
- Load `manifest.csv`: ✅ 5,764 ID samples + 10,384 OOD samples.
- Forward pass model (CLIP ViT-B/16, CPU): ✅ trả về đủ `logits` `[B,8]`,
  `local_logits` `[B,196,8]`, `global_feat` `[B,512]`.
- 1-epoch training end-to-end (CPU, 3 batches): ✅ loss giảm, accuracy
  update, checkpoint ghi ra `outputs/runs/full_model/checkpoints/`.
- Test stage với 3 OOD methods (msp/glmcm/local_mcm): ✅ trả về
  AUROC/AUPR/FPR cho từng method.
