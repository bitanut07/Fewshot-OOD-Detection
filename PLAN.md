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

# Bug fix đã xử lý trong lúc chạy thực tế trên GPU

## CUDA `device-side assert triggered` khi train
Lỗi gốc: `GlobalAlignmentLoss` đang `mean(dim=1)` text 3D `[B, C, D]` rồi
`matmul` → logits `[B, B]`. Khi `num_classes > batch_size` (few-shot chỉ
có 6 mẫu train), class label (vd `5`) vượt ngoài số cột `B` → CrossEntropy
assert phía CUDA. Lỗi bị `try/except` âm thầm nuốt nên lan sang op kế
tiếp (`softmax` trong `entropy_select_topk`) mới crash.

Đã fix:
- `src/losses/alignment_loss.py::GlobalAlignmentLoss`:
  - Nhận đúng text 3D → tính per-sample cosine → logits `[B, num_classes]`.
  - Thêm CPU-side check label range, báo rõ nếu sai shape thay vì assert CUDA.
- `src/losses/total_loss.py::entropy_select_topk`:
  - Check labels nằm trong `[0, C)` trước khi softmax/topk.
- `src/trainer/train.py`:
  - Thêm `_remap_train_labels(...)` map label gốc → chỉ số liên tục theo
    `data.id_classes` (tránh out-of-range khi config `id_classes` không
    liên tục).
  - Bỏ `try/except Exception` bọc `ga_fn(...)` để lỗi không bị che.

# Eval standalone (GLOCAL-FSL-OOD)

## Script mới: `src/scripts/eval_ood.py`
Hoàn chỉnh lại từ script cũ (dùng API đã biến mất). Giờ:
- Load config + optional `--override`, set seed, chọn GPU.
- Build **ID test loader** bằng cùng stratified split với `train_fsl.py`
  (chung seed) → cùng tập test như lúc train.
- Build **OOD loader** từ `mode="ood"` của `BoneXRayDataset`
  (có thể tắt bằng `--no-ood`).
- Dựng `GLocalFSLOODModel` + `load_checkpoint(...)` từ file `.pt`.
- Gọi `src/trainer/test.test(...)` để tính:
  - ID classification: `accuracy, precision, recall, f1_score, auroc`.
  - OOD detection song song 3 phương pháp: `msp`, `glmcm`, `local_mcm`
    với `AUROC / AUPR-In / AUPR-Out / FPR@95`.
- In kết quả gọn ra stdout, dump JSON đầy đủ ra
  `outputs/eval/<experiment_name>.json` (hoặc path qua `--save-json`).

Flags chính:
- `--config` (bắt buộc) — experiment YAML
- `--override` — override YAML
- `--checkpoint` (bắt buộc) — file `.pt`
- `--gpu` — index GPU (default 0)
- `--batch-size` — override batch size eval
- `--temperature` — softmax temperature cho OOD scoring
- `--no-ood` — chỉ eval ID classification
- `--save-json` — đường dẫn lưu kết quả

Ví dụ lệnh:
```bash
# Eval chuẩn bằng best checkpoint
python src/scripts/eval_ood.py \
  --config configs/experiment/exp_full_model.yaml \
  --checkpoint outputs/runs/full_model/checkpoints/best.pt

# Chỉ ID classification, không OOD
python src/scripts/eval_ood.py \
  --config configs/experiment/exp_full_model.yaml \
  --checkpoint outputs/runs/full_model/checkpoints/best.pt \
  --no-ood

# Tuning temperature cho OOD scoring
python src/scripts/eval_ood.py \
  --config configs/experiment/exp_full_model.yaml \
  --checkpoint outputs/runs/full_model/checkpoints/best.pt \
  --temperature 0.5

# Lưu JSON tùy chọn đường dẫn
python src/scripts/eval_ood.py \
  --config configs/experiment/exp_full_model.yaml \
  --checkpoint outputs/runs/full_model/checkpoints/best.pt \
  --save-json outputs/eval/full_model_t0.5.json
```

## Quan hệ với các module eval đã có
| Đang dùng ở đâu                              | Vai trò                                                           |
| -------------------------------------------- | ----------------------------------------------------------------- |
| `src/trainer/validate.py::validate`          | Validate giữa epoch khi train (ID metrics + MSP sanity check)     |
| `src/trainer/test.py::test`                  | Test cuối run trong `Trainer.test()`, cũng dùng lại trong script  |
| `src/scripts/eval_ood.py`                    | Eval standalone (không cần train), đọc checkpoint + in/save JSON  |

# LLM disk-optimization cho GPU cloud (~32GB disk)

## Bối cảnh
- Node GPU thuê thường chỉ ~32GB disk.
- Qwen-7B FP16 ≈ 15GB → nếu HF cache mặc định (`~/.cache/huggingface`) thì
  disk hết chỗ, không ghi được checkpoint, I/O chậm, train crash.
- LLM chỉ dùng **một lần** để sinh description, không tham gia training.

## Thay đổi đã làm

### 1. Redirect HF cache sang `/tmp`
File mới: `src/models/encoders/text_generation/hf_env.py`
- `setup_hf_cache(path=None)` — set `HF_HOME`, `HF_HUB_CACHE`,
  `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE` về `path`
  (default `/tmp/hf-cache-fewshot-ood`). **Idempotent**, gọi bao nhiêu lần
  cũng được.
- `cleanup_hf_cache(path=None)` — xoá toàn bộ cache, trả về MB đã free.
- `cleanup_model_cache(model_id, path=None)` — xoá cache 1 model cụ thể.
- `hf_local_generator.py` gọi `setup_hf_cache()` **ngay đầu file**, trước
  `from transformers import ...`, nên mọi đường đi import đều auto-redirect.

### 2. Quantization + low-memory load
`src/models/encoders/text_generation/hf_local_generator.py`:
- Thêm `quantization: "4bit" | "8bit" | None` (qua `BitsAndBytesConfig`,
  `nf4` + double-quant cho 4-bit).
- Mặc định `low_cpu_mem_usage=True`, `device_map="auto"` — Accelerate stream
  weight thẳng sang GPU, không cần full copy trên CPU.
- Method `unload(delete_cache=None)`:
  - `.to("cpu")` model, `del` + `gc.collect()` + `torch.cuda.empty_cache()`,
  - tuỳ chọn xoá cache trên disk.
- Hỗ trợ context manager (`with HFLocalGenerator(...) as gen:`).

### 3. API backend (zero disk)
File mới: `src/models/encoders/text_generation/api_generator.py`
- `APITextGenerator(provider, model_name, api_key_env, base_url, ...)`.
- Provider: `openai`, `anthropic`, `openai-compatible` (vLLM/Groq/Together).
- Dùng SDK nếu có (`openai`, `anthropic`), fallback raw `requests` POST.
- API key đọc từ env var → **không commit key vào repo**.
- Không tải weight nào → 0 MB disk, 0 MB VRAM LLM.

### 4. Factory chọn backend
File mới: `src/models/encoders/text_generation/factory.py`
- `build_generator(llm_cfg)` đọc `use_local_llm`:
  - `True` → `HFLocalGenerator` (với quant + cleanup tùy chọn).
  - `False` → `APITextGenerator`.
- `release_generator(gen, delete_cache=None)` — helper gọi `unload(...)`
  đồng bộ cho cả hai backend.

### 5. Config flags mới
`configs/model/llm_qwen.yaml` (+ `configs/default.yaml` mirror):
```yaml
model:
  llm:
    use_local_llm: true            # false → dùng API
    cleanup_cache: true            # xoá weight sau khi sinh description
    model_name: "Qwen/Qwen2.5-7B-Instruct"
    device_map: "auto"
    torch_dtype: "float16"
    quantization: "4bit"           # "4bit" | "8bit" | null
    low_cpu_mem_usage: true
    cache_dir: "/tmp/hf-cache-fewshot-ood"
    trust_remote_code: true
    api:
      provider: "openai"           # hoặc "anthropic", "openai-compatible"
      model_name: "gpt-4o-mini"
      api_key_env: "OPENAI_API_KEY"
      base_url: null
      timeout: 60.0
```

### 6. Script `generate_llm_descriptions.py` an toàn disk
- Gọi `setup_hf_cache(verbose=True)` **đầu file** (trước mọi import HF).
- Dùng `build_generator(...)` thay vì khởi tạo `HFLocalGenerator` trực tiếp.
- Đặt generation trong `try / finally` → dù thành công hay crash vẫn chạy
  `release_generator(gen, delete_cache=...)` để free VRAM + (optionally) disk.
- Thêm CLI flags:
  - `--use-api` — ép dùng API backend cho lần chạy này.
  - `--force-cleanup` — luôn xoá cache sau khi xong.
  - `--no-cleanup` — giữ cache (debug / chạy lại nhiều lần).

### 7. Script xoá cache LLM độc lập
File mới: `src/scripts/cleanup_llm_cache.py` (quét đủ 6 vị trí HF cache
phổ biến: `/tmp/hf-cache-*`, `$HF_HOME`, `$HF_HUB_CACHE`,
`$HUGGINGFACE_HUB_CACHE`, `$TRANSFORMERS_CACHE`, `~/.cache/huggingface`,
`/root/.cache/huggingface`).
```bash
# Xem có cache ở đâu (KHÔNG xoá)
python src/scripts/cleanup_llm_cache.py --scan

# Xoá SẠCH mọi vị trí cache HF đã phát hiện (khuyến nghị trên cloud)
python src/scripts/cleanup_llm_cache.py --all --yes

# Chỉ xoá weight của 1 model, ở mọi vị trí
python src/scripts/cleanup_llm_cache.py --all --model "Qwen/Qwen2.5-7B-Instruct" --yes

# Xoá cache + prune checkpoint cũ (giữ best.pt + last.pt)
python src/scripts/cleanup_llm_cache.py --all --prune-checkpoints outputs/runs --yes

# Chỉ xoá /tmp cache mặc định (hành vi cũ)
python src/scripts/cleanup_llm_cache.py
```

## Disk usage — before vs after

| Giai đoạn                             | Trước (FP16, `~/.cache`) | Sau (4-bit + `/tmp` + cleanup) | Sau (API)   |
| ------------------------------------- | ------------------------ | ------------------------------ | ----------- |
| Download LLM lần đầu                  | ~15 GB (đĩa chính)       | ~15 GB nhưng ở `/tmp` (tmpfs)  | 0 GB        |
| Sau khi sinh description              | ~15 GB còn nguyên        | 0 GB (tự xoá)                  | 0 GB        |
| Dung lượng khả dụng trước training    | ~17 GB                   | ~30 GB                         | ~32 GB      |
| VRAM khi load LLM                     | ~14 GB                   | ~4 GB                          | 0 GB        |

## Workflow khuyến nghị trên cloud

```bash
# 1) Sinh description 1 lần (LLM nạp → sinh → tự xoá)
python src/scripts/generate_llm_descriptions.py \
  --config configs/experiment/exp_full_model.yaml \
  --force-cleanup

# 2) (Nếu lỡ còn cache sót) Xoá thủ công
python src/scripts/cleanup_llm_cache.py --also-home

# 3) Train bình thường, disk đã thoáng
python src/scripts/train_fsl.py \
  --config configs/experiment/exp_full_model.yaml --do-test
```

Hoặc dùng API (bỏ hẳn LLM local):
```bash
export OPENAI_API_KEY=sk-...
python src/scripts/generate_llm_descriptions.py \
  --config configs/experiment/exp_full_model.yaml --use-api
```

# Checkpoint disk-efficient (fix lỗi zip short-write)

## Lỗi gặp phải
```
RuntimeError: [enforce fail at inline_container.cc:672] .
unexpected pos 878616000 vs 878615888
```
Lỗi ghi file của `torch.save`: thiếu đúng 112 byte cuối của zip ~838MB
→ **đĩa hết chỗ giữa lúc ghi checkpoint**. Kết hợp với LLM cache 15GB
còn sót, mỗi epoch lại sinh thêm 1 file `epoch_N.pt` ~838MB → đầy đĩa.

## Fix đã làm
- `src/utils/checkpoint.py`:
  - `save_checkpoint(...)` ghi **atomic**: `torch.save` vào `<path>.tmp`
    rồi `os.replace` → nếu hết đĩa thì không để lại file hỏng.
  - Thêm `keep_only_best=True` và `prune_checkpoint_dir(keep=...)`:
    tự xoá mọi `*.pt` / `*.pt.tmp` khác không nằm trong whitelist.
- `src/trainer/trainer.py`:
  - Flag config `train.keep_only_best: true` (default).
  - Mỗi lần `save(...)` chỉ ghi **`last.pt`** (overwrite tại chỗ);
    khi `is_best=True` thêm **`best.pt`** (cũng overwrite).
  - Khi khởi tạo `Trainer`, tự `prune_checkpoint_dir(...)` để xoá
    checkpoint `epoch_N.pt` cũ từ run trước.
- `configs/default.yaml`:
```yaml
train:
  save_best: true
  keep_only_best: true   # chỉ giữ best.pt + last.pt, xoá mọi file khác
```

## Kết quả
- Disk checkpoint steady state ≈ 2 × 838MB = **~1.7GB** thay vì
  `N_epochs × 838MB` (40 epoch → ~33GB, chắc chắn đầy đĩa 32GB).
- Không còn `epoch_1.pt`, `epoch_5.pt`, ... tích luỹ trên đĩa.
- Atomic save tránh tạo file `.pt` hỏng khi disk đầy → lần sau không
  crash lúc `load_checkpoint`.

## Nếu cần quay lại save mỗi epoch (debug)
```yaml
train:
  keep_only_best: false
```
