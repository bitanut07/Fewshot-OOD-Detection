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
