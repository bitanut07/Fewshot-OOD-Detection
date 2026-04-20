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
