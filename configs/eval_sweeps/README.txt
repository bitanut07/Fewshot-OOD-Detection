Sweep overrides dùng với:

  python src/scripts/train_fsl.py \
    --config configs/experiment/exp_magla_clip.yaml \
    --override configs/eval_sweeps/<file>.yaml \
    --do-test

– `*_s8.yaml`: giữ k_shot=8 từ exp_magla_clip (8 mẫu).
– `*_s16.yaml`: đặt fewshot.k_shot=16 (16 mẫu / class).

Tham số k trong bảng (entropy top-k OOD) = `ood.topk`.

Danh sách lệnh đầy đủ (sweep k, ablation, gợi ý X–12):

  scripts/run_eval_magla_tables.sh

