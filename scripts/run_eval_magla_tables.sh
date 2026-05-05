#!/usr/bin/env bash
# =============================================================================
# Lệnh train/eval theo các bảng tham chiếu (MAGLA-CLIP: sweep k, 8 vs 16 mẫu,
# ablation Table 4.7, gợi ý MAGLA(X–12)).
#
# Chạy từ thư mục gốc repo:
#   chmod +x scripts/run_eval_magla_tables.sh
#   ./scripts/run_eval_magla_tables.sh          # in toàn bộ lệnh (dry-run)
#
# Hoặc copy từng khối vào terminal.
#
# Checkpoint train: outputs/runs/<experiment_name>/checkpoints/best.pt
# JSON eval:       outputs/eval/<experiment_name>.json
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CFG_MAGLA="configs/experiment/exp_magla_clip.yaml"
CFG_BASELINE="configs/experiment/exp_baseline_clip.yaml"
PY_TRAIN="python src/scripts/train_fsl.py"
PY_EVAL="python src/scripts/eval_ood.py"

echo "=============================================="
echo "Repo: $ROOT"
echo "=============================================="

# -----------------------------------------------------------------------------
# 1) Bảng MAGLA-CLIP: tham số k (entropy / OOD top-k) × 8 mẫu hoặc 16 mẫu/class
#    k ↔ config: ood.topk  |  8 mẫu ↔ k_shot=8 (mặc định exp_magla_clip)
#              |  16 mẫu ↔ fewshot.k_shot=16 (override *_s16.yaml)
# -----------------------------------------------------------------------------

echo ""
echo "########## 1a) Sweep k ∈ {50,60,70,80,90} — 8 mẫu/class (train + test) ##########"
for K in 50 60 70 80 90; do
  echo "${PY_TRAIN} --config ${CFG_MAGLA} --override configs/eval_sweeps/magla_k${K}_s8.yaml --do-test"
done

echo ""
echo "########## 1b) Sweep k — 16 mẫu/class ##########"
for K in 50 60 70 80 90; do
  echo "${PY_TRAIN} --config ${CFG_MAGLA} --override configs/eval_sweeps/magla_k${K}_s16.yaml --do-test"
done

echo ""
echo "########## 1c) Eval lại (đã có checkpoint): ví dụ k=50, 8-shot ##########"
echo "${PY_EVAL} --config ${CFG_MAGLA} --override configs/eval_sweeps/magla_k50_s8.yaml \\"
echo "  --checkpoint outputs/runs/magla_clip_k50_s8/checkpoints/best.pt"

# -----------------------------------------------------------------------------
# 2) Ablation Table 4.7 (map sang module repo)
#    (1) Linear probe — repo chưa có head tách; gần nhất: baseline CLIP
#    (2) Template text only  → ablation_02_template_only.yaml
#    (3) + LLM desc, chưa refinement → ablation_03_llm_no_refiner.yaml
#    (4) + Text refinement, chưa “adapter” local → ablation_04_refiner_no_adapter.yaml
#    (5) Full MAGLA        → exp_magla_clip.yaml (không override ablation)
# -----------------------------------------------------------------------------

echo ""
echo "########## 2) Ablation (train + test) ##########"
echo ""
echo "# (1) Baseline / linear-probe-style (CLIP + chỉnh logits global; không pipeline MAGLA đầy đủ)"
echo "${PY_TRAIN} --config ${CFG_BASELINE} --do-test"
echo ""
echo "# (2) Template only — không LLM descriptions"
echo "${PY_TRAIN} --config ${CFG_MAGLA} --override configs/eval_sweeps/ablation_02_template_only.yaml --do-test"
echo ""
echo "# (3) LLM descriptions — tắt Text Refiner"
echo "${PY_TRAIN} --config ${CFG_MAGLA} --override configs/eval_sweeps/ablation_03_llm_no_refiner.yaml --do-test"
echo ""
echo "# (4) Text Refiner — tắt local contrastive (map “Adapter” phần local)"
echo "${PY_TRAIN} --config ${CFG_MAGLA} --override configs/eval_sweeps/ablation_04_refiner_no_adapter.yaml --do-test"
echo ""
echo "# (5) Full MAGLA-CLIP"
echo "${PY_TRAIN} --config ${CFG_MAGLA} --do-test"

echo ""
echo "########## Ablation — eval-only (ví dụ ablation 02) ##########"
echo "${PY_EVAL} --config ${CFG_MAGLA} --override configs/eval_sweeps/ablation_02_template_only.yaml \\"
echo "  --checkpoint outputs/runs/ablation_02_template_only/checkpoints/best.pt"

# -----------------------------------------------------------------------------
# 3) Bảng MAGLA-CLIP (X–12): số tham số / khối ViT train được
#    Hiện CLIPImageEncoder chỉ freeze toàn backbone; chưa có config “unfreeze
#    từ block X đến 12”. Khi triển khai, thêm ví dụ:
#      model.clip.unfreeze_from_block: 9
#    và tạo override tương tự các file magla_k*_s8.yaml.
# -----------------------------------------------------------------------------

echo ""
echo "########## 3) MAGLA-CLIP (5–12) … (12–12) — placeholder ##########"
echo "# Chưa có lệnh tự động; cần mở khóa ViT theo dải layer trong encoder."
echo "# Full run hiện tại (đối chiếu thời gian/200 epoch):"
echo "${PY_TRAIN} --config ${CFG_MAGLA} --do-test"

echo ""
echo "=============================================="
echo "Override nhỏ nằm trong: configs/eval_sweeps/"
echo "Few-shot 16 mẫu/base: configs/train/fewshot_16shot.yaml"
echo "Template-only prompts: data/prompts/ablation_template_only.yaml"
echo "=============================================="
