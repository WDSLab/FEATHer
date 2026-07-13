#!/bin/bash
# ============================================================================
# Clone baseline upstream repositories.
#
# Each baseline lives at baselines/<Name>/<Name>-main/ (gitignored). The
# matching wrapper.py adapts the upstream Model(configs) interface to our
# unified Model(seq_len, pred_len, d_model, ...) call.
#
# Idempotent: skips any baseline already cloned.
#
# Usage:
#     bash setup_baselines.sh           # clone all
#     bash setup_baselines.sh DLinear   # clone one
# ============================================================================

set -e

cd "$(dirname "$0")/baselines"

declare -A REPOS=(
  ["DLinear/DLinear-main"]="https://github.com/cure-lab/LTSF-Linear.git"
  ["PatchTST/PatchTST-main"]="https://github.com/yuqinie98/PatchTST.git"
  ["iTransformer/iTransformer-main"]="https://github.com/thuml/iTransformer.git"
  ["FITS/FITS-main"]="https://github.com/VEWOXIC/FITS.git"
  ["SparseTSF/SparseTSF-main"]="https://github.com/lss-1138/SparseTSF.git"
  ["TimeMixer/TimeMixer-main"]="https://github.com/kwuking/TimeMixer.git"
  ["TimesNet/TimesNet-main"]="https://github.com/thuml/Time-Series-Library.git"
  ["TQNet/TQNet-main"]="https://github.com/ACAT-SCUT/TQNet.git"
  ["LMS_AutoTSF/LMS_AutoTSF-main"]="https://github.com/mribrahim/LMS-TSF.git"
  ["DiPE_Linear/DiPE_Linear-main"]="https://github.com/wintertee/DiPE-Linear.git"
  ["MDMLP_EIA/MDMLP_EIA-main"]="https://github.com/zh1985csuccsu/MDMLP-EIA.git"
)

# Out-of-scope (cited in paper, not benchmarked):
#   S-Mamba       https://github.com/wzhwzhwzh0921/S-D-Mamba   (CUDA selective-scan, no MCU port)
#   TimeCMA       https://github.com/ChenxiLiu-HNU/TimeCMA     (frozen LLM features, > 1K params budget)

# Optional positional arg: clone only this baseline
TARGET_NAME="${1:-}"

for target in "${!REPOS[@]}"; do
  name="${target%%/*}"
  if [ -n "$TARGET_NAME" ] && [ "$name" != "$TARGET_NAME" ]; then
    continue
  fi

  # Make sure parent dir exists
  mkdir -p "$(dirname "$target")"

  if [ -d "$target" ]; then
    echo "[skip ] $target already present"
  else
    url="${REPOS[$target]}"
    echo "[clone] $url"
    echo "        -> $target"
    git clone --depth 1 "$url" "$target"
  fi
done

# ---------------------------------------------------------------------------
# Python deps that upstream baselines import at MODULE LOAD (not just runtime).
# iTransformer/LMS_AutoTSF/TimeMixer/TimesNet share a layers/SelfAttention_Family.py
# that does `from reformer_pytorch import LSHSelfAttention` at import time — so
# these 4 models fail with `No module named 'reformer_pytorch'` even though LSH
# attention is never used. Install it (no-op if already present).
if [ -z "$TARGET_NAME" ]; then
  echo ""
  echo "[deps ] ensuring reformer_pytorch (import-time dep of 4 baselines)"
  python -c "import reformer_pytorch" 2>/dev/null \
    && echo "        already installed" \
    || pip install "reformer_pytorch==1.4.4"
fi

echo ""
echo "All requested baselines are present under baselines/."
echo "Next: write baselines/<Name>/wrapper.py for each, register in"
echo "baselines/__init__.py, then run the smoke test:"
echo "    python run_forecast.py --model <Name> --data ETTh1 --pred_len 96 \\"
echo "        --num_seeds 1 --num_epochs 2 --exp_tag smoke"
