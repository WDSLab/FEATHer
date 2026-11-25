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
#     bash setup_baselines.sh FITS      # clone one
# ============================================================================

set -e

cd "$(dirname "$0")/baselines"

declare -A REPOS=(
  ["FITS/FITS-main"]="https://github.com/VEWOXIC/FITS.git"
  ["SparseTSF/SparseTSF-main"]="https://github.com/lss-1138/SparseTSF.git"
  ["MixLinear/MixLinear-main"]="https://github.com/aitianma/MixLinear.git"
  # CMoS ICML'25 (official, commit b696a0c; data is git-lfs, not needed)
  ["CMoS/CMoS-main"]="https://github.com/CSTCloudOps/CMoS.git"
  # TimeBase (ICML'25) and PhaseFormer (ICLR'26)
  ["TimeBase/TimeBase-main"]="https://github.com/hqh0728/TimeBase.git"
  ["PhaseFormer/PhaseFormer-main"]="https://github.com/neumyor/PhaseFormer_TSL.git"
)

# Out-of-scope (cited in paper, not benchmarked):
#   S-Mamba       https://github.com/wzhwzhwzh0921/S-D-Mamba   (CUDA selective-scan, no MCU port)
#   TimeCMA       https://github.com/ChenxiLiu-HNU/TimeCMA     (frozen LLM features, > 1K params budget)


# Upstream commits the paper's runs used. A fresh clone is checked out at exactly this commit.
declare -A PINS=(
  ["CMoS"]="b696a0c33b5ad8f03ad483d43b95fcb5564aa939"
  ["FITS"]="d040bb015b6299da26d879b90dd19c80fb72c160"
  ["MixLinear"]="71b5db62e38b1cb108faa1e1d4687287b4568f3b"
  ["PhaseFormer"]="ed1db61c6abfa9326d5ca2a56c6c4ba53ea592ab"
  ["SparseTSF"]="b8c2740eecc84d8095ffce49ba5acafe68e53bb8"
  ["TimeBase"]="369b330f3d77371fcc7e8c75c808d01330c40899"
)

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
    git clone "$url" "$target"
    pin="${PINS[$name]:-}"
    if [ -n "$pin" ]; then
      git -C "$target" checkout --quiet "$pin" && echo "        pinned at ${pin:0:7}"
    fi
  fi
done

echo ""
echo "All requested baselines are present under baselines/."
echo "Next: write baselines/<Name>/wrapper.py for each, register in"
echo "baselines/__init__.py (MODELS) + configs/<Model>.yaml, then run the smoke test:"
echo "    python scripts/train.py --model <Name> --data ETTh1 --pred_len 96 \\"
echo "        --seeds 2025 --num_epochs 2 --results_csv results/_incoming/smoke.csv"
