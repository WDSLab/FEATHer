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
  # Edge-tier additions (2026-07-31): MixLinear ICLR'26 (0.1K), CycleNet NeurIPS'24
  ["MixLinear/MixLinear-main"]="https://github.com/aitianma/MixLinear.git"
  ["CycleNet/CycleNet-main"]="https://github.com/ACAT-SCUT/CycleNet.git"
  ["MDMLP_EIA/MDMLP_EIA-main"]="https://github.com/zh1985csuccsu/MDMLP-EIA.git"
  # 2026-10-06: CMoS ICML'25 (official, commit b696a0c; data is git-lfs, not needed)
  ["CMoS/CMoS-main"]="https://github.com/CSTCloudOps/CMoS.git"
  # 2026-10-06: completeness check additions
  ["TimeBase/TimeBase-main"]="https://github.com/hqh0728/TimeBase.git"
  ["PhaseFormer/PhaseFormer-main"]="https://github.com/neumyor/PhaseFormer_TSL.git"
  ["SAMformer/SAMformer-main"]="https://github.com/romilbert/samformer.git"
)

# Out-of-scope (cited in paper, not benchmarked):
#   S-Mamba       https://github.com/wzhwzhwzh0921/S-D-Mamba   (CUDA selective-scan, no MCU port)
#   TimeCMA       https://github.com/ChenxiLiu-HNU/TimeCMA     (frozen LLM features, > 1K params budget)


# Upstream commits the paper's runs used (10-07). A fresh clone is checked out at exactly this commit.
declare -A PINS=(
  ["CMoS"]="b696a0c33b5ad8f03ad483d43b95fcb5564aa939"
  ["CycleNet"]="d807e51fc2dcd143885ee639d97965a7ab0926f4"
  ["DLinear"]="0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6"
  ["DiPE_Linear"]="b26e25015fba2c6a32e1f4eae8d337ccd9fe4067"
  ["FITS"]="d040bb015b6299da26d879b90dd19c80fb72c160"
  ["LMS_AutoTSF"]="412884a19e162a069825e5c68efcb875742dd22f"
  ["MDMLP_EIA"]="0d68442c014a4f796d3396a879d1bf08c260c5b5"
  ["MixLinear"]="71b5db62e38b1cb108faa1e1d4687287b4568f3b"
  ["PatchTST"]="204c21efe0b39603ad6e2ca640ef5896646ab1a9"
  ["PhaseFormer"]="ed1db61c6abfa9326d5ca2a56c6c4ba53ea592ab"
  ["SAMformer"]="71f10eaa696f2a098798779ee14b6ecd6b69bcd9"
  ["SparseTSF"]="b8c2740eecc84d8095ffce49ba5acafe68e53bb8"
  ["TQNet"]="15e19cb23483ed52398566c4baa959168cfffa57"
  ["TimeBase"]="369b330f3d77371fcc7e8c75c808d01330c40899"
  ["TimeMixer"]="e24610583b36fdd8c76cc17a8df4e65759a5f460"
  ["TimesNet"]="4e938a1767106324dd753b2a44832bf870a0252e"
  ["iTransformer"]="c2426e68ca13f74aaec08045c5c724d8ad328124"
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
echo "baselines/__init__.py (MODELS) + configs/<Model>.yaml, then run the smoke test:"
echo "    python scripts/train.py --model <Name> --data ETTh1 --pred_len 96 \\"
echo "        --seeds 2025 --num_epochs 2 --exp_tag smoke --results_csv results/_incoming/smoke.csv"
