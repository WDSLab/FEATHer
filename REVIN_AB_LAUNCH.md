# RevIN-fragility A/B — WindSCADA/EWELD expansion (LAUNCH SHEET)

Goal: verify whether the Steel finding (`no_revin` → clean −13% **and** robust
Dw 1.20→1.08) reproduces on WindSCADA/EWELD, to decide **removal vs robust-RevIN**.

Flags already wired (personal push `ffbffde`): `--no_revin`, `--gate_uniform`,
`--use_residual_mlp --mlp_rank 16`. **NEW (2026-08-01): `--robust_revin`** =
median/MAD instance-norm (zero extra params, canonical byte-identical when off;
forwarded by both orchestrators to the worker so eval-time forward matches
training). Arm `feather_mlp_robustrevin` = MLP-on + robust-RevIN = the proposed
fix; compare vs `feather_mlp16` (standard RevIN) and `feather_mlp_norevin`
(removal). Arms are MLP-on except `feather_base`.
Checkpoints keyed by `exp_tag`; robustness loads by `train_exp_tag`. **Robustness
flags MUST mirror training** (else eval-time forward diverges).

CSVs (isolated): accuracy → `results/feather_mlp_ab.csv`,
robustness → `results/robust_mlp_ab.csv`. Robustness: **`--jobs_per_gpu 2`**
(torn-row NaN-tolerant resume is in, but fewer torn rows = cleaner).

Current gaps: WindSCADA needs {norevin, gateoff}; EWELD needs {base, mlp16,
norevin, gateoff} — EWELD is absent from both A/B CSVs.

---
## STEP 1 — train + save checkpoints  → results/feather_mlp_ab.csv
(Linux old server, bash. On Windows new server swap `\`→backtick.)

```bash
# WindSCADA: fill the two missing arms
python run_forecast.py --model FEATHer --data WindSCADA --exp_tag feather_mlp_norevin \
  --results_csv results/feather_mlp_ab.csv --use_residual_mlp --mlp_rank 16 --no_revin   --num_seeds 3 --save_model
python run_forecast.py --model FEATHer --data WindSCADA --exp_tag feather_mlp_gateoff \
  --results_csv results/feather_mlp_ab.csv --use_residual_mlp --mlp_rank 16 --gate_uniform --num_seeds 3 --save_model

# EWELD: all four arms
python run_forecast.py --model FEATHer --data EWELD --exp_tag feather_base \
  --results_csv results/feather_mlp_ab.csv                                    --num_seeds 3 --save_model
python run_forecast.py --model FEATHer --data EWELD --exp_tag feather_mlp16 \
  --results_csv results/feather_mlp_ab.csv --use_residual_mlp --mlp_rank 16              --num_seeds 3 --save_model
python run_forecast.py --model FEATHer --data EWELD --exp_tag feather_mlp_norevin \
  --results_csv results/feather_mlp_ab.csv --use_residual_mlp --mlp_rank 16 --no_revin   --num_seeds 3 --save_model
python run_forecast.py --model FEATHer --data EWELD --exp_tag feather_mlp_gateoff \
  --results_csv results/feather_mlp_ab.csv --use_residual_mlp --mlp_rank 16 --gate_uniform --num_seeds 3 --save_model
```

## STEP 2 — robustness eval  → results/robust_mlp_ab.csv  (--jobs_per_gpu 2)
Run only after STEP 1 checkpoints for the same arm exist.

```bash
# WindSCADA (two new arms)
python run_robustness.py --model FEATHer --data WindSCADA --train_exp_tag feather_mlp_norevin \
  --exp_tag robust --results_csv results/robust_mlp_ab.csv --use_residual_mlp --mlp_rank 16 --no_revin   --jobs_per_gpu 2
python run_robustness.py --model FEATHer --data WindSCADA --train_exp_tag feather_mlp_gateoff \
  --exp_tag robust --results_csv results/robust_mlp_ab.csv --use_residual_mlp --mlp_rank 16 --gate_uniform --jobs_per_gpu 2

# EWELD (all four arms)
python run_robustness.py --model FEATHer --data EWELD --train_exp_tag feather_base \
  --exp_tag robust --results_csv results/robust_mlp_ab.csv                                                 --jobs_per_gpu 2
python run_robustness.py --model FEATHer --data EWELD --train_exp_tag feather_mlp16 \
  --exp_tag robust --results_csv results/robust_mlp_ab.csv --use_residual_mlp --mlp_rank 16               --jobs_per_gpu 2
python run_robustness.py --model FEATHer --data EWELD --train_exp_tag feather_mlp_norevin \
  --exp_tag robust --results_csv results/robust_mlp_ab.csv --use_residual_mlp --mlp_rank 16 --no_revin    --jobs_per_gpu 2
python run_robustness.py --model FEATHer --data EWELD --train_exp_tag feather_mlp_gateoff \
  --exp_tag robust --results_csv results/robust_mlp_ab.csv --use_residual_mlp --mlp_rank 16 --gate_uniform  --jobs_per_gpu 2
```

## STEP 1c/2c — robust-RevIN arm (NEW, the proposed fix)
After `git pull` (needs the 2026-08-01 `--robust_revin` code). Run on the 3
robustness datasets so each has the 3-way {mlp16, norevin, robustrevin}.
Single-line (Windows/new-server friendly).

```bash
# --- train + ckpt  -> feather_mlp_ab.csv ---
python run_forecast.py --model FEATHer --data Steel     --exp_tag feather_mlp_robustrevin --results_csv results/feather_mlp_ab.csv --use_residual_mlp --mlp_rank 16 --robust_revin --num_seeds 3 --save_model
python run_forecast.py --model FEATHer --data WindSCADA --exp_tag feather_mlp_robustrevin --results_csv results/feather_mlp_ab.csv --use_residual_mlp --mlp_rank 16 --robust_revin --num_seeds 3 --save_model
python run_forecast.py --model FEATHer --data EWELD     --exp_tag feather_mlp_robustrevin --results_csv results/feather_mlp_ab.csv --use_residual_mlp --mlp_rank 16 --robust_revin --num_seeds 3 --save_model

# --- robustness eval  -> robust_mlp_ab.csv  (--jobs_per_gpu 2) ---
python run_robustness.py --model FEATHer --data Steel     --train_exp_tag feather_mlp_robustrevin --exp_tag robust --results_csv results/robust_mlp_ab.csv --use_residual_mlp --mlp_rank 16 --robust_revin --jobs_per_gpu 2
python run_robustness.py --model FEATHer --data WindSCADA --train_exp_tag feather_mlp_robustrevin --exp_tag robust --results_csv results/robust_mlp_ab.csv --use_residual_mlp --mlp_rank 16 --robust_revin --jobs_per_gpu 2
python run_robustness.py --model FEATHer --data EWELD     --train_exp_tag feather_mlp_robustrevin --exp_tag robust --results_csv results/robust_mlp_ab.csv --use_residual_mlp --mlp_rank 16 --robust_revin --jobs_per_gpu 2
```
Decision extends to a 3-way: robust-RevIN wins if its Dw ≈ norevin (robust) AND
its clean MSE ≥ mlp16 / not worse than norevin — especially on EWELD if removal
hurts clean there. `--robust_revin` MUST be present in BOTH the train and the
robustness command for a given arm (else eval forward diverges from training).

## STEP 3 — verdict
```bash
PYTHONUTF8=1 python tools/audit/revin_ab_summary.py
# (local, reads Dropbox-synced CSVs by default; or pass --results_dir)
```
Reads the DECISION block. Interpretation:
- **REMOVAL WINS on WindSCADA+EWELD too** → RevIN is just harmful; simplest fix =
  drop it (weaker novelty, but clean+robust double win holds broadly).
- **NEED ROBUST-RevIN on some dataset** (Dw drops but clean HURTS) → median/MAD
  robust-RevIN is the justified general fix = **the paper contribution → build it.**
- **no fragility off Steel** (baseline Dw ~1.0, e.g. WindSCADA base=1.008) →
  the "RevIN fragility" spine is Steel-specific → rethink the novelty framing.

Note already-in-hand: WindSCADA base Dw=1.008 (robust) but mlp16=1.085 — on
WindSCADA the MLP, not RevIN, adds the small fragility. Watch this in the verdict.
