# Robustness track (archived 2026-08-24)

Sensor-fault robustness was a paper pillar until the measurements came in:
across the edge-7 lineup on the 6 manufacturing datasets, FEATHer degraded
**worst** — relative degradation Dw 1.211 vs 1.006–1.034 for the peers, and
worst absolute corrupt-MSE too (7/7 on both). Keeping the claim would have
argued against us, so the track was dropped from the JMS submission
(2026-08-04 decision) and the code moved here.

The diagnosis that came out of it is still useful and is cited in the paper's
discussion: the fragility traced to **RevIN**, not the FFT gate — corrupted
windows poison the instance mean/std that RevIN computes and then inverts, so
an impulse spike is amplified twice. Turning the gate off changed nothing
(Dw 1.202 → 1.200) while disabling RevIN dropped it to 1.076.

## What is here

| path | was |
|---|---|
| `run_robustness.py` | root orchestrator (checkpoint × 20 corruption conditions) |
| `scripts/run_robustness_worker.py` | inference-only worker |
| `tools/robust_summary.py` | μ_rel / AUC / mCE-vs-SparseTSF / degradation curves |
| `tools/gate_logging.py` | per-band FFT gate weights under corruption |
| `tools/revin_ab_summary.py` | RevIN A/B aggregation (the diagnosis above) |
| `utils/noise.py` | 4-axis corruption (gauss / miss / impulse / quant) |
| `utils/severity.py` | torch-free severity grids + `corruption_seed` |

Result CSVs stay under `results/_archive/robustness/`.

## Restoring

Move the files back to their original locations —
`run_robustness.py` → repo root, `scripts/run_robustness_worker.py` →
`scripts/benchmarks/run_robustness.py`, `tools/*` → `tools/paper/`
(`revin_ab_summary.py` → `tools/audit/`), `utils/*` → `utils/` — and re-add
`"noise", "severity"` to `_SUBMODULES` in `utils/__init__.py`, which is the
only edit made outside this directory.

The worker needs `--save_model` checkpoints from the main sweep; those live on
the server, not in this repo.
