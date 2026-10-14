"""Memory-matched comparison: per dataset (D channels), each FEATHer variant uses the longest window L it can
afford on the 20 KB part (channels-per-chip(L) >= D); compare the two at those windows.  No new training."""
import pandas as pd, numpy as np, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
R = r"C:\Users\lee\Dropbox\github\research\timeseries\FEATHer\results"
sv = pd.read_csv(f"{R}/analysis/sram_vs_lookback_20261006.csv")
print("== sram_vs_lookback columns:", list(sv.columns)); print(sv.to_string()[:1500])
raw = pd.read_csv(f"{R}/raw_mfg.csv")
f = raw[(raw.config_id == 'frozen') & (raw.role.isin(['main', 'lookback'])) & (raw.model.isin(['FEATHer_mix', 'FEATHer_CI']))]
cell = f.groupby(['model', 'seq_len', 'data', 'pred_len']).MSE.mean()
ds = cell.groupby(['model', 'seq_len', 'data']).mean().unstack('data')   # H-averaged per dataset
print("\n== H-averaged MSE per dataset\n", ds.round(4).to_string())
D = {'Steel': 6, 'MetroPT3': 7, 'WindSCADA': 14, 'N2O': 14, 'EWELD': 14, 'BlastFurnace': 28}
# channels per chip bounds (CLAUDE.md / T12 own-buffer): CI 48/24/13, mix 43/18/7  (L96/192/336)
CAP = {'FEATHer_CI': {96: 48, 192: 24, 336: 13}, 'FEATHer_mix': {96: 43, 192: 18, 336: 7}}
rows = []
for d, n in D.items():
    pick = {}
    for m in CAP:
        Ls = [L for L in (96, 192, 336) if CAP[m][L] >= n]
        pick[m] = max(Ls) if Ls else None
    mse = {m: (cell[(m, pick[m], d)].mean() if pick[m] else np.nan) for m in CAP}
    pc = {m: cell[(m, pick[m], d)] for m in CAP}  # per-H series
    per_h = ((pc['FEATHer_CI'].values / pc['FEATHer_mix'].values - 1) * 100).round(1).tolist()
    rows.append(dict(data=d, D=n, L_CI=pick['FEATHer_CI'], L_mix=pick['FEATHer_mix'], MSE_CI=round(mse['FEATHer_CI'], 4),
                     MSE_mix=round(mse['FEATHer_mix'], 4), CI_vs_mix_pct=round((pc['FEATHer_CI'].values / pc['FEATHer_mix'].values - 1).mean() * 100, 2),
                     per_H=per_h))
out = pd.DataFrame(rows)
print("\n== memory-matched per dataset (CI_vs_mix_pct > 0 => mix better; mean of per-H %)\n", out.to_string(index=False))
# hypothetical channel-count bands where CI affords one more doubling than mix (no dataset of that size -> 6-dataset means)
print("\n== channel-count bands (6-dataset H-averaged means)")
allmean = cell.groupby(['model', 'seq_len']).mean()
for lo, hi, Lci, Lmx in [(1, 7, 336, 336), (8, 13, 336, 192), (14, 18, 192, 192), (19, 24, 192, 96), (25, 43, 96, 96), (44, 48, 96, None)]:
    a = allmean[('FEATHer_CI', Lci)]
    b = allmean[('FEATHer_mix', Lmx)] if Lmx else np.nan
    print(f"D {lo:2d}-{hi:2d}: CI L{Lci} {a:.4f} vs mix L{Lmx} {b:.4f} -> CI/mix-1 = {(a/b-1)*100:+.1f} %" if Lmx else f"D {lo}-{hi}: CI L{Lci} {a:.4f}; mix does not fit")
out.to_csv(f"{R}/analysis/memory_matched_20261007.csv", index=False)
print("\nsaved results/analysis/memory_matched_20261007.csv")
