"""Organize the baseline results CSV into a tidy, sorted, seed-aggregated table.

Sort order (as requested): model -> seq_len (lookback) -> data -> pred_len.
Aggregates over seeds (mean/std of MSE/MAE); reports seed count so partial cells
are visible. Re-run whenever more rows land on the server.

Usage:
    python tools/paper/organize_baseline_results.py            # default live CSV -> csv+preview
    python tools/paper/organize_baseline_results.py --csv <path> --out <path>
"""
import argparse
import csv
import statistics as st
from collections import defaultdict

DEFAULT_CSV = r"C:/Users/lee/Dropbox/WDSLab/05. 실험/FEATHer/results/fcst_results.csv"
MFG = ["Steel", "WindSCADA", "BlastFurnace", "N2O", "MetroPT3", "EWELD"]
LTSF = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Exchange",
        "Electricity", "Traffic"]
DATA_ORDER = {d: i for i, d in enumerate(MFG + LTSF)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=DEFAULT_CSV)
    p.add_argument("--out", default="results/baseline_organized.csv")
    p.add_argument("--preview", type=int, default=25)
    a = p.parse_args()

    rows = list(csv.DictReader(open(a.csv, newline='', encoding='utf-8')))
    # aggregate over seeds, keyed by (model, seq_len, data, pred_len)
    agg = defaultdict(lambda: {"mse": [], "mae": [], "params": set(), "tags": set()})
    for r in rows:
        k = (r['model'], int(r['seq_len']), r['data'], int(r['pred_len']))
        try:
            agg[k]["mse"].append(float(r['MSE']))
            agg[k]["mae"].append(float(r['MAE']))
        except (ValueError, KeyError):
            continue
        agg[k]["params"].add(int(r['num_params']))
        agg[k]["tags"].add(r['exp_tag'])

    def sort_key(k):
        model, sl, data, h = k
        return (model, sl, DATA_ORDER.get(data, 99), h)

    out_rows = []
    for k in sorted(agg, key=sort_key):
        model, sl, data, h = k
        v = agg[k]
        n = len(v["mse"])
        mse_m = sum(v["mse"]) / n
        mae_m = sum(v["mae"]) / n
        mse_s = st.pstdev(v["mse"]) if n > 1 else 0.0
        mae_s = st.pstdev(v["mae"]) if n > 1 else 0.0
        out_rows.append({
            "model": model, "seq_len": sl, "data": data, "pred_len": h,
            "n_seeds": n, "exp_tag": "|".join(sorted(v["tags"])),
            "MSE_mean": round(mse_m, 4), "MSE_std": round(mse_s, 4),
            "MAE_mean": round(mae_m, 4), "MAE_std": round(mae_s, 4),
            "num_params": "|".join(str(x) for x in sorted(v["params"])),
        })

    fields = ["model", "seq_len", "data", "pred_len", "n_seeds", "exp_tag",
              "MSE_mean", "MSE_std", "MAE_mean", "MAE_std", "num_params"]
    with open(a.out, "w", newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(out_rows)

    print(f"organized {len(rows)} raw rows -> {len(out_rows)} aggregated cells")
    print(f"models: {sorted(set(r['model'] for r in out_rows))}")
    print(f"seq_lens: {sorted(set(r['seq_len'] for r in out_rows))}")
    print(f"wrote -> {a.out}\n")
    print(f"--- preview (first {a.preview}) ---")
    print(f"{'model':<13}{'seq':>5}{'data':<13}{'H':>5}{'seeds':>6}{'MSE':>9}{'MAE':>9}")
    for r in out_rows[:a.preview]:
        print(f"{r['model']:<13}{r['seq_len']:>5}{r['data']:<13}{r['pred_len']:>5}"
              f"{r['n_seeds']:>6}{r['MSE_mean']:>9.3f}{r['MAE_mean']:>9.3f}")


if __name__ == "__main__":
    main()
