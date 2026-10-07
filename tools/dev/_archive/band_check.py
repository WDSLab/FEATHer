# -*- coding: utf-8 -*-
"""Do the band branches LEARN the frequency ranges their kernel sizes claim?  (2026-09-18)

Claim: POINT (conv k=3) takes the fast changes, MID (conv k=5) the mid ones, LOW
(AvgPool4 + interpolation) the slow ones.  Kernel size does NOT enforce this — a
k=3 kernel [1/3,1/3,1/3] is a low-pass — so it has to be measured.

For every trained checkpoint (and the same model at its random initialisation):
  * the gain |H(f)| of each branch at every rfft bin f = 0..L/2 (all 49 bins, not
    the gate's three slices), per sensor — each branch is a linear operator;
  * the spectral centroid of |H|^2 (which frequency a branch passes on average);
  * per (seed, sensor): is the order centroid POINT > MID > LOW?

Pre-agreed verdict (09-18): "bands are learned" iff on >= 5 of 6 datasets the order
holds for >= 90 % of (seed, sensor) pairs AND the trained rate exceeds the init rate
by >= 20 points.  Otherwise -> constrain the kernels (zero-sum POINT/MID, method A).

    python tools/dev/band_check.py
    python tools/dev/band_check.py --data Steel --max_seeds 5
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools", "dev"))
os.chdir(REPO)

from gate_check import argv_for  # noqa: E402

BRANCH = ["POINT", "MID", "LOW"]


def branch_ops(model, L):
    return [
        model.conv_point,
        model.conv_mid,
        lambda h: F.interpolate(model.pool_low(h), size=L, mode="linear", align_corners=False),
    ]


def response(model, D, L, device):
    """|H[s, f, d]| at rfft bins f = 0..L/2 (bias removed; energy aliased by the
    pooled LOW branch to other bins is discarded)."""
    t = torch.arange(L, dtype=torch.float64, device=device)
    fb = np.arange(0, L // 2 + 1)
    H = np.zeros((3, len(fb), D))
    with torch.no_grad():
        for s, op in enumerate(branch_ops(model, L)):
            y0 = op(torch.zeros(1, D, L, device=device)).double()
            for i, f in enumerate(fb):
                w = 2 * np.pi * f / L
                c = torch.cos(w * t).float().view(1, 1, L).expand(1, D, L).contiguous()
                sn = torch.sin(w * t).float().view(1, 1, L).expand(1, D, L).contiguous()
                z = (op(c).double() - y0) + 1j * (op(sn).double() - y0)
                # at DC and Nyquist sin == 0 and cos^2 averages to 1, so the same
                # projection returns the gain there too, no extra factor
                ph = torch.exp(-1j * w * t)
                H[s, i] = np.abs((z[0] * ph).mean(dim=1).cpu().numpy())
    return fb, H


def centroid(fb, H):
    """(3, D) power-weighted mean frequency bin of each branch per sensor."""
    P = H ** 2
    return (fb[None, :, None] * P).sum(1) / (P.sum(1) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_prefix", default="mix0914")
    ap.add_argument("--arm", default="mx_in1_r4_gsensor")
    ap.add_argument("--data", default="Steel,EWELD,WindSCADA,BlastFurnace,N2O,MetroPT3")
    ap.add_argument("--max_seeds", type=int, default=0, help="0 = all")
    ap.add_argument("--out", default="results/_band_check.csv")
    ap.add_argument("--fig", default="results/_band_check.png")
    a = ap.parse_args()

    import pandas as pd
    from gate_swap import load_worker
    worker = load_worker()
    device = torch.device("cpu")
    rows, curves = [], {}
    for data in [s.strip() for s in a.data.split(",")]:
        argvs, _ = argv_for(worker, a.log_prefix, a.arm, data)
        best = None
        for argv in argvs:                           # val-selected lr, as in gate_check
            sys.argv = ["run_forecast.py"] + argv
            cand = worker.parse_args()
            ref = pd.read_csv(cand.results_csv, on_bad_lines="skip")
            r = ref[(ref.exp_tag == cand.exp_tag) & (ref.data == data)]
            if not r.empty and (best is None or r.val_loss.median() < best[0]):
                best = (r.val_loss.median(), argv)
        if best is None:
            print(f"[skip] {data}: no finished rows")
            continue
        sys.argv = ["run_forecast.py"] + best[1]
        args = worker.parse_args()
        worker._apply_overrides(args)
        args.label_len = min(args.label_len, args.seq_len)
        L, H_len = int(args.seq_len), int(args.pred_len)
        df, _, _ = worker.data_factory.data_select(data, args.root_path)
        nf = df.shape[1] - 1
        # seeds from the results CSV, not the log command: a killed re-run of the
        # same arm overwrites the log with a partial seed list
        ref = pd.read_csv(args.results_csv, on_bad_lines="skip")
        seeds = sorted(ref[(ref.exp_tag == args.exp_tag) & (ref.data == data)
                           & (ref.pred_len == H_len)].seed.astype(int).unique())
        if a.max_seeds:
            seeds = seeds[:a.max_seeds]
        print(f"{data}: {args.exp_tag} lr {args.lr} D={nf} seeds {len(seeds)}")
        cv = {"trained": [], "init": []}
        for seed in seeds:
            ckpt = worker.checkpoint_path(args.exp_tag, args.model, data, H_len, args.seq_len, seed)
            if not os.path.exists(ckpt):
                print(f"  [skip] s{seed}: no checkpoint")
                continue
            for state in ("init", "trained"):
                torch.manual_seed(seed)
                model = worker.get_model(args.model, args, n_features=nf,
                                         seq_len=L, pred_len=H_len).float().to(device)
                if state == "trained":
                    model.load_state_dict(torch.load(ckpt, map_location=device))
                model.eval()
                fb, H = response(model, nf, L, device)
                c = centroid(fb, H)                                   # (3, D)
                cv[state].append(H / (H.max(axis=1, keepdims=True) + 1e-12))
                for d in range(nf):
                    rows.append(dict(data=data, seed=seed, sensor=d, state=state,
                                     c_POINT=c[0, d], c_MID=c[1, d], c_LOW=c[2, d],
                                     peak_POINT=int(fb[H[0, :, d].argmax()]),
                                     peak_MID=int(fb[H[1, :, d].argmax()]),
                                     peak_LOW=int(fb[H[2, :, d].argmax()]),
                                     dc_POINT=H[0, 0, d], dc_MID=H[1, 0, d], dc_LOW=H[2, 0, d],
                                     ordered=bool(c[0, d] > c[1, d] > c[2, d]),
                                     P_gt_M=bool(c[0, d] > c[1, d]),
                                     M_gt_L=bool(c[1, d] > c[2, d])))
        curves[data] = {k: np.concatenate(v, axis=2) for k, v in cv.items() if v}   # (3, F, seeds*D)

    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    pd.set_option("display.width", 250)

    print("\n=== spectral centroid (rfft bin; period = 96/bin), median over seed x sensor ===")
    print(f"{'':13s} {'state':8s} {'POINT':>7s} {'MID':>7s} {'LOW':>7s} | "
          f"{'P>M>L':>6s} {'P>M':>6s} {'M>L':>6s} | n")
    verdict = {}
    for data in out.data.unique():
        rate = {}
        for state in ("init", "trained"):
            g = out[(out.data == data) & (out.state == state)]
            rate[state] = g.ordered.mean()
            print(f"{data:13s} {state:8s} {g.c_POINT.median():7.2f} {g.c_MID.median():7.2f} "
                  f"{g.c_LOW.median():7.2f} | {100*g.ordered.mean():5.0f}% {100*g.P_gt_M.mean():5.0f}% "
                  f"{100*g.M_gt_L.mean():5.0f}% | {len(g)}")
        verdict[data] = rate["trained"] >= 0.90 and rate["trained"] - rate["init"] >= 0.20
    print("\n=== DC gain |H(0)| (how much of a constant each branch lets through), trained, median ===")
    for data in out.data.unique():
        g = out[(out.data == data) & (out.state == "trained")]
        print(f"{data:13s} POINT {g.dc_POINT.median():.2f}  MID {g.dc_MID.median():.2f}  LOW {g.dc_LOW.median():.2f}")
    npass = sum(verdict.values())
    print(f"\nper-dataset pass (>=90% ordered AND >=20 pts above init): "
          + ", ".join(f"{d} {'PASS' if v else 'fail'}" for d, v in verdict.items()))
    print(f"VERDICT: {npass}/{len(verdict)} datasets pass -> "
          + ("bands are LEARNED (>=5/6)" if npass >= 5 else "NOT established -> method A"))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(curves)
        fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4.2 * ((n + 1) // 2), 7), squeeze=False)
        col = {"POINT": "#d62728", "MID": "#2ca02c", "LOW": "#1f77b4"}
        for ax, (data, cv) in zip(axes.flat, curves.items()):
            for s, b in enumerate(BRANCH):
                for state, ls in (("init", ":"), ("trained", "-")):
                    if state in cv:
                        ax.plot(fb, np.median(cv[state][s], axis=1), ls, color=col[b],
                                label=f"{b} {state}", lw=1.8 if state == "trained" else 1.2)
            ax.set_title(data)
            ax.set_xlabel("rfft bin (period = 96 / bin)")
            ax.set_ylabel("|H| / max")
            ax.grid(alpha=.3)
        axes.flat[0].legend(fontsize=7)
        for ax in list(axes.flat)[n:]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(a.fig, dpi=130)
        print(f"wrote {a.fig}")
    except ImportError:
        print("matplotlib missing - no figure")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
