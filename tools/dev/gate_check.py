# -*- coding: utf-8 -*-
"""Does the FFT gate WORK as designed?  (2026-09-16, step 1 of the gate check)

Step 1a  --synthetic   no data, no checkpoint.  Checks the gate computation:
    A. energy ratios land in the right rfft slice for sines at known bins
    B. band order: high-frequency slice -> POINT (band 0), low -> LOW (band 2)
    C. banded_sensor has no cross-sensor coupling (perturb sensor j, gate of
       every other sensor bit-identical)
    D. identical sensors -> banded_sensor == banded exactly
    E. a zero-initialised proj still receives gradient

Step 1b  --ckpt        trained checkpoints (run_dev --save_model).  See main_ckpt.

    python tools/dev/gate_check.py --synthetic
    python tools/dev/gate_check.py --ckpt --log_prefix mix0914 --arm mx_in1_r4_gsensor
"""
import argparse
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
os.chdir(REPO)

from baselines.FEATHer.FEATHer import FFTFrequencyGate  # noqa: E402

L, S = 96, 3


def expected_ratios(x, bounds):
    """Independent numpy reference: (B, S, D) power share per slice, low -> high."""
    P = np.abs(np.fft.rfft(x.astype(np.float64), axis=1)) ** 2
    tot = P.sum(axis=1) + 1e-6
    return np.stack([P[:, lo:hi].sum(axis=1) for lo, hi in bounds], axis=1) / tot[:, None, :]


def gate_features(gate, x):
    """The (B, S, D) centred log-ratio features the gate feeds to proj (band order)."""
    with torch.no_grad():
        P = torch.abs(torch.fft.rfft(x.float(), dim=1)) ** 2
        tot = P.sum(dim=1, keepdim=True) + 1e-6
        r = torch.stack([P[:, lo:hi].sum(dim=1) for lo, hi in gate._bounds], dim=1) / tot
        f = torch.log(r.flip(dims=[1]) + 1e-6)
        return r, f - f.mean(dim=1, keepdim=True)


def synthetic():
    ok = True
    torch.manual_seed(0)
    t = np.arange(L)
    gate = FFTFrequencyGate(seq_len=L, num_bands=S, gate_mode="banded_sensor")
    print("slices (rfft bins, low -> high):", gate._bounds)

    # A + B: one sensor per test bin
    bins = [3, 10, 20, 28, 36, 45]
    x = np.stack([np.sin(2 * np.pi * b * t / L) for b in bins], axis=1)[None]   # (1, L, D)
    xt = torch.tensor(x, dtype=torch.float32)
    ref = expected_ratios(x, gate._bounds)
    r, _ = gate_features(gate, xt)
    errA = float(np.abs(r.numpy() - ref).max())
    # proj = large identity -> the gate picks the band whose slice holds the energy
    with torch.no_grad():
        gate.proj.weight.copy_(torch.eye(S) * 50.0)
    g = gate(xt)[0].detach()                          # (D, S)
    print("\nA/B  sine bin -> energy slice | gate argmax band (0=POINT 1=MID 2=LOW)")
    for d, b in enumerate(bins):
        sl = int(np.argmax(ref[0, :, d]))
        band = int(torch.argmax(g[d]))
        want = S - 1 - sl
        flag = "OK" if band == want else "FAIL"
        ok &= band == want
        print(f"     bin {b:2d} -> slice {sl} (share {ref[0, sl, d]:.3f}) | band {band} "
              f"(expect {want}) {flag}   gate {np.round(g[d].numpy(), 3)}")
    print(f"A    max |ratio - numpy reference| = {errA:.2e}", "OK" if errA < 1e-5 else "FAIL")
    ok &= errA < 1e-5

    # C: cross-sensor coupling, on a random proj and a random multi-sensor input
    with torch.no_grad():
        gate.proj.weight.normal_()
        gate.proj.bias.normal_()
    xr = torch.randn(8, L, 6)
    g0 = gate(xr).detach()
    worst = 0.0
    for j in range(6):
        xp = xr.clone()
        xp[:, :, j] = torch.randn(8, L) * 3.0
        gp = gate(xp).detach()
        others = [d for d in range(6) if d != j]
        worst = max(worst, float((gp[:, others] - g0[:, others]).abs().max()))
        ok &= bool((gp[:, j] - g0[:, j]).abs().max() > 1e-3)     # the sensor itself moves
    print(f"\nC    perturb one sensor -> max change of OTHER sensors' gates = {worst:.1e}",
          "OK" if worst == 0.0 else "FAIL")
    ok &= worst == 0.0

    # D: identical sensors -> banded_sensor equals banded
    gb = FFTFrequencyGate(seq_len=L, num_bands=S, gate_mode="banded")
    gb.load_state_dict(gate.state_dict())
    xs = torch.randn(8, L, 1).repeat(1, 1, 6)
    diffD = float((gate(xs) - gb(xs)[:, None, :]).abs().max())
    print(f"D    identical sensors: |banded_sensor - banded| = {diffD:.1e}",
          "OK" if diffD < 1e-6 else "FAIL")
    ok &= diffD < 1e-6

    # E: zero-init proj receives gradient
    gz = FFTFrequencyGate(seq_len=L, num_bands=S, gate_mode="banded_sensor")
    w = torch.randn(1, 1, S)
    loss = (gz(xr) * w).sum()
    loss.backward()
    gn = float(gz.proj.weight.grad.abs().sum())
    print(f"E    zero-init start gate = {np.round(gz(xr)[0, 0].detach().numpy(), 4)}, "
          f"|dL/dW| = {gn:.3e}", "OK" if gn > 0 else "FAIL")
    ok &= gn > 0

    print("\nALL PASS" if ok else "\nSOME CHECK FAILED")
    return ok


SLICE = ["low", "mid", "high"]          # rfft slices, low -> high
BRANCH = ["POINT", "MID", "LOW"]         # gate/band order (POINT reads the high slice)


def branch_ops(model):
    """The three band branches as linear operators on (N, D, L)."""
    import torch.nn.functional as F
    return [
        model.conv_point,
        model.conv_mid,
        lambda h: F.interpolate(model.pool_low(h), size=L, mode="linear", align_corners=False),
    ]


def branch_response(model, D, device):
    """Complex gain H[s, f, d] of each branch at rfft bins f = 1..L/2-1.

    Every branch is a real LINEAR operator (bias removed), so its response to
    exp(i w t) is T(cos) + i T(sin); projecting that back onto exp(i w t) gives
    the gain at w.  Energy the pooled LOW branch aliases to other bins is
    discarded, which is what "gain at that frequency" should mean."""
    t = torch.arange(L, dtype=torch.float32, device=device)
    fbins = torch.arange(1, L // 2, device=device)
    H = torch.zeros(S, len(fbins), D, dtype=torch.complex64)
    with torch.no_grad():
        for s, op in enumerate(branch_ops(model)):
            y0 = op(torch.zeros(1, D, L, device=device))
            for i, f in enumerate(fbins.tolist()):
                w = 2 * np.pi * f / L
                c = torch.cos(w * t).view(1, 1, L).expand(1, D, L).contiguous()
                sn = torch.sin(w * t).view(1, 1, L).expand(1, D, L).contiguous()
                z = (op(c) - y0).double() + 1j * (op(sn) - y0).double()     # (1, D, L)
                ph = torch.exp(-1j * w * t.double())
                H[s, i] = (z[0] * ph).mean(dim=1).cpu()
    return fbins.cpu().numpy(), H.numpy()                                   # (S, F, D)


def argv_for(worker, log_prefix, arm, data):
    """Worker argv per learning rate for (arm, data).

    A dev log is overwritten when run_dev re-runs the same arm on other datasets,
    so the command for `data` may be gone.  Then take the arm's command for any
    dataset and swap in `data`: --data, --seeds (from the results CSV) and the
    dataset-specific keys of --model_overrides, rebuilt with the orchestrator's
    merge (method defaults <- dataset overrides, minus the keys the template
    dropped because they were given on the CLI).  main_ckpt then requires every
    checkpoint to reproduce its CSV test MSE, which catches a wrong rebuild."""
    import glob
    import re
    import pandas as pd
    from baselines import get_method_defaults, get_dataset_overrides
    from gate_swap import worker_argv
    try:
        return worker_argv(log_prefix, arm, data), False
    except SystemExit:
        pass
    exact = re.compile(re.escape(f"{log_prefix}_{arm}_") + r"[0-9.e+-]+_h\d+\.log$")
    out = []
    for log in sorted(glob.glob(os.path.join(REPO, "results", "_devlogs", f"{log_prefix}_{arm}_*_h*.log"))):
        if not exact.match(os.path.basename(log)):
            continue
        with open(log, encoding="utf-8", errors="replace") as f:
            cmd = next((l for l in f if "CMD:" in l and "scripts/benchmarks/run_forecast.py" in l), None)
        if cmd is None:
            continue
        toks = cmd.split("CMD:", 1)[1].split()
        argv = toks[toks.index("scripts/benchmarks/run_forecast.py") + 1:]
        src = argv[argv.index("--data") + 1]
        i = argv.index("--model_overrides") + 1
        tmpl = dict(kv.split("=", 1) for kv in argv[i].split(";"))
        m_src = {**get_method_defaults("FEATHer"), **get_dataset_overrides("FEATHer", src)}
        dropped = set(m_src) - set(tmpl)
        m_dst = {**get_method_defaults("FEATHer"), **get_dataset_overrides("FEATHer", data)}
        argv[i] = ";".join(f"{k}={v}" for k, v in sorted(m_dst.items()) if k not in dropped)
        argv[argv.index("--data") + 1] = data
        tag = argv[argv.index("--exp_tag") + 1]
        csv = argv[argv.index("--results_csv") + 1]
        ref = pd.read_csv(csv, on_bad_lines="skip")
        seeds = sorted(ref[(ref.exp_tag == tag) & (ref.data == data)].seed.astype(int).unique())
        if not seeds:
            continue
        argv[argv.index("--seeds") + 1] = ",".join(map(str, seeds))
        out.append(argv)
    if not out:
        raise SystemExit(f"no usable command for {arm} / {data}")
    return out, True


def main_ckpt(a):
    import pandas as pd
    from torch.utils.data import DataLoader, Subset
    sys.path.insert(0, os.path.join(REPO, "tools", "dev"))
    from gate_swap import load_worker

    worker = load_worker()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for data in [s.strip() for s in a.data.split(",")]:
        best = None
        argvs, rebuilt = argv_for(worker, a.log_prefix, a.arm, data)
        if rebuilt:
            print(f"[{data}] command rebuilt from another dataset's log (verified by test MSE below)")
        for argv in argvs:
            sys.argv = ["run_forecast.py"] + argv
            cand = worker.parse_args()
            ref = pd.read_csv(cand.results_csv, on_bad_lines="skip")
            r = ref[(ref.exp_tag == cand.exp_tag) & (ref.data == data)]
            if r.empty:
                continue
            v = r.val_loss.median()
            if best is None or v < best[0]:
                best = (v, argv)
        if best is None:
            print(f"[skip] {data}: no finished rows")
            continue
        sys.argv = ["run_forecast.py"] + best[1]
        args = worker.parse_args()
        worker._apply_overrides(args)
        args.label_len = min(args.label_len, args.seq_len)
        H_len = int(args.pred_len)
        ref_all = pd.read_csv(args.results_csv, on_bad_lines="skip")
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        df, _, _ = worker.data_factory.data_select(data, args.root_path)
        nf = df.shape[1] - 1
        sets = {}
        for split in ("train", "val", "test"):
            sets[split], _ = worker.data_factory.data_provider(
                args.root_path, data, args.features, args.batch_size,
                args.seq_len, args.label_len, H_len, split,
                starting_percent=args.starting_percent, percent=args.percent)
        if hasattr(worker, "resolve_gate_bounds"):
            worker.resolve_gate_bounds(args, sets["train"], nf)
        val = sets["val"]
        if len(val) > a.max_windows:
            val = Subset(val, np.linspace(0, len(val) - 1, a.max_windows).astype(int).tolist())
        print(f"\n{data}: {args.exp_tag} lr {args.lr} gate {args.gate_mode} D={nf} "
              f"val windows {len(val)}")

        for seed in seeds:
            ckpt = worker.checkpoint_path(args.exp_tag, args.model, data, H_len,
                                          args.seq_len, seed)
            if not os.path.exists(ckpt):
                print(f"[skip] s{seed}: no checkpoint")
                continue
            torch.manual_seed(seed)
            model = worker.get_model(args.model, args, n_features=nf,
                                     seq_len=args.seq_len, pred_len=H_len).float().to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            model.eval()
            gate = model.freq_gate

            # --- (0) the rebuilt model must reproduce the reported test MSE --
            se, n = 0.0, 0
            with torch.no_grad():
                for bx, by, _, _ in DataLoader(sets["test"], batch_size=512, shuffle=False,
                                               num_workers=0):
                    yp = model(bx.float().to(device)[:, :, :nf])[:, -H_len:].cpu().numpy()
                    d_ = yp - by.float()[:, :, :nf][:, -H_len:].numpy()
                    fin = np.isfinite(d_)
                    se += float((d_[fin] ** 2).sum()); n += int(fin.sum())
            mse = se / n
            rep = ref_all[(ref_all.exp_tag == args.exp_tag) & (ref_all.data == data)
                          & (ref_all.seed == seed)].MSE
            rep = float(rep.iloc[0]) if len(rep) else float("nan")
            match = abs(mse - rep) <= 1e-5 * max(1.0, abs(rep))
            print(f"  s{seed}: test MSE {mse:.6f} vs CSV {rep:.6f} {'OK' if match else 'MISMATCH'}")
            if not match:
                raise SystemExit(f"{data} s{seed}: checkpoint does not reproduce the CSV - config wrong")

            # --- (1) branch frequency responses --------------------------
            fb, Hc = branch_response(model, nf, device)
            sl_idx = [np.where((fb >= lo) & (fb < hi))[0] for lo, hi in gate._bounds]
            gain = np.stack([np.abs(Hc[:, ix]).mean(axis=1) for ix in sl_idx], axis=1)  # (S_branch, S_slice, D)

            # --- (2) gate vs spectrum on VAL windows ---------------------
            rec = {"r": [], "g": []}

            def hook(mod, inp, out):
                r, _ = gate_features(mod, inp[0]) if mod._bounds_ps is None else (None, None)
                rec["r"].append(r.cpu())
                rec["g"].append(out.detach().cpu())
            hd = gate.register_forward_hook(hook)
            loader = DataLoader(val, batch_size=512, shuffle=False, num_workers=0)
            with torch.no_grad():
                for bx, _, _, _ in loader:
                    model(bx.float().to(device)[:, :, :nf])
            hd.remove()
            R = torch.cat(rec["r"]).numpy()          # (N, S, D) slice shares low -> high
            G = torch.cat(rec["g"]).numpy()          # (N, D, S) band weights POINT, MID, LOW

            def corr(u, v):
                u = u - u.mean(); v = v - v.mean()
                den = np.sqrt((u * u).sum() * (v * v).sum())
                return float((u * v).sum() / den) if den > 0 else np.nan

            # per sensor: corr(share of slice k, weight of the band that reads it)
            cmat = np.full((S, S), np.nan)
            for k in range(S):          # slice
                for b in range(S):      # band
                    cmat[k, b] = np.nanmean([corr(np.log(R[:, k, d] + 1e-6), G[:, d, b])
                                             for d in range(nf)])

            # --- (3) does the gate reshape the effective filter? ----------
            # effective response per window/sensor: |sum_s g_s H_s(f)|, averaged per slice
            Gm = G.mean(0)                                       # (D, S) mean gate
            def eff(gw):                                         # gw (..., D, S)
                Z = np.einsum("...ds,sfd->...fd", gw, Hc)
                return np.stack([np.abs(Z[..., ix, :]).mean(axis=-2) for ix in sl_idx], axis=-1)
            E_own = eff(G)                                       # (N, D, S_slice)
            E_bar = eff(Gm)                                      # (D, S_slice)
            rel = np.abs(E_own - E_bar[None]) / (E_bar[None] + 1e-12)
            p10, p90 = np.percentile(E_own, 10, axis=0), np.percentile(E_own, 90, axis=0)
            W = gate.proj.weight.detach().cpu().numpy()
            row = dict(data=data, seed=seed, D=nf,
                       proj_W_fro=float(np.linalg.norm(W)),
                       proj_b_norm=float(np.linalg.norm(gate.proj.bias.detach().cpu().numpy())))
            for b in range(S):
                for k in range(S):
                    row[f"gain_{BRANCH[b]}_{SLICE[k]}"] = float(gain[b, k].mean())
                row[f"g_mean_{BRANCH[b]}"] = float(G[:, :, b].mean())
                row[f"g_winstd_{BRANCH[b]}"] = float(G[:, :, b].std(0).mean())
                row[f"g_sensorstd_{BRANCH[b]}"] = float(Gm[:, b].std())
            for k in range(S):
                row[f"share_{SLICE[k]}"] = float(R[:, k].mean())
                row[f"eff_{SLICE[k]}"] = float(E_bar[:, k].mean())
                row[f"eff_relchange_{SLICE[k]}"] = float(rel[..., k].mean())
                row[f"eff_p90p10_{SLICE[k]}"] = float(((p90 - p10)[:, k] / (E_bar[:, k] + 1e-12)).mean())
                for b in range(S):
                    row[f"corr_{SLICE[k]}_{BRANCH[b]}"] = cmat[k, b]
            rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    med = out.groupby("data").median(numeric_only=True)
    order = [d for d in a.data.split(",") if d in med.index]
    med = med.reindex(order)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)

    print("\n=== (1) branch gain |H| averaged over each rfft slice (median over seeds, mean over sensors) ===")
    print("    a real band bank: POINT largest on high, LOW largest on low")
    for d, r in med.iterrows():
        print(f"{d:13s} " + " | ".join(
            f"{BRANCH[b]} " + " ".join(f"{SLICE[k][0]}{r[f'gain_{BRANCH[b]}_{SLICE[k]}']:.2f}"
                                       for k in range(S)) for b in range(S)))
    print("\n=== energy share of the rfft slices in VAL windows (low / mid / high) ===")
    for d, r in med.iterrows():
        print(f"{d:13s} " + " / ".join(f"{r[f'share_{SLICE[k]}']:.3f}" for k in range(S)))
    print("\n=== (2) gate: mean weight, spread across windows, spread across sensors; proj norm ===")
    for d, r in med.iterrows():
        print(f"{d:13s} " + "  ".join(
            f"{BRANCH[b]} {r[f'g_mean_{BRANCH[b]}']:.3f} (win {r[f'g_winstd_{BRANCH[b]}']:.3f}, "
            f"sens {r[f'g_sensorstd_{BRANCH[b]}']:.3f})" for b in range(S))
              + f"  |W| {r['proj_W_fro']:.2f}")
    print("\n=== (2) corr( log share of slice, gate weight ), mean over sensors; diagonal = the matching band ===")
    for d, r in med.iterrows():
        print(f"{d:13s} " + "  ".join(
            f"{SLICE[k]}->" + ",".join(f"{BRANCH[b]}{r[f'corr_{SLICE[k]}_{BRANCH[b]}']:+.2f}" for b in range(S))
            for k in range(S)))
    print("\n=== (3) effective filter |sum g_s H_s|: mean gain per slice, change vs the mean gate ===")
    print("    relchange = mean |E_window - E_meangate| / E_meangate ;  p90-p10 = window spread / E_meangate")
    for d, r in med.iterrows():
        print(f"{d:13s} " + "  ".join(
            f"{SLICE[k]} {r[f'eff_{SLICE[k]}']:.2f} (chg {100*r[f'eff_relchange_{SLICE[k]}']:.1f}%, "
            f"p90-p10 {100*r[f'eff_p90p10_{SLICE[k]}']:.1f}%)" for k in range(S)))
    print(f"\nwrote {a.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--ckpt", action="store_true")
    ap.add_argument("--log_prefix", default="mix0914")
    ap.add_argument("--arm", default="mx_in1_r4_gsensor")
    ap.add_argument("--data", default="Steel,EWELD,WindSCADA,BlastFurnace,N2O,MetroPT3")
    ap.add_argument("--max_windows", type=int, default=3000)
    ap.add_argument("--out", default="results/_gate_check_gsensor.csv")
    a = ap.parse_args()
    if a.synthetic:
        sys.exit(0 if synthetic() else 1)
    if a.ckpt:
        main_ckpt(a)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
