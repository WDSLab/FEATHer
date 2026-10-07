# -*- coding: utf-8 -*-
"""How much could a PERFECT per-window gate gain?  (2026-09-21)

The FFT gate keeps tying its input-independent twin.  Two possible reasons:
  (1) the best gate value hardly changes from window to window, or
  (2) it does change, but the spectrum cannot tell which value to use.
This separates them.  For a trained checkpoint the forecast is LINEAR in the
gate value (every stage after the gate is linear), so for every test window and
sensor the gate value that minimises that window's error has a closed form:

    y = Y_fix + sum_s g_s Y_s ,   g in the gate's own range

  decision "band"  (gate_mode banded_sensor): g = the 3 band weights (simplex),
                    Y_s = head(band_s), Y_fix = head(mixer correction)
  decision "g2"    (--gate_mix fft):  g = alpha in [0, 2],
                    Y_1 = head(alpha-free correction), Y_fix = head(rest)
  decision "g4"    (--head2 fft):     g in [0, 1],  Y_1 = Y_A - Y_B, Y_fix = Y_B

Reported test MSE (dataset scale, same as the results CSV), per dataset:
  model     the checkpoint as trained (must reproduce the CSV MSE)
  const     the best single value for the whole test set          ("never adapt")
  oracle    per window x sensor best value, chosen WITH the target ("perfect adapt",
            an optimistic upper bound)
  fft_lin   per window x sensor value predicted from the gate's spectral features
            by a least-squares map fitted on VAL oracle values  (1-D decisions only)

    python tools/dev/oracle_gate.py --decision band --arm mx_in1_r4_gsensor --log_prefix mix0914
    python tools/dev/oracle_gate.py --decision g4   --arm g4_fft            --log_prefix mix0914
"""
import argparse
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools", "dev"))
os.chdir(REPO)

from gate_check import argv_for  # noqa: E402


# ----------------------------------------------------------------------------
# quadratic minimisation:  min_g  g'Gg - 2c'g   over the gate's range
# ----------------------------------------------------------------------------

def solve_interval(G, c, lo, hi):
    """1-D: G (...,), c (...,) -> clipped minimiser."""
    g = np.where(G > 1e-12, c / np.maximum(G, 1e-12), 0.5 * (lo + hi))
    return np.clip(g, lo, hi)


def solve_simplex3(G, c):
    """3-D over the probability simplex, vectorised.  G (...,3,3), c (...,3).
    Enumerate the active sets: interior, the 3 edges, the 3 vertices; keep the
    feasible candidate with the lowest objective."""
    shp = c.shape[:-1]
    G = G.reshape(-1, 3, 3)
    c = c.reshape(-1, 3)
    cands = []
    # interior: minimise with sum g = 1 via g3 = 1 - g1 - g2
    E = np.array([[1, 0], [0, 1], [-1, -1]], float)          # g = E z + e
    e = np.array([0, 0, 1], float)
    Gz = np.einsum("ia,nij,jb->nab", E, G, E)
    cz = np.einsum("ia,ni->na", E, c) - np.einsum("ia,nij,j->na", E, G, e)
    det = Gz[:, 0, 0] * Gz[:, 1, 1] - Gz[:, 0, 1] * Gz[:, 1, 0]
    ok = np.abs(det) > 1e-14
    inv = np.zeros_like(Gz)
    inv[ok, 0, 0] = Gz[ok, 1, 1] / det[ok]
    inv[ok, 1, 1] = Gz[ok, 0, 0] / det[ok]
    inv[ok, 0, 1] = -Gz[ok, 0, 1] / det[ok]
    inv[ok, 1, 0] = -Gz[ok, 1, 0] / det[ok]
    z = np.einsum("nab,nb->na", inv, cz)
    gi = np.einsum("ia,na->ni", E, z) + e
    gi[~ok] = np.nan
    cands.append(gi)
    # edges: one weight 0, the other two on [0, 1]
    for k in range(3):
        i, j = [t for t in range(3) if t != k]
        # g_i = t, g_j = 1 - t
        d = np.zeros(3); d[i], d[j] = 1, -1
        b = np.zeros(3); b[j] = 1
        a2 = np.einsum("i,nij,j->n", d, G, d)
        a1 = np.einsum("ni,i->n", c, d) - np.einsum("i,nij,j->n", d, G, b)
        t = solve_interval(a2, a1, 0.0, 1.0)
        cands.append(t[:, None] * d + b)
    for k in range(3):
        v = np.zeros((len(c), 3)); v[:, k] = 1
        cands.append(v)
    best, best_f = None, None
    for g in cands:
        feas = np.all(np.isfinite(g), 1) & np.all(g >= -1e-9, 1)
        f = np.einsum("ni,nij,nj->n", np.nan_to_num(g), G, np.nan_to_num(g)) - 2 * np.einsum("ni,ni->n", c, np.nan_to_num(g))
        f[~feas] = np.inf
        if best is None:
            best, best_f = np.nan_to_num(g).copy(), f
        else:
            m = f < best_f
            best[m], best_f[m] = g[m], f[m]
    return best.reshape(*shp, 3)


# ----------------------------------------------------------------------------
# decomposing a forward pass into  y = Y_fix + sum g_s Y_s  (normalised scale)
# ----------------------------------------------------------------------------

def head(model, h):
    """The model's linear SPK head (plus the fixed head-2 blend when present)."""
    return model.sparse_head(h)


@torch.no_grad()
def decompose(model, x, decision):
    """Returns Y_fix (B,H,D), Ys (B,S,H,D), g_model (B,D,S), feats (B,D,F), mu, sd."""
    cap = {}
    hs = [model.band_nl.register_forward_hook(lambda m, i, o: cap.__setitem__("bands", o)),
          model.sparse_head.register_forward_hook(lambda m, i, o: cap.__setitem__("hidden", i[0])),
          model.freq_gate.register_forward_hook(lambda m, i, o: cap.__setitem__("g", o))]
    orig_mixer = model._mixer
    model._mixer = lambda v: cap.setdefault("delta", orig_mixer(v))
    y = model(x)
    model._mixer = orig_mixer
    for h in hs:
        h.remove()
    mu = x.mean(1, keepdim=True)
    sd = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5)
    xn = (x - mu) / sd
    hidden, delta = cap["hidden"], cap["delta"]
    if decision == "band":
        bands = cap["bands"]                                   # (B,S,L,D)
        Ys = torch.stack([head(model, bands[:, s]) for s in range(bands.shape[1])], 1)
        Y_fix = head(model, delta)
        g = cap["g"]                                           # (B,D,S)
        feats = model.freq_gate._feats                         # (B,D,S)
    elif decision == "g2":
        alpha = 2 * torch.sigmoid(model.mix_gain(model.freq_gate._feats)).permute(0, 2, 1)  # (B,1,D)
        h0 = hidden - alpha * delta
        Ys = head(model, delta).unsqueeze(1)
        Y_fix = head(model, h0)
        g = alpha.permute(0, 2, 1)                             # (B,D,1)
        feats = model.freq_gate._feats
    elif decision == "g4":
        YA = model.sparse_head(hidden)
        YB = model._head2_readout(hidden)
        Ys = (YA - YB).unsqueeze(1)
        Y_fix = YB
        g = model._head2_weight(xn).permute(0, 2, 1)           # (B,D,1)
        P = torch.fft.rfft(xn.float(), dim=1).abs() ** 2
        tot = P.sum(dim=1) + 1e-6
        feats = torch.stack([torch.log(P[:, b].sum(dim=1) / tot + 1e-6) for b in model._h2_bins], -1)
    else:
        raise ValueError(decision)
    # sanity: the decomposition reproduces the model output (normalised scale)
    y_rec = Y_fix + torch.einsum("bdS,bShd->bhd", g, Ys)
    y_n = (y - mu[:, :1]) / sd[:, :1]
    err = (y_rec - y_n).abs().max().item()
    return Y_fix, Ys, g, feats, mu[:, 0], sd[:, 0], y, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decision", required=True, choices=["band", "g2", "g4"])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--log_prefix", default="mix0914")
    ap.add_argument("--data", default="Steel,EWELD,WindSCADA,BlastFurnace")
    ap.add_argument("--max_seeds", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import pandas as pd
    from torch.utils.data import DataLoader
    from gate_swap import load_worker
    worker = load_worker()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lo_hi = {"g2": (0.0, 2.0), "g4": (0.0, 1.0)}
    rows = []
    for data in [s.strip() for s in a.data.split(",")]:
        argvs, _ = argv_for(worker, a.log_prefix, a.arm, data)
        best = None
        for argv in argvs:
            sys.argv = ["run_forecast.py"] + argv
            cand = worker.parse_args()
            ref = pd.read_csv(cand.results_csv, on_bad_lines="skip")
            r = ref[(ref.exp_tag == cand.exp_tag) & (ref.data == data)]
            if not r.empty and (best is None or r.val_loss.median() < best[0]):
                best = (r.val_loss.median(), argv)
        sys.argv = ["run_forecast.py"] + best[1]
        args = worker.parse_args()
        worker._apply_overrides(args)
        args.label_len = min(args.label_len, args.seq_len)
        H = int(args.pred_len)
        ref = pd.read_csv(args.results_csv, on_bad_lines="skip")
        rsel = ref[(ref.exp_tag == args.exp_tag) & (ref.data == data) & (ref.pred_len == H)]
        seeds = sorted(rsel.seed.astype(int).unique())[:a.max_seeds]
        df, _, _ = worker.data_factory.data_select(data, args.root_path)
        nf = df.shape[1] - 1
        sets = {}
        for split in ("val", "test"):
            sets[split], _ = worker.data_factory.data_provider(
                args.root_path, data, args.features, args.batch_size, args.seq_len,
                args.label_len, H, split, starting_percent=args.starting_percent, percent=args.percent)
        print(f"\n{data}: {args.exp_tag} D={nf} seeds {seeds}")
        for seed in seeds:
            ckpt = worker.checkpoint_path(args.exp_tag, args.model, data, H, args.seq_len, seed)
            torch.manual_seed(seed)
            model = worker.get_model(args.model, args, n_features=nf, seq_len=args.seq_len,
                                     pred_len=H).float().to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            model.eval()
            acc = {}
            for split in ("val", "test"):
                G_all, c_all, k_all, w_all, F_all, go_all = [], [], [], [], [], []
                H1_all, n2_all = [], []
                sse_model, n_cells, max_err = 0.0, 0, 0.0
                for bx, by, _, _ in DataLoader(sets[split], batch_size=512, shuffle=False, num_workers=0):
                    x = bx.float().to(device)[:, :, :nf]
                    T = by.float().to(device)[:, -H:, :nf]
                    Y_fix, Ys, g, feats, mu, sd, y, err = decompose(model, x, a.decision)
                    max_err = max(max_err, err)
                    fin = torch.isfinite(T)
                    d = torch.where(fin, y - T, torch.zeros_like(T))
                    sse_model += float((d ** 2).sum()); n_cells += int(fin.sum())
                    # normalised target; per (window, sensor) quadratic in g, weighted by sd^2
                    Tn = torch.where(fin, (T - mu[:, None]) / sd[:, None], torch.zeros_like(T))
                    m = fin.float()
                    R = (Tn - Y_fix) * m                                        # (B,H,D)
                    Ysm = Ys * m[:, None]
                    w = (sd ** 2)                                               # (B,D)
                    G = torch.einsum("bshd,bthd->bdst", Ysm, Ysm)               # (B,D,S,S)
                    c = torch.einsum("bshd,bhd->bds", Ysm, R)                   # (B,D,S)
                    k = (R ** 2).sum(1)                                         # (B,D)
                    hh = H // 2
                    G1 = torch.einsum("bshd,bthd->bdst", Ysm[:, :, :hh], Ysm[:, :, :hh])
                    c1 = torch.einsum("bshd,bhd->bds", Ysm[:, :, :hh], R[:, :hh])
                    G2_ = torch.einsum("bshd,bthd->bdst", Ysm[:, :, hh:], Ysm[:, :, hh:])
                    c2 = torch.einsum("bshd,bhd->bds", Ysm[:, :, hh:], R[:, hh:])
                    k2 = (R[:, hh:] ** 2).sum(1)
                    n2_all.append(fin[:, hh:].sum().item())
                    H1_all.append((G1.cpu().double(), c1.cpu().double(),
                                   G2_.cpu().double(), c2.cpu().double(), k2.cpu().double()))
                    G_all.append(G.cpu().double()); c_all.append(c.cpu().double())
                    k_all.append(k.cpu().double()); w_all.append(w.cpu().double())
                    F_all.append(feats.cpu().double()); go_all.append(g.cpu().double())
                G = torch.cat(G_all).numpy(); c = torch.cat(c_all).numpy()
                k = torch.cat(k_all).numpy(); w = torch.cat(w_all).numpy()
                Fe = torch.cat(F_all).numpy(); go = torch.cat(go_all).numpy()

                def sse(g):                                     # g (N,D,S) -> dataset-scale SSE
                    q = np.einsum("nds,ndst,ndt->nd", g, G, g) - 2 * np.einsum("nds,nds->nd", c, g) + k
                    return float((w * q).sum())
                S = G.shape[-1]
                # const: one value for the whole split (weighted aggregate quadratic)
                Gt = np.einsum("nd,ndst->st", w, G); ct = np.einsum("nd,nds->s", w, c)
                if S == 1:
                    gc = solve_interval(Gt[0, 0], ct[0], *lo_hi[a.decision]).reshape(1)
                    go_ = solve_interval(G[..., 0, 0], c[..., 0], *lo_hi[a.decision])[..., None]
                else:
                    gc = solve_simplex3(Gt[None], ct[None])[0]
                    go_ = solve_simplex3(G, c)
                # (a) choose on the first half of the horizon, score the second half
                G1, c1, G2h, c2h, k2 = [torch.cat([t[i] for t in H1_all]).numpy() for i in range(5)]
                def sse2(g):
                    q = np.einsum("nds,ndst,ndt->nd", g, G2h, g) - 2 * np.einsum("nds,nds->nd", c2h, g) + k2
                    return float((w * q).sum())
                n2 = sum(n2_all)
                G2t = np.einsum("nd,ndst->st", w, G2h); c2t = np.einsum("nd,nds->s", w, c2h)
                if S == 1:
                    g_first = solve_interval(G1[..., 0, 0], c1[..., 0], *lo_hi[a.decision])[..., None]
                    gc2 = solve_interval(G2t[0, 0], c2t[0], *lo_hi[a.decision]).reshape(1)
                else:
                    g_first = solve_simplex3(G1, c1)
                    gc2 = solve_simplex3(G2t[None], c2t[None])[0]
                half = dict(const=sse2(np.broadcast_to(gc2, go.shape)) / n2,
                            first=sse2(g_first) / n2, model=sse2(go) / n2)
                # (b) causal: use the oracle value of the window whose target ended
                #     just before this window's input starts (known at forecast time)
                Hn = H
                sub_ = slice(Hn, None)
                g_lag = go_[:-Hn] if len(go_) > Hn else go_[:0]
                def sse_sub(g, idx):
                    q = (np.einsum("nds,ndst,ndt->nd", g, G[idx], g)
                         - 2 * np.einsum("nds,nds->nd", c[idx], g) + k[idx])
                    return float((w[idx] * q).sum())
                n_sub = n_cells * (len(go_) - Hn) / max(len(go_), 1)
                Gs = np.einsum("nd,ndst->st", w[sub_], G[sub_]); cs = np.einsum("nd,nds->s", w[sub_], c[sub_])
                gcs = (solve_interval(Gs[0, 0], cs[0], *lo_hi[a.decision]).reshape(1) if S == 1
                       else solve_simplex3(Gs[None], cs[None])[0])
                lag = dict(const=sse_sub(np.broadcast_to(gcs, g_lag.shape), sub_) / n_sub,
                           lag=sse_sub(g_lag, sub_) / n_sub, model=sse_sub(go[sub_], sub_) / n_sub)
                acc[split] = dict(half=half, lag=lag, model=sse(go) / n_cells, const=sse(np.broadcast_to(gc, go.shape)) / n_cells,
                                  oracle=sse(go_) / n_cells, sse_model=sse_model / n_cells,
                                  max_err=max_err, go=go_, go_model=go, F=Fe, sse=sse,
                                  n=n_cells, gconst=gc, gstd_oracle=float(go_.std(axis=(0, 1)).mean()),
                                  gstd_model=float(go.std(axis=(0, 1)).mean()))
            t = acc["test"]
            row = dict(data=data, seed=seed, decision=a.decision,
                       mse_csv=float(rsel[rsel.seed == seed].MSE.iloc[0]),
                       mse_model_direct=t["sse_model"], mse_model=t["model"],
                       mse_const=t["const"], mse_oracle=t["oracle"],
                       g_const=",".join(f"{v:.3f}" for v in np.ravel(t["gconst"])),
                       gstd_model=t["gstd_model"], gstd_oracle=t["gstd_oracle"],
                       decomp_err=max(acc["val"]["max_err"], t["max_err"]),
                       half_const=t["half"]["const"], half_first=t["half"]["first"], half_model=t["half"]["model"],
                       lag_const=t["lag"]["const"], lag_prev=t["lag"]["lag"], lag_model=t["lag"]["model"])
            if S == 1:
                # can the spectrum predict the oracle value?  fit on VAL, apply on TEST
                v = acc["val"]
                Xv = np.concatenate([v["F"].reshape(-1, v["F"].shape[-1]),
                                     np.ones((v["F"].shape[0] * v["F"].shape[1], 1))], 1)
                yv = v["go"].reshape(-1)
                beta, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
                Xt = np.concatenate([t["F"].reshape(-1, t["F"].shape[-1]),
                                     np.ones((t["F"].shape[0] * t["F"].shape[1], 1))], 1)
                gp = np.clip(Xt @ beta, *lo_hi[a.decision]).reshape(t["go"].shape)
                row["mse_fft_lin"] = t["sse"](gp) / t["n"]
                yt = t["go"].reshape(-1)
                row["r2_fft"] = 1 - ((Xt @ beta - yt) ** 2).mean() / yt.var()
            rows.append(row)
            print(f"  s{seed}: csv {row['mse_csv']:.5f} direct {row['mse_model_direct']:.5f} "
                  f"quad {row['mse_model']:.5f} (decomp err {row['decomp_err']:.1e}) | "
                  f"const {row['mse_const']:.5f}  oracle {row['mse_oracle']:.5f}"
                  + (f"  fft_lin {row['mse_fft_lin']:.5f} (R2 {row['r2_fft']:.2f})" if S == 1 else ""))

    out = pd.DataFrame(rows)
    path = a.out or f"results/_oracle_gate_{a.decision}.csv"
    out.to_csv(path, index=False)
    print("\n=== test MSE, mean over seeds; % vs 'const' (never adapt) ===")
    for data, g in out.groupby("data", sort=False):
        m = g.mean(numeric_only=True)
        pc = lambda v: 100 * (v / m.mse_const - 1)
        line = (f"{data:13s} const {m.mse_const:.4f} | model(FFT) {pc(m.mse_model):+6.2f}% | "
                f"oracle {pc(m.mse_oracle):+6.2f}%")
        if "mse_fft_lin" in g:
            line += f" | best linear-from-FFT {pc(m.mse_fft_lin):+6.2f}% (R2 {m.r2_fft:.2f})"
        line += f" | g spread: model {m.gstd_model:.3f} oracle {m.gstd_oracle:.3f}"
        print(line)
        print(f"{'':13s}   carry-over: choose on 1st half -> 2nd half {100*(m.half_first/m.half_const-1):+6.2f}%"
              f" (model {100*(m.half_model/m.half_const-1):+6.2f}%) | previous window's best g "
              f"{100*(m.lag_prev/m.lag_const-1):+6.2f}% (model {100*(m.lag_model/m.lag_const-1):+6.2f}%)")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
