"""Scalar (C-style) emulation of MixLinear forward — verifies the port ALGORITHM
against the bit-exact staged reference, using only Python scalar loops (no torch
broadcasting). If this matches, transcribing to C is mechanical.

Config: seq=96, pred=96, D=7, period=24 -> seg_x=seg_y=4, sqrt=2, lpf=4, alpha=0.5.
"""
import argparse, math, sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
import torch
import torch.nn.functional as F
from baselines.MixLinear.wrapper import build

S, P_len, D, PER = 96, 96, 7, 24
SEGX = SEGY = 4
SQ = 2
LPF = 4
ALPHA = 0.5

torch.manual_seed(2025)
wrapped = build(argparse.Namespace(), D, S, P_len).eval()
m = wrapped.model
g = torch.Generator().manual_seed(7)
x0t = torch.randn(1, S, D, generator=g)

# ---- reference (staged == model, proven 0 diff) ----
with torch.no_grad():
    seq_mean_t = x0t.mean(dim=1).unsqueeze(1)
    xp = (x0t - seq_mean_t).permute(0, 2, 1)
    xc_t = m.conv1d(xp.reshape(-1, 1, S)).reshape(-1, D, S) + xp
    xr_t = xc_t.reshape(1, D, -1, PER).permute(0, 1, 3, 2)
    x_o = F.pad(xr_t, (0, SQ**2 - xr_t.shape[-1], 0, 0, 0, 0)).reshape(1, D, PER, SQ, SQ)
    x_o = m.TLinear1(x_o).permute(0, 1, 2, 4, 3)
    x_t_ref = m.TLinear2(x_o).permute(0, 1, 2, 4, 3).reshape(1, D, PER, -1).permute(0, 1, 3, 2).reshape(1, D, -1).permute(0, 2, 1)
    xf = torch.fft.fft(xr_t, dim=3)[:, :, :, :LPF]
    xf = m.FLinear2(m.FLinear1(xf)).reshape(1, D, PER, -1)
    x_f_ref = torch.fft.ifft(xf, dim=3).float().permute(0, 1, 3, 2).reshape(1, D, -1).permute(0, 2, 1)
    y_ref = (x_t_ref[:, :P_len, :] * ALPHA + seq_mean_t + x_f_ref[:, :P_len, :] * (1 - ALPHA))[0]

# ---- weights as plain python floats ----
conv_w = m.conv1d.weight.detach().flatten().tolist()          # 25
TL1 = m.TLinear1.weight.detach().tolist()                     # [2][2]  (out,in)
TL2 = m.TLinear2.weight.detach().tolist()
FL1re = m.FLinear1.weight.real.detach().tolist()              # [2][4]
FL1im = m.FLinear1.weight.imag.detach().tolist()
FL2re = m.FLinear2.weight.real.detach().tolist()              # [4][2]
FL2im = m.FLinear2.weight.imag.detach().tolist()
x0 = x0t[0].tolist()                                          # [96][7]
COS = [math.cos(2 * math.pi * i / SEGX) for i in range(SEGX)]
SIN = [math.sin(2 * math.pi * i / SEGX) for i in range(SEGX)]

# ---- scalar forward ----
mean = [sum(x0[t][c] for t in range(S)) / S for c in range(D)]
# channel-major (x - mean)
x = [[x0[s][c] - mean[c] for s in range(S)] for c in range(D)]
# conv1d residual (shared 1->1, same padding=12, k=25)
PAD = 12
xc = [[0.0] * S for _ in range(D)]
for c in range(D):
    for s in range(S):
        acc = 0.0
        for j in range(25):
            ip = s + j - PAD
            if 0 <= ip < S:
                acc += conv_w[j] * x[c][ip]
        xc[c][s] = acc + x[c][s]
# segment: xr[c][p][sx] = xc[c][sx*PER + p]
xr = [[[xc[c][sx * PER + p] for sx in range(SEGX)] for p in range(PER)] for c in range(D)]

x_t = [[0.0] * D for _ in range(P_len)]   # (96,7)
x_f = [[0.0] * D for _ in range(P_len)]
for c in range(D):
    for p in range(PER):
        seg = xr[c][p]                      # 4 reals
        # --- time branch: grid[r][col] = seg[r*2+col] ---
        grid = [[seg[r * SQ + col] for col in range(SQ)] for r in range(SQ)]
        # TL1 on last dim (col): t1[r][o] = sum_col TL1[o][col]*grid[r][col]
        t1 = [[sum(TL1[o][col] * grid[r][col] for col in range(SQ)) for o in range(SQ)] for r in range(SQ)]
        # permute -> t1p[o][r] = t1[r][o]
        # TL2 on last dim (r): t2[o][o2] = sum_r TL2[o2][r]*t1p[o][r] = sum_r TL2[o2][r]*t1[r][o]
        t2 = [[sum(TL2[o2][r] * t1[r][o] for r in range(SQ)) for o2 in range(SQ)] for o in range(SQ)]
        # permute -> t2p[o2][o]=t2[o][o2]; flatten kk=o2*2+o ; then t = kk*PER + p
        for o2 in range(SQ):
            for o in range(SQ):
                kk = o2 * SQ + o
                x_t[kk * PER + p][c] = t2[o][o2]
        # --- freq branch ---
        Xre = [sum(seg[sx] * COS[(j * sx) % SEGX] for sx in range(SEGX)) for j in range(LPF)]
        Xim = [-sum(seg[sx] * SIN[(j * sx) % SEGX] for sx in range(SEGX)) for j in range(LPF)]
        # FLinear1 (4->2) complex
        Y1re = [sum(Xre[j] * FL1re[o][j] - Xim[j] * FL1im[o][j] for j in range(LPF)) for o in range(2)]
        Y1im = [sum(Xre[j] * FL1im[o][j] + Xim[j] * FL1re[o][j] for j in range(LPF)) for o in range(2)]
        # FLinear2 (2->4) complex
        Zre = [sum(Y1re[i] * FL2re[o][i] - Y1im[i] * FL2im[o][i] for i in range(2)) for o in range(SEGY)]
        Zim = [sum(Y1re[i] * FL2im[o][i] + Y1im[i] * FL2re[o][i] for i in range(2)) for o in range(SEGY)]
        # iFFT real (length 4): xf[k] = (1/N) sum_j (Zre[j]cos - Zim[j]sin)
        for k in range(SEGY):
            xf_k = sum(Zre[j] * COS[(j * k) % SEGX] - Zim[j] * SIN[(j * k) % SEGX] for j in range(SEGY)) / SEGY
            x_f[k * PER + p][c] = xf_k

# mix
y = [[x_t[t][c] * ALPHA + mean[c] + x_f[t][c] * (1 - ALPHA) for c in range(D)] for t in range(P_len)]

# ---- compare ----
def maxdiff(a, ref):
    return max(abs(a[i][j] - ref[i][j].item()) for i in range(len(a)) for j in range(len(a[0])))

print(f"x_t maxdiff = {maxdiff(x_t, x_t_ref[0]):.3e}")
print(f"x_f maxdiff = {maxdiff(x_f, x_f_ref[0]):.3e}")
print(f"y   maxdiff = {maxdiff(y, y_ref):.3e}")
