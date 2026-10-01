"""What the time -> frequency transform actually looks like, on one real ETTh1 window.

Labels are English on purpose: the default matplotlib font has no Korean glyphs
and renders them as boxes.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils.data_factory import data_provider

L = 96
ds, dl = data_provider("data/", "ETTh1", "M", 256, L, 48, 96, "test")
xb = next(iter(dl))[0].float().numpy()
x = xb[0, :, 0]
x = (x - x.mean()) / x.std()          # same instance-norm the model sees

X = np.fft.rfft(x)
amp = np.abs(X) * 2 / L
amp[0] /= 2
freqs = np.arange(len(X))
t = np.arange(L)

fig, ax = plt.subplots(2, 2, figsize=(13, 7.5))

# (a) time domain
ax[0, 0].plot(t, x, lw=1.6, color="#2b6cb0")
ax[0, 0].set_title("(a) TIME domain: the signal, 96 numbers", fontsize=11)
ax[0, 0].set_xlabel("time (steps)")
ax[0, 0].set_ylabel("value")
ax[0, 0].grid(alpha=.3)

# (b) frequency domain
ax[0, 1].bar(freqs, amp, width=.8, color="#c05621")
ax[0, 1].set_title("(b) FREQUENCY domain: the SAME signal, 49 numbers\n"
                   "(the time axis is replaced, not extended)", fontsize=11)
ax[0, 1].set_xlabel("frequency index k     (period = 96/k steps)")
ax[0, 1].set_ylabel("amplitude")
ax[0, 1].grid(alpha=.3)
for k in np.argsort(amp)[::-1][:3]:
    if k == 0:
        continue
    ax[0, 1].annotate("k=%d\nperiod %.0f" % (k, L / k), xy=(k, amp[k]),
                      xytext=(k + 4, amp[k] * .93), fontsize=8,
                      arrowprops=dict(arrowstyle="->", lw=.8))

# (c) individual components are plain sinusoids
comp_ax = ax[1, 0]
for k in [k for k in np.argsort(amp)[::-1] if k > 0][:3]:
    Xk = np.zeros_like(X)
    Xk[k] = X[k]
    comp_ax.plot(t, np.fft.irfft(Xk, n=L), lw=1.4,
                 label="k=%d  (period %.0f steps, amp %.2f)" % (k, L / k, amp[k]))
comp_ax.set_title("(c) each frequency component is just one sin/cos wave", fontsize=11)
comp_ax.set_xlabel("time (steps)")
comp_ax.legend(fontsize=8)
comp_ax.grid(alpha=.3)

# (d) cumulative reconstruction
rec_ax = ax[1, 1]
rec_ax.plot(t, x, lw=2.2, color="black", label="original", alpha=.85)
order = np.argsort(np.abs(X))[::-1]
for k in (1, 3, 10):
    Xk = np.zeros_like(X)
    Xk[order[:k]] = X[order[:k]]
    r = np.fft.irfft(Xk, n=L)
    ev = 100 * (1 - ((r - x) ** 2).sum() / (x ** 2).sum())
    rec_ax.plot(t, r, lw=1.3, ls="--",
                label="top %d components  (%.0f%% explained)" % (k, ev))
rec_ax.set_title("(d) a handful of components already rebuilds the signal", fontsize=11)
rec_ax.set_xlabel("time (steps)")
rec_ax.legend(fontsize=8)
rec_ax.grid(alpha=.3)

for a in ax.ravel():
    a.tick_params(labelsize=8)
plt.tight_layout()
out = "analysis/diagnostics/fft_explained.png"
plt.savefig(out, dpi=140)
print("saved:", os.path.abspath(out))
print("input 96 reals -> rfft 49 complex -> %d degrees of freedom (identical)"
      % (49 * 2 - 2))
