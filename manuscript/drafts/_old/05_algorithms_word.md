# Algorithm 1 — FEATHer Train Pipeline

> **Source**: `tex_workspace/feather_raw.tex` lines 540--610 (longtable
> converted to pseudocode). **Critical R8 #5 fixes applied below** —
> the corrected lines are flagged inline.

---

## Algorithm 1: FEATHer training step

**Input**
- Input sequence $X \in \mathbb{R}^{L \times D}$
- Number of active branches $B \in \{2, 3, 4\}$ and active set $\mathcal{B} \subseteq \{p, h, m, l\}$
- Period $P$ and horizon $H$
- Model parameters $\Theta$ (decomposition filters, DTK, gating, SPK)

**Output**
- Forecast $\widehat{Y} \in \mathbb{R}^{H \times D}$

**Procedure** — repeat until convergence:

```
// Step 1: Structured multiscale temporal decomposition
 1: x̃ ← InstanceNorm(x)
 2: h^{(p)} ← DWConv1D(x̃; kernel = 1)            ▷ point branch
 3: if B = 4 then
 4:     h^{(h)} ← DWConv1D(x̃; kernel = 3)        ▷ high branch     [R8 #5 fix: was h^{(p)} in TPAMI]
 5: end if
 6: if B ≥ 3 then
 7:     h^{(m)} ← DWConv1D(x̃; kernel = 5)        ▷ mid branch
 8: end if
 9: x↓ ← AvgPool1D(x̃; stride = 4)
10: h^{(l)} ← Upsample1D(x↓; L)                    ▷ low branch

// Step 2: Dense Temporal Kernel (DTK), shared across branches
11: for each active branch s ∈ 𝓑 do
12:     z^{(s)} ← h^{(s)} W_in
13:     u^{(s)} ← DWConv(z^{(s)}; k_temp)
14:     h̃^{(s)} ← u^{(s)} W_out                    [R8 #5 fix: unified h̃^{(s)} notation; TPAMI used h̃(s)/H'^{(s)} inconsistently]
15: end for

// Step 3: Adaptive Branch-level Gating
16: F ← FFT(x̃)                                    ▷ spectral signature
17: M ← |F|                                        ▷ magnitude spectrum
18: a ← MeanChannel(M)                             [R8 #5 fix: removed extra ')' — TPAMI line 21 wrote "MeanChannel(M))"]
19: z_g ← Conv1D(a)
20: g ← softmax(Pool(z_g))                         ▷ g ∈ ℝ^B (band weights)
21: h ← Σ_{s ∈ 𝓑} g_s · h̃^{(s)}

// Step 4: Sparse Period Kernel (SPK)
22: h_agg ← h + DWConv(h; k_slide)                 ▷ sliding aggregation
23: Reshape h_agg into H̃_d ∈ ℝ^{P × n}            ▷ n = L / P
24: for each channel d = 1, …, D and phase p = 1, …, P do
25:     Y_d(p, :) ← H̃_d(p, :) · W
26: end for
27: Reassemble Y_d into ŷ ∈ ℝ^{H × D}

// Step 5: Loss computation and parameter update
28: L ← MSE(ŷ, y) + λ_spec · L_spec                [Note: spec loss is FEATHer-specific (utils/losses.py)]
29: Θ ← Θ − η ∇_Θ L
```

End repeat.

---

## Notation table (R8 #5 — unified across body, algorithm, and figures)

| Symbol | Meaning |
|---|---|
| $X \in \mathbb{R}^{L \times D}$ | Input window (length $L$, $D$ variables) |
| $\widetilde{X}$ | Instance-normalized input |
| $X^{(b)}$, $H^{(b)}$ | Branch input / DTK output (used in Section 3) |
| $h^{(s)}$, $\widetilde{h}^{(s)}$ | Algorithm-level branch tensors (post-decomposition / post-DTK) |
| $\mathcal{B}$ | Active branch set $\subseteq \{p, h, m, l\}$ |
| $B = |\mathcal{B}|$ | Number of active branches in $\{2, 3, 4\}$ |
| $S$ | DTK latent width |
| $P$ | SPK period |
| $n = L/P$ | Number of complete cycles in the input window |
| $m = H/P$ | Number of cycles in the forecast horizon |
| $W_{\text{in}}, W_{\text{out}}, k_{\text{temp}}$ | DTK projection / depthwise kernel |
| $g \in \mathbb{R}^B$ | Branch fusion weights (softmax over gating logits) |
| $\lambda_{\text{spec}}$ | Spectral-separation loss weight (FEATHer-specific) |

---

## R8 #5 audit summary

| TPAMI line | Issue | Corrected here |
|---|---|---|
| Line 5 of TPAMI Algorithm 1 | `h^{(p)} ← DWConv1d(...; kernel = 3)` in the "high branch" block | `h^{(h)} ← DWConv1D(...; kernel = 3)` (Line 4 above) |
| Line 21 of TPAMI Algorithm 1 | `a ← MeanChannel(M))` had a stray closing paren | `a ← MeanChannel(M)` (Line 18 above) |
| Body vs. algorithm symbol mismatch | TPAMI alternated between $X^{(b)}/H^{(b)}$, $h^{(s)}$, $\widetilde{h}(s)$ | Body keeps $X^{(b)}/H^{(b)}$; algorithm uses $h^{(s)}/\widetilde{h}^{(s)}$; notation table above maps the two consistently |
| Training-objective ambiguity (R8 #5 last bullet) | TPAMI body said "such as MAE or MSE" while Sec 5 used MSE in practice | This pseudocode states the actual loss explicitly: $L = \text{MSE} + \lambda_{\text{spec}} L_{\text{spec}}$. Body Sec 3.6 to be reworded accordingly. |
