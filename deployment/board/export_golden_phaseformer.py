# -*- coding: utf-8 -*-
"""Export golden vectors + weights for the PhaseFormer port (src/phaseformer_infer.c).

PhaseFormer (ICLR 2026).  Layers, latent
width, routers, heads and the position embedding come from the candidate
setting (per H), the period from the dataset's period rule; --config / --data
resolve them exactly as the experiments do (tools/results/protocol.resolve,
industrial group).  The structural values are written into the weights header.

Real upstream model via the benchmark wrapper; every weight re-drawn at
random (README: blind-golden rule).  Stage vectors are taken with forward
hooks on the real model, so nothing is re-implemented here.

Run:  python deployment/board/export_golden_phaseformer.py [--config ETTh2] [--data MetroPT3] [--pred_len H] [--d_model D]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from baselines.PhaseFormer.wrapper import build   # noqa: E402
from tools.results.protocol import resolve        # noqa: E402
from golden_io import emit_array              # noqa: E402

WEIGHT_SEED = 2025
INPUT_SEED = 7
# wrapper defaults (baselines/PhaseFormer/wrapper.py) for keys a setting leaves unset
KEYS = {"phaseformer_period_len": 24, "phaseformer_layers": 3, "phaseformer_latent_dim": 4,
        "phaseformer_routers": 8, "phaseformer_heads": 1, "phaseformer_pos_embed": True,
        "phaseformer_enc_hidden": 16, "phaseformer_pred_hidden": 32}

OUT_DIR = Path(__file__).resolve().parent / "golden"


def geometry(config: str, H: int, data: str = "MetroPT3") -> dict:
    _, _, ov = resolve("PhaseFormer", "mfg", data, H, "cand:" + config)
    return {k: (bool(ov.get(k, v)) if isinstance(v, bool) else int(ov.get(k, v))) for k, v in KEYS.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_len", type=int, default=96)
    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--d_model", type=int, default=7)
    ap.add_argument("--data", default="MetroPT3", help="industrial dataset (period rule)")
    ap.add_argument("--f64", action="store_true",
                    help="stage/output vectors from a float64 run of the same weights (precision check: "
                         "the C error vs float64 should not exceed PyTorch float32's)")
    ap.add_argument("--config", default="ETTh2", help="candidate: a dataset name of the authors' settings")
    a = ap.parse_args()
    L, H, D = a.seq_len, a.pred_len, a.d_model
    geo = geometry(a.config, H, a.data)
    P, NL, DL = geo["phaseformer_period_len"], geo["phaseformer_layers"], geo["phaseformer_latent_dim"]
    R, NH, PE = geo["phaseformer_routers"], geo["phaseformer_heads"], int(geo["phaseformer_pos_embed"])
    print(f"config={a.config} data={a.data} L={L} H={H} D={D} period={P} layers={NL} latent={DL} "
          f"routers={R} heads={NH} pos_embed={PE}")

    torch.manual_seed(WEIGHT_SEED)
    model = build(argparse.Namespace(**geo), D, L, H).eval()
    with torch.no_grad():                       # blind-golden rule: all weights random
        for name, p in model.named_parameters():
            if "norm" in name and name.endswith("weight") or ".in_proj.1.weight" in name:
                p.copy_(1.0 + 0.3 * torch.randn_like(p))     # LayerNorm gains around 1
            elif p.dim() == 2 and "router" not in name and "pos_embedding" not in name:
                # Linear weights scaled by 1/sqrt(fan_in): with a flat 0.5 the wide settings
                # (latent 32/128) blow activations up until float32 itself (PyTorch included)
                # misses the 1e-5/1e-4 tolerance against float64
                p.normal_(0.0, p.shape[1] ** -0.5)
            else:
                p.normal_(0.0, 0.5)
    print(f"params = {sum(p.numel() for p in model.parameters())}")
    up = model.model
    units = up.routing_layers

    cap = {}
    up.embedding.register_forward_hook(lambda m, i, o: cap.__setitem__("z0", o.detach().clone()))
    up.predictor.register_forward_pre_hook(lambda m, i: cap.__setitem__("zl", i[0].detach().clone()))

    g = torch.Generator().manual_seed(INPUT_SEED)
    x = (torch.randn(1, L, D, generator=g) * 3.0
         + torch.linspace(-5.0, 5.0, D).view(1, 1, D))
    if a.f64:
        with torch.no_grad():
            y32 = model(x)
        model = model.double()
        x = x.double()
    with torch.no_grad():
        y_ref = model(x)
        if a.f64:
            print(f"torch float32 vs float64: max abs {(y32.double() - y_ref).abs().max().item():.3e}")
        mu = x.mean(dim=1)[0]
        sigma = (x.var(dim=1, unbiased=False) + 1e-5).sqrt()[0]
    assert y_ref.shape == (1, H, D)

    def stack(fn, idx=None):
        idx = range(NL) if idx is None else idx
        return torch.stack([fn(units[li]) for li in idx])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "phaseformer_weights.h").open("w", newline="\n") as f:
        f.write("/* Auto-generated by deployment/board/export_golden_phaseformer.py - do not edit. */\n")
        f.write(f"/* config={a.config}, data={a.data}, seed={WEIGHT_SEED} */\n")
        f.write("#ifndef PHASEFORMER_WEIGHTS_H\n#define PHASEFORMER_WEIGHTS_H\n\n")
        f.write(f"#define PF_H_GEN {H}\n#define PF_L_GEN {L}\n")
        f.write(f"#define PF_P {P}\n#define PF_DL {DL}\n#define PF_NL {NL}\n#define PF_R {R}\n"
                f"#define PF_NH {NH}\n#define PF_PE {PE}\n\n")
        emit_array(f, "PF_EMB_W", up.embedding.projection.weight)     # (DL, NX)
        emit_array(f, "PF_EMB_B", up.embedding.projection.bias)
        emit_array(f, "PF_EMB_G", up.embedding.norm.weight)
        emit_array(f, "PF_EMB_BT", up.embedding.norm.bias)
        if PE:
            emit_array(f, "PF_POS", stack(lambda u: u.interact.pos_embedding))   # (NL, P, DL)
        emit_array(f, "PF_ROUTER", stack(lambda u: u.interact.router))           # (NL, R, DL)
        for tag, att in (("S", "router_sender"), ("R", "router_receiver")):
            for k, proj in (("Q", "query_projection"), ("K", "key_projection"),
                            ("V", "value_projection"), ("O", "out_projection")):
                lin = lambda u: getattr(getattr(u.interact, att), proj)
                emit_array(f, f"PF_{tag}{k}_W", stack(lambda u: lin(u).weight))  # (NL, DL, DL)
                emit_array(f, f"PF_{tag}{k}_B", stack(lambda u: lin(u).bias))
        emit_array(f, "PF_LN1_G", stack(lambda u: u.interact.norm1.weight))
        emit_array(f, "PF_LN1_B", stack(lambda u: u.interact.norm1.bias))
        emit_array(f, "PF_LN2_G", stack(lambda u: u.interact.norm2.weight))
        emit_array(f, "PF_LN2_B", stack(lambda u: u.interact.norm2.bias))
        emit_array(f, "PF_M1_W", stack(lambda u: u.interact.mlp[0].weight))     # (NL, 4DL, DL)
        emit_array(f, "PF_M1_B", stack(lambda u: u.interact.mlp[0].bias))
        emit_array(f, "PF_M2_W", stack(lambda u: u.interact.mlp[2].weight))     # (NL, DL, 4DL)
        emit_array(f, "PF_M2_B", stack(lambda u: u.interact.mlp[2].bias))
        if NL > 1:
            mid_in, mid_out = range(1, NL), range(0, NL - 1)
            emit_array(f, "PF_IN_W", stack(lambda u: u.in_proj[0].weight, mid_in))   # (NL-1, DL, NX)
            emit_array(f, "PF_IN_B", stack(lambda u: u.in_proj[0].bias, mid_in))
            emit_array(f, "PF_IN_G", stack(lambda u: u.in_proj[1].weight, mid_in))
            emit_array(f, "PF_IN_BT", stack(lambda u: u.in_proj[1].bias, mid_in))
            emit_array(f, "PF_OUT_W", stack(lambda u: u.out_proj.weight, mid_out))   # (NL-1, NX, DL)
            emit_array(f, "PF_OUT_B", stack(lambda u: u.out_proj.bias, mid_out))
        emit_array(f, "PF_PRED_W", up.predictor.decoder.weight)       # (NY, DL)
        emit_array(f, "PF_PRED_B", up.predictor.decoder.bias)
        f.write("#endif /* PHASEFORMER_WEIGHTS_H */\n")

    with (OUT_DIR / "phaseformer_vectors.h").open("w", newline="\n") as f:
        f.write("/* Auto-generated by deployment/board/export_golden_phaseformer.py - do not edit. */\n")
        f.write(f"/* weight_seed={WEIGHT_SEED}, input_seed={INPUT_SEED} */\n")
        f.write("#ifndef PHASEFORMER_VECTORS_H\n#define PHASEFORMER_VECTORS_H\n\n")
        f.write(f"#define PF_D_GEN {D}\n\n")
        emit_array(f, "PG_X", x[0])                 # (L, D)
        emit_array(f, "PG_MU", mu)
        emit_array(f, "PG_SIGMA", sigma)
        emit_array(f, "PG_Z0", cap["z0"][0])        # (D, P, DL)
        emit_array(f, "PG_ZL", cap["zl"][0])        # (D, P, DL)
        emit_array(f, "PG_Y", y_ref[0])             # (H, D)
        f.write("#endif /* PHASEFORMER_VECTORS_H */\n")

    print(f"wrote {OUT_DIR / 'phaseformer_weights.h'}")
    print(f"wrote {OUT_DIR / 'phaseformer_vectors.h'}")


if __name__ == "__main__":
    main()
