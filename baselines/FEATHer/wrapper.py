"""Build FEATHer from the worker's args (scripts/train.py)."""
import inspect

from baselines.FEATHer import ablation, variants
from baselines.FEATHer.FEATHer import FEATHer

# worker flag (attribute, value) -> the class that swaps a part; anything else builds FEATHer itself
VARIANTS = {
    ("gate_mode", "none"): ablation.FEATHerNoGate,         # ablation gatenone
    ("gate_mode", "static"): ablation.FEATHerStaticGate,   # ablation gatestatic
    ("use_gate", False): ablation.FEATHerUniformGate,      # ablation gateuni (--gate_uniform)
    ("head_mode", "linear"): ablation.FEATHerLinearHead,   # ablation headlin
    ("front", "none"): variants.FEATHerNoFront,            # control FEATHer_frontnone
    ("mix_sep", 1): variants.FEATHerMixSep,                # control FEATHer_mixsep
}


def build(args, n_features, seq_len, pred_len):
    """Every FEATHer argument the worker sets (not None) is passed through; the rest keep the class default."""
    picked = [cls for (name, value), cls in VARIANTS.items() if getattr(args, name, None) == value]
    assert len(picked) <= 1, f"more than one FEATHer variant selected: {picked}"
    cls = picked[0] if picked else FEATHer
    names = set(inspect.signature(FEATHer.__init__).parameters) - {"self", "seq_len", "pred_len", "d_model"}
    kw = {n: getattr(args, n) for n in names if getattr(args, n, None) is not None}
    if "agg_shared" in kw:
        kw["agg_shared"] = bool(kw["agg_shared"])               # the worker passes 0 / 1
    return cls(seq_len=seq_len, pred_len=pred_len, d_model=n_features, **kw)
