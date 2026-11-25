"""C-array writer shared by the export_golden_*.py scripts.  python deployment/board/export/golden_io.py writes
the 96-point pocketfft plan golden/mcu_rfft96_plan.h (board_prep.install writes it before every build)."""
from pathlib import Path

import torch

OUT_DIR = Path(__file__).resolve().parents[1] / "golden"


def load_ckpt(model, path: str) -> None:
    """Trained weights instead of the golden's random ones (streaming replay); strict: a checkpoint of another
    structure stops the export."""
    model.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
    print(f"loaded checkpoint {path}")


def emit_array(f, name: str, t: torch.Tensor) -> None:
    flat = t.detach().to(torch.float32).flatten().tolist()
    f.write(f"/* shape={list(t.shape)} */\n")
    f.write(f"static const float {name}[{len(flat)}] = {{\n")
    for i in range(0, len(flat), 6):
        chunk = ", ".join(f"{v:.9e}f" for v in flat[i:i + 6])
        f.write(f"  {chunk},\n")
    f.write("};\n\n")


if __name__ == "__main__":
    from pocket_plan import write_rfft96
    print("wrote", write_rfft96(OUT_DIR))
