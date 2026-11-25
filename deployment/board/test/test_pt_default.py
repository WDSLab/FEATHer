# -*- coding: utf-8 -*-
"""pt_default.py vs torch itself (DEFAULT build forced, one thread): every mean / sum / var / layer norm / softmax /
avg_pool / upsample / vector norm call of the tuned cells, compared bit for bit.  --c: the same calls through
src/mcu_pt.c too (x86 gcc in WSL, test/test_pt.c), also bit for bit.
    python test_pt_default.py FEATHer_core,TimeBase [H list] [tiers] [--c]"""
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ["ATEN_CPU_CAPABILITY"] = "default"
import numpy as np   # noqa: E402
import torch         # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import board_run as br     # noqa: E402
import op_count as OC      # noqa: E402
import pt_default as PD    # noqa: E402

torch.set_num_threads(1)
assert torch.backends.cpu.get_cpu_capability() == "NO AVX"


def emulate(name, args, kw, c):
    x = args[0]
    if name in ("mean", "sum"):
        dims = args[1] if len(args) > 1 and args[1] is not None else []
        return (PD.mean if name == "mean" else PD.sum_)(x, dims, c)
    if name == "var":
        return PD.var(x, args[1], kw.get("correction", 1), c)
    if name == "native_layer_norm":
        return PD.layer_norm(x, args[2], args[3], args[4], c)
    if name == "_softmax":
        return PD.softmax(x, c)
    if name == "avg_pool2d":
        return PD.avg_pool(x, args[1][1], c)
    if name == "upsample_linear1d":
        return PD.upsample_linear(x, args[1][0], c)
    if name == "linalg_vector_norm":
        return PD.vector_norm_last2(x, c)
    return None


def c_case(name, args, kw):
    """(op code, ints, floats) for test_pt.c, or None when mcu_pt.c has no such kernel."""
    x = args[0]
    shape, stride = list(x.shape), list(x.stride())
    st = x.untyped_storage()
    buf = torch.empty(0).set_(st, 0, (st.nbytes() // 4,), (1,))[x.storage_offset():].numpy()
    if name in ("mean", "sum"):
        dims = args[1] if len(args) > 1 and args[1] is not None else range(len(shape))
        red = sum(1 << (d % len(shape)) for d in dims)
        return (0 if name == "sum" else 1), [len(shape), *shape, *[0] * (4 - len(shape)), *stride,
                                             *[0] * (4 - len(shape)), red], buf
    if name == "var":
        d = args[1][0] % len(shape)
        keep = [k for k in range(len(shape)) if k != d and shape[k] > 1]
        assert len(keep) == 1
        return 2, [shape[keep[0]], stride[keep[0]], shape[d], stride[d], int(kw.get("correction", 1))], buf
    if name == "native_layer_norm":
        n = x.shape[-1]
        assert x.is_contiguous()
        g, b = args[2].detach().numpy(), args[3].detach().numpy()
        return 3, [x.numel() // n, n], np.concatenate([x.detach().numpy().ravel(), g, b, [args[4]]]).astype(np.float32)
    if name == "_softmax":
        assert x.is_contiguous()
        return 4, [x.numel() // x.shape[-1], x.shape[-1]], x.detach().numpy().ravel()
    if name == "linalg_vector_norm" and x.shape[-1] * x.shape[-2] > 1:
        assert x.is_contiguous()
        return 5, [x.shape[0], x.shape[-1] * x.shape[-2]], x.detach().numpy().ravel()
    return None


def run_c(cases):
    """cases: [(op, ints, floats)] -> outputs from test_pt.c."""
    board = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(dir=board / "build" if (board / "build").exists() else None) as tmp:
        t = Path(tmp)
        with (t / "in.bin").open("wb") as f:
            f.write(struct.pack("<i", len(cases)))
            for op, ints, fl in cases:
                f.write(struct.pack("<ii", op, len(ints)) + struct.pack(f"<{len(ints)}i", *ints))
                fl = np.ascontiguousarray(fl, np.float32)
                f.write(struct.pack("<i", fl.size) + fl.tobytes())
        wsl = lambda p: "/mnt/" + str(p).replace(":", "").replace("\\", "/").lower()[0] + str(p)[2:].replace("\\", "/")
        cmd = (f"cd {wsl(board)} && gcc -O2 -ffp-contract=off -fno-fast-math -Wall -Isrc {os.environ.get('PT_CFLAGS', '')} src/mcu_pt.c test/test_pt.c "
               f"-o {wsl(t)}/test_pt -lm && {wsl(t)}/test_pt {wsl(t)}/in.bin {wsl(t)}/out.bin")
        subprocess.run(["wsl", "-e", "bash", "-lc", cmd], check=True)
        data = (t / "out.bin").read_bytes()
    outs, p = [], 0
    for _ in cases:
        n = struct.unpack_from("<i", data, p)[0]
        outs.append(np.frombuffer(data, np.float32, n, p + 4).copy())
        p += 4 + 4 * n
    return outs


def main():
    use_c = "--c" in sys.argv
    sys.argv = [a for a in sys.argv if a != "--c"]
    models = sys.argv[1].split(",")
    Hs = [int(h) for h in (sys.argv[2] if len(sys.argv) > 2 else "96,192,336,720").split(",")]
    tiers = (sys.argv[3] if len(sys.argv) > 3 else "1k,chip").split(",")
    tuned = br.P.tuned()
    bad, seen, rows = 0, {}, [0, 0]
    cases, refs = [], []
    _fin = OC.Ops.finish

    def fin(self, r):
        nonlocal bad, rows
        for e in self.rec:
            c = PD.Cnt()
            try:
                y = emulate(e["name"], e["args"], e["kw"], c)
            except NotImplementedError as err:
                print("  NOT MODELLED", e["name"], tuple(e["args"][0].shape), err)
                bad += 1
                continue
            if y is None:
                continue
            o = e["out"][0] if isinstance(e["out"], (tuple, list)) else e["out"]
            t = o.detach().contiguous().numpy().astype(np.float32).ravel()
            same = np.array_equal(t.view(np.uint32), np.asarray(y, np.float32).ravel().view(np.uint32))
            if e["name"] == "_softmax":                       # expf of the library differs now and then: count rows
                K = o.shape[-1]
                d = (t.view(np.uint32) != np.asarray(y, np.float32).ravel().view(np.uint32)).reshape(-1, K).any(1)
                rows[0] += int(d.sum())
                rows[1] += d.size
            if use_c:
                cc = c_case(e["name"], e["args"], e["kw"])
                if cc:
                    cases.append(cc)
                    refs.append((e["name"], o.shape[-1] if o.dim() else 1, t, np.asarray(y, np.float32).ravel()))
            key = (e["name"], tuple(e["args"][0].shape), e["args"][0].stride())
            seen.setdefault(key, [0, 0])[same] += 1
            if not same:
                bad += 1
        return _fin(self, r)

    OC.Ops.finish = fin
    _disp = OC.Ops.__torch_dispatch__

    def disp(self, func, types, args=(), kwargs=None):      # keep the inputs as they were (CMoS divides x in place later)
        if str(func.overloadpacket.__name__) not in ("var", "mean", "sum", "native_layer_norm", "_softmax"):
            return _disp(self, func, types, args, kwargs)
        snap = [torch.empty_strided(a.shape, a.stride()).copy_(a) if isinstance(a, torch.Tensor) and i == 0 else a
                for i, a in enumerate(args)]
        out = _disp(self, func, types, args, kwargs)
        if self.rec and self.rec[-1]["args"] is args:
            self.rec[-1]["args"] = snap
        return out

    OC.Ops.__torch_dispatch__ = disp
    for cell in br.tuned_cells(models, Hs, tiers):
        knobs = tuned[f"{cell['tier']}|{cell['model']}|{cell['data']}|{cell['H']}"]["knobs"]
        OC.torch_ops(cell["tier"], cell["model"], cell["data"], cell["H"], knobs)
    for k, (n_bad, n_ok) in sorted(seen.items(), key=str):
        if n_bad:
            print("  DIFF", k, f"{n_bad} bad / {n_ok} ok")
    print(f"softmax rows differing (expf library): {rows[0]} of {rows[1]}")
    if use_c:
        nb, nt, sr = 0, 0, [0, 0, 0]
        for (name, K, t, y), c in zip(refs, run_c(cases)):
            if name == "_softmax":                            # C (glibc expf) vs torch, and vs pt_default (exact expf)
                d = (t.view(np.uint32) != c.view(np.uint32)).reshape(-1, K).any(1)
                d2 = (y.view(np.uint32) != c.view(np.uint32)).reshape(-1, K).any(1)
                sr[0] += int(d.sum()); sr[1] += int(d2.sum()); sr[2] += d.size
                continue
            nt += 1
            if not np.array_equal(t.view(np.uint32), c.view(np.uint32)):
                nb += 1
                print("  C DIFF", name, t.size, int((t.view(np.uint32) != c.view(np.uint32)).sum()), "values")
        print(f"C: {nt - nb} of {nt} calls bit-exact with torch; softmax rows differing vs torch {sr[0]}, "
              f"vs pt_default {sr[1]} of {sr[2]}")
    print(f"{len(seen)} call shapes, {sum(v[1] for v in seen.values())} calls bit-exact, {bad} not")


if __name__ == "__main__":
    main()
