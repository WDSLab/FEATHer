# -*- coding: utf-8 -*-
"""Float operations and peak stack of one forecast of the F103 firmware, counted in an emulator, against the same count
of the PyTorch model.

Unit (both sides): one float add / sub, mul, div or compare = 1, one math-library call (exp, sqrt, erf, log ...) = 1.
  C       the F103 has no FPU: each of these is a call into the soft-float library or libm.  The firmware that
          board_run.py builds (board_prep.install + board_run.build) runs in Unicorn (Cortex-M3) and every such call
          made from the port code is counted.  The stack is painted before the run and its high-water mark read after
          (as main.c.board does on the board); the forecast is checked against the golden output.
  PyTorch one forward pass (batch 1, all channels) under a dispatch mode that counts each aten op by RULES below, in
          the same unit, per kind.  A sum of n values is n adds (an accumulator that starts at 0, or at the bias:
          a bias added right after mm / bmm / a bias-free conv costs nothing more, as in addmm).
          FFTs are not counted by rule (their algorithm is the library's): they stay at 2.5 N log2 N per real
          transform (5 N log2 N complex), reported apart.  An op without a rule stops the count.
--calibrate checks the two counts on the shared kernels of mcu_kernels.c (test/op_count_calib.c).

    python deployment/board/tools/op_count.py --calibrate
    python deployment/board/tools/op_count.py --plan --models CMoS,TimeBase
    python deployment/board/tools/op_count.py --models CMoS --only 3,4 --out <tmp>/ops.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from elftools.elf.elffile import ELFFile
from torch.utils._python_dispatch import TorchDispatchMode
from unicorn import Uc, UC_ARCH_ARM, UC_MODE_THUMB, UC_MODE_MCLASS, UC_HOOK_CODE
from unicorn.arm_const import UC_ARM_REG_SP, UC_ARM_REG_LR, UC_ARM_REG_R0, UC_ARM_REG_R1, UC_ARM_REG_R2, \
    UC_CPU_ARM_CORTEX_M3

import board_prep as bp
import board_run as br
import chip_verdict as cv

OBJDUMP = br.TOOLS / "arm-none-eabi-objdump.exe"
ADDR2LINE = br.TOOLS / "arm-none-eabi-addr2line.exe"
KINDS = ("add", "mul", "div", "cmp", "libm", "dadd", "dmul", "ddiv", "dcmp")   # d* = double (torch.var's Welford)
KIND = [("add", r"__aeabi_f(add|sub|rsub)$|__(add|sub)sf3$"), ("mul", r"__aeabi_fmul$|__mulsf3$"),
        ("div", r"__aeabi_fdiv$|__divsf3$"), ("cmp", r"__aeabi_c?fcmp|__(eq|ne|lt|le|gt|ge|unord)sf2$"),
        ("dadd", r"__aeabi_d(add|sub|rsub)$|__(add|sub)df3$"), ("dmul", r"__aeabi_dmul$|__muldf3$"),
        ("ddiv", r"__aeabi_ddiv$|__divdf3$"), ("dcmp", r"__aeabi_c?dcmp|__(eq|ne|lt|le|gt|ge|unord)df2$"),
        ("conv", r"__aeabi_(i2f|ui2f|l2f|ul2f|f2iz|f2uiz|f2lz|f2ulz|f2d|d2f|i2d|ui2d|l2d|ul2d|d2iz|d2lz)$|__float|__fix"
                 r"|__extendsfdf2|__truncdfsf2"),
        ("double", r"__aeabi_d|__aeabi_f2d|__aeabi_d2f|df[23]$|__extendsfdf2|__truncdfsf2"),
        ("mem", r"^(memcpy|memset|memmove|__aeabi_mem\w*)$")]
COLS = ["tier", "model", "trial", "data", "H", "D", "port", *[f"C_{k}" for k in KINDS], "C_conv", "C_double",
        "C_ops", *[f"T_{k}" for k in KINDS], "T_ops", "T_fft", "ratio", "C_fft", "C_by_func", "T_by_op", "libm_calls",
        "stack_B", "bss_port_B", "golden_maxabs", "emu_s"]
STUB, HALT, RAM, RAM_LEN = 0x30000000, 0x30000100, 0x20000000, 0x5000
PAINT = 0xAAAAAAAA


# ------------------------------------------------------------------ C side (emulator)
def kind(name: str) -> str:
    for k, rx in KIND:
        if re.search(rx, name):
            return k
    return "libm"


def macro(proj: Path, header: str, exprs: list[str]) -> list[int]:
    """Integer values of macro expressions as the port headers define them (board_cfg.h included)."""
    c = proj / "_m.c"
    c.write_text(f'#include "{header}"\n' + "".join(f"@@ ({e})\n" for e in exprs))
    out = br.run([br.GCC, "-E", "-P", "-mcpu=cortex-m3", "-ICore/Inc", "-DSTM32F103xB", "_m.c"], cwd=proj)
    c.unlink()
    return [int(eval(re.sub(r"(\d+)[uU]?[lL]*", r"\1", l[2:]).replace("/", "//")))
            for l in out.splitlines() if l.startswith("@@")]


def golden(proj: Path, stem: str) -> tuple[np.ndarray, np.ndarray]:
    txt = (proj / "Core" / "Inc" / f"{stem}_vectors.h").read_text(encoding="utf-8")
    arr = lambda suf: np.array([float(v.rstrip("f")) for v in re.search(
        r"static const float \w+_" + suf + r"\[\d+\] = \{(.*?)\};", txt, re.S).group(1).replace(",", " ").split()],
        dtype=np.float32)
    return arr("X"), arr("Y")


def port_globals(objs: list[Path]) -> set[str]:
    """Global symbols defined in the port objects."""
    out = set()
    for o in objs:
        with o.open("rb") as fh:
            for s in ELFFile(fh).get_section_by_name(".symtab").iter_symbols():
                if s["st_info"]["bind"] != "STB_LOCAL" and s["st_shndx"] != "SHN_UNDEF" and s.name:
                    out.add(s.name)
    return out


def symbols(elf: ELFFile, port_files: set[str], globs: set[str]):
    """name -> addr, port functions (address ranges), port .bss bytes, s_x of main.c."""
    tab = elf.get_section_by_name(".symtab")
    addr, ranges, bss, sx, cur = {}, [], 0, None, None
    for s in tab.iter_symbols():
        t, b = s["st_info"]["type"], s["st_info"]["bind"]
        if t == "STT_FILE":
            cur = s.name.replace("\\", "/").rsplit("/", 1)[-1]
            continue
        if s.name:
            addr.setdefault(s.name, s["st_value"])
        in_port = (b == "STB_LOCAL" and cur in port_files) or (b != "STB_LOCAL" and s.name in globs)
        if t == "STT_FUNC" and in_port:
            ranges.append((s["st_value"] & ~1, (s["st_value"] & ~1) + s["st_size"]))
        if t == "STT_OBJECT" and in_port and s["st_value"] >= RAM and s["st_shndx"] != "SHN_UNDEF":
            bss += s["st_size"]
        if t == "STT_OBJECT" and b == "STB_LOCAL" and cur == "main.c" and s.name == "s_x":
            sx = s["st_value"]
    return addr, ranges, bss, sx


def call_sites(elf_path: Path, ranges, names_in_port: set[str]) -> dict[int, str]:
    """Branches from port code to functions outside it: address -> target name."""
    out = {}
    rx = re.compile(r"^\s*([0-9a-f]+):\s+(?:[0-9a-f ]+\s+)?(bl|blx|b|b\.w|b\.n)\s+[0-9a-f]+\s+<([^>+]+)>\s*$")
    inside = lambda a: any(lo <= a < hi for lo, hi in ranges)
    for line in br.run([OBJDUMP, "-d", "--no-show-raw-insn", elf_path]).splitlines():
        m = rx.match(line)
        if m and inside(int(m.group(1), 16)) and m.group(3) not in names_in_port:
            out[int(m.group(1), 16)] = m.group(3)
    return out


def owners(elf_path: Path, addrs: list[int]) -> tuple[dict[int, str], dict[int, str]]:
    """Innermost (inlined) function of each address and its source file, from the debug line table."""
    out, src, cur = {}, {}, None
    for line in br.run([ADDR2LINE, "-a", "-f", "-i", "-e", elf_path] + [hex(a) for a in addrs]).splitlines():
        if line.startswith("0x"):
            cur = int(line, 16)
        elif cur is not None and cur not in out and ":" not in line:
            out[cur] = line.strip()
        elif cur is not None and cur not in src and ":" in line:
            src[cur] = Path(line.rsplit(":", 1)[0].replace("\\", "/")).name
    return out, src


class Machine:
    """The F103 image in Unicorn: 1 MB flash, the 20 KB SRAM (an overflow faults), a page of `bx lr` stubs."""

    def __init__(self, elf_path: Path, port_files: set[str], objs: list[Path]):
        self.elf_path = elf_path
        with elf_path.open("rb") as f:
            elf = ELFFile(f)
            self.addr, ranges, self.bss_port, self.sx = symbols(elf, port_files, port_globals(objs))
            secs = [(s["sh_addr"], s.data() if s["sh_type"] != "SHT_NOBITS" else b"\0" * s["sh_size"])
                    for s in elf.iter_sections() if s["sh_flags"] & 2 and s["sh_size"]]
        in_port = {n for n, a in self.addr.items() if any(lo <= (a & ~1) < hi for lo, hi in ranges)}
        self.sites = call_sites(elf_path, ranges, in_port)
        uc = self.uc = Uc(UC_ARCH_ARM, UC_MODE_THUMB | UC_MODE_MCLASS)
        uc.ctl_set_cpu_model(UC_CPU_ARM_CORTEX_M3)
        uc.mem_map(0x08000000, 0x100000)
        uc.mem_map(RAM, RAM_LEN)
        uc.mem_map(STUB, 0x1000)
        uc.mem_write(STUB, b"\x70\x47" * 0x80)                  # bx lr
        for a, data in secs:
            uc.mem_write(a, data)
        self.estack = self.addr["_estack"]
        self.end = self.addr.get("_end", (self.addr["_ebss"] + 7) & ~7)     # _end: PROVIDEd only when referenced
        assert self.estack == RAM + RAM_LEN, hex(self.estack)
        uc.mem_write(self.end, PAINT.to_bytes(4, "little") * ((self.estack - self.end) // 4))
        self.hits = {}
        def on_site(_uc, a, _size, _u):
            self.hits[a] = self.hits.get(a, 0) + 1
        for a in self.sites:
            uc.hook_add(UC_HOOK_CODE, on_site, begin=a, end=a)

    def call(self, fn: str, *args) -> int:
        self.uc.reg_write(UC_ARM_REG_SP, self.estack)
        self.uc.reg_write(UC_ARM_REG_LR, HALT | 1)
        for r, v in zip((UC_ARM_REG_R0, UC_ARM_REG_R1, UC_ARM_REG_R2), args):
            self.uc.reg_write(r, v)
        self.uc.emu_start(self.addr[fn] | 1, HALT)
        return self.uc.reg_read(UC_ARM_REG_R0)

    def tally(self) -> dict:
        """Counts since the last tally, per kind and per (inlined) function; peak stack so far."""
        n = {k: 0 for k, _ in KIND} | {"libm": 0}
        libm, part, fft = {}, {}, {k: 0 for k in KINDS}
        own, src = owners(self.elf_path, list(self.hits))
        for a, c in self.hits.items():
            k = kind(self.sites[a])
            n[k] += c
            if k in KINDS:
                part[own[a]] = part.get(own[a], 0) + c
                if src.get(a) == "mcu_pocket.c":                      # the FFT's own operations
                    fft[k] += c
            if k == "libm":
                libm[self.sites[a]] = libm.get(self.sites[a], 0) + c
        self.hits.clear()
        ram = np.frombuffer(bytes(self.uc.mem_read(self.end, self.estack - self.end)), np.uint32)
        return dict(**{f"C_{k}": n[k] for k in (*KINDS, "conv", "double")}, C_ops=sum(n[k] for k in KINDS),
                    C_fft=" ".join(f"{k}:{v}" for k, v in fft.items() if v),
                    C_by_func=" ".join(f"{k}:{v}" for k, v in sorted(part.items(), key=lambda kv: -kv[1])),
                    libm_calls=" ".join(f"{k}:{v}" for k, v in sorted(libm.items())),
                    stack_B=(self.estack - self.end) - 4 * int(np.argmax(ram != PAINT)))


def emulate(elf_path: Path, proj: Path, bm: str, H: int, D: int, port_files: set[str]) -> dict:
    stem = bp.MODELS[bm][2]
    x, ygold = golden(proj, stem)
    m = Machine(elf_path, port_files, [proj / "obj" / (Path(f).stem + ".o") for f in port_files])
    m.uc.mem_write(m.sx, x.tobytes())
    y = np.full((H, D), np.nan, dtype=np.float32)
    def on_emit(_uc, _a, _s, _u):
        d, t = m.uc.reg_read(UC_ARM_REG_R0), m.uc.reg_read(UC_ARM_REG_R1)
        y[t, d] = np.frombuffer(m.uc.reg_read(UC_ARM_REG_R2).to_bytes(4, "little"), np.float32)[0]
    m.uc.hook_add(UC_HOOK_CODE, on_emit, begin=STUB, end=STUB)
    t0 = time.time()
    fwd = "fea2ci_forward_emit" if bm == "feather_ci" else f"{stem}_forward_emit"   # every port
    for d in range(D):
        m.call(fwd, m.sx, d, STUB | 1)
    emu_s = time.time() - t0
    return dict(**m.tally(), bss_port_B=m.bss_port, emu_s=round(emu_s, 1),
                golden_maxabs=float(np.nanmax(np.abs(y - ygold.reshape(H, D)))) if not np.isnan(y).any() else "NaN")


# ------------------------------------------------------------------ PyTorch side (rules)
FREE = {"view", "_unsafe_view", "reshape", "clone", "copy_", "detach", "alias", "expand", "permute", "transpose", "t",
        "unsqueeze", "squeeze", "slice", "select", "cat", "stack", "zeros", "fill_", "lift_fresh", "_to_copy", "flip",
        "constant_pad_nd", "diagonal", "diag_embed", "empty", "randn"}      # randn = the input window, not the model
LIBM = {"sqrt", "exp", "log", "erf", "tanh", "sin", "cos", "rsqrt"}


import pt_default as PD  # noqa: E402

PT_KERNELS = {
    "mean": lambda a, c: PD.mean(a[0], a[1] if len(a) > 1 and a[1] is not None else [], c),
    "sum": lambda a, c: PD.sum_(a[0], a[1] if len(a) > 1 and a[1] is not None else [], c),
    "var": lambda a, c: PD.var(a[0], a[1], 0, c),
    "native_layer_norm": lambda a, c: PD.layer_norm(a[0], a[2], a[3], a[4], c),
    "_softmax": lambda a, c: PD.softmax(a[0], c),
    "avg_pool2d": lambda a, c: PD.avg_pool(a[0], a[1][1], c),
    "upsample_linear1d": lambda a, c: PD.upsample_linear(a[0], a[1][0], c),
    "linalg_vector_norm": lambda a, c: PD.vector_norm_last2(a[0], c),
}


def rule(name: str, args, out) -> dict:
    """Operations of one aten op in the unit above, per kind ({} = free)."""
    n = name.rstrip("_") if name not in FREE else name
    if n in FREE:
        return {}
    o = out[0] if isinstance(out, (tuple, list)) else out
    x = args[0]
    cplx = lambda t: isinstance(t, torch.Tensor) and t.is_complex()
    if n in ("mm", "addmm", "bmm"):
        a, b = (args[1], args[2]) if n == "addmm" else (args[0], args[1])
        mac = o.numel() * a.shape[-1]                       # addmm: the bias starts the accumulator
        f = {0: (1, 1), 1: (2, 2), 2: (4, 4)}[cplx(a) + cplx(b)]
        return {"mul": f[0] * mac, "add": f[1] * mac}
    if n == "convolution":
        w = args[1]                                         # every tap, the zero padding included (as
        taps = o.shape[0] * o.shape[1] * (w.shape[1])       # torch.utils.flop_counter counts a convolution)
        for d in range(2, x.dim()):
            taps *= o.shape[d] * w.shape[d]
        return {"mul": taps, "add": taps}
    if n in ("add", "sub", "rsub"):
        return {"add": o.numel() * (2 if cplx(o) else 1)}
    if n == "mul":
        c = cplx(x) + cplx(args[1])
        return {"mul": o.numel() * {0: 1, 1: 2, 2: 4}[c]} | ({"add": 2 * o.numel()} if c == 2 else {})
    if n == "div":
        return {"div": o.numel() * (2 if cplx(o) else 1)}
    if n == "relu":
        return {"cmp": o.numel()}
    if n == "abs":
        return {"mul": 2 * o.numel(), "add": o.numel(), "libm": o.numel()} if cplx(x) else {}   # real: sign bit
    if n == "pow":
        return {"mul": o.numel()} if args[1] == 2 else {"libm": o.numel()}
    if n in LIBM:
        return {"libm": o.numel()}
    if n in PT_KERNELS:                                      # PyTorch's DEFAULT-build kernel, step by step
        c = PD.Cnt()
        PT_KERNELS[n](args, c)
        c["libm"] += c.pop("floor", 0)
        return dict(c)
    if n == "gelu":                                          # 0.5 x (1 + erf(x / sqrt 2))
        return {"mul": 3 * o.numel(), "add": o.numel(), "libm": o.numel()}
    if n.startswith("_fft"):
        real = o if n.startswith("_fft_c2r") else x
        N = math.prod(real.shape[d] for d in args[1])
        k = 5.0 if n.startswith("_fft_c2c") else 2.5
        return {"fft": k * N * math.log2(max(N, 2)) * real.numel() / N}
    raise NotImplementedError(f"no counting rule for aten.{name} {[tuple(a.shape) for a in args if isinstance(a, torch.Tensor)]}")




class Ops(TorchDispatchMode):
    """Records every aten op of a forward pass; finish() counts all of them in the unit above (every operation
    PyTorch executes - also values sliced away later, a second output, the same op run twice)."""

    def __init__(self):
        super().__init__()
        self.rec, self.sums = [], {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        name = str(func.overloadpacket.__name__)
        key = lambda t: t.untyped_storage().data_ptr()
        if func.is_view:                                     # a view by its schema: nothing computed
            return out
        bias = False                                         # a bias added to a fresh mm / bmm / bias-free conv sum
        if name in ("mm", "bmm"):
            self.sums[key(out)] = -1                         # bias along the last dim
        elif name == "convolution" and args[2] is None:
            self.sums[key(out)] = 1                          # bias along the channel dim
        elif name == "add" and isinstance(args[1], torch.Tensor):
            s, b = (args[0], args[1]) if key(args[0]) in self.sums else (args[1], args[0])
            d = self.sums.get(key(s))
            bias = (d is not None and s.shape == out.shape and b.numel() == s.shape[d]
                    and sum(n > 1 for n in b.shape) <= 1)       # one value per output feature, nothing else
            if bias:
                del self.sums[key(s)]
        self.rec.append(dict(name=name, args=args, kw=kwargs or {}, out=out, bias=bias))
        return out

    def finish(self, result=None) -> "Ops":
        self.n, self.by = {k: 0 for k in (*KINDS, "fft")}, {}
        for e in self.rec:
            if e["bias"]:                                    # the accumulator of the sum starts at the bias
                continue
            for kk, v in rule(e["name"], e["args"], e["out"]).items():
                self.n[kk] += v
                self.by[e["name"]] = self.by.get(e["name"], 0) + v
        return self


def torch_ops(tier: str, model: str, data: str, H: int, knobs: dict) -> Ops:
    """One forward pass of the setting, built as count_flops.py (and the worker) builds it, counted by the rules."""
    from argparse import Namespace
    sys.path.insert(0, str(bp.ROOT / "tools" / "paper"))
    import count_flops as CF
    group = CF.P.tier_group(tier)
    _, _, settings = CF.P.resolve(model, group, data, H, CF.P.mfg_main_config_id(model))
    args = Namespace(**{**CF.S.DEFAULT_ARGS, "model": CF.P.registry_name(model), **settings,
                        **{k: v for k, v in knobs.items() if k != "lr"}})
    C = CF.S.CHANNELS[data]
    torch.manual_seed(0)
    net = CF.get_model(args.model, args, n_features=C, seq_len=CF.S.L, pred_len=H).float().eval()
    x = torch.randn(1, CF.S.L, C)
    ops = Ops()
    with torch.no_grad(), ops:
        y = net(x)
    return ops.finish(y[0] if isinstance(y, (tuple, list)) else y)


def t_cols(t: Ops) -> dict:
    return dict(**{f"T_{k}": round(t.n[k]) for k in KINDS}, T_ops=round(sum(t.n[k] for k in KINDS)),
                T_fft=round(t.n["fft"]), T_by_op=" ".join(f"{k}:{round(v)}" for k, v in
                                                          sorted(t.by.items(), key=lambda kv: -kv[1])))


# ------------------------------------------------------------------ calibration
CALIB = {   # wrapper in test/op_count_calib.c -> the same computation in PyTorch
    "c_conv3_b":  lambda F: F.conv1d(torch.randn(1, 7, 96), torch.randn(7, 1, 3), torch.randn(7), padding=1, groups=7),
    "c_conv5":    lambda F: F.conv1d(torch.randn(1, 7, 96), torch.randn(7, 1, 5), padding=2, groups=7),
    "c_pool":     lambda F: F.avg_pool1d(torch.randn(1, 7, 96), 4),
    "c_softmax":  lambda F: torch.softmax(torch.randn(37), 0),
    "c_gelu":     lambda F: F.gelu(torch.randn(100)),
    "c_norm":     lambda F: (lambda x: (x.mean(0), torch.sqrt(x.var(0, unbiased=False) + 1e-5)))(torch.randn(96, 7)),
}


def calibrate() -> None:
    import torch.nn.functional as F
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        (t / "Core" / "Src").mkdir(parents=True)
        (t / "Core" / "Inc").mkdir()
        for f in ("mcu_kernels", "mcu_pt"):
            shutil.copy2(bp.SRC / f"{f}.c", t / "Core" / "Src")
            shutil.copy2(bp.SRC / f"{f}.h", t / "Core" / "Inc")
        shutil.copy2(bp.SRC / "mcu_emit.h", t / "Core" / "Inc")
        shutil.copy2(bp.BOARD / "test" / "op_count_calib.c", t / "Core" / "Src")
        shutil.copy2(bp.DEFAULT_PROJECT / "STM32F103RBTX_FLASH.ld", t)
        objs = []
        for c in ("mcu_kernels", "mcu_pt", "op_count_calib"):
            objs.append(t / f"{c}.o")
            br.run([br.GCC, f"Core/Src/{c}.c"] + br.CFLAGS + ["-o", objs[-1]], cwd=t)
        elf = t / "calib.elf"
        br.run([br.GCC, "-o", elf, *objs, *br.LDFLAGS, "-nostartfiles", *[f"-Wl,-u,{w}" for w in CALIB]], cwd=t)
        m = Machine(elf, {"mcu_kernels.c", "mcu_pt.c", "op_count_calib.c"}, objs)
        print(f"{'kernel':<11} " + "  ".join(f"{k:>11}" for k in KINDS) + "   (C / PyTorch)")
        for w, f in CALIB.items():
            m.call(w)
            c = m.tally()
            ops = Ops()
            with ops:
                r = f(F)
            ops.finish(r)
            same = all(c[f"C_{k}"] == ops.n[k] for k in KINDS)
            print(f"{w:<11} " + "  ".join(f"{c[f'C_{k}']:>5}/{round(ops.n[k]):<5}" for k in KINDS)
                  + ("   same" if same else f"   C {c['C_ops']} vs {round(sum(ops.n[k] for k in KINDS))}"))


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=br.P.FEATHER_CORE)
    ap.add_argument("--H", default="96,192,336,720")
    ap.add_argument("--tiers", default="1k,chip")
    ap.add_argument("--only", default=None, help="comma list of cell numbers (see --plan)")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--calibrate", action="store_true", help="the shared kernels: C count vs PyTorch rule count")
    ap.add_argument("--out", type=Path, required=False)
    a = ap.parse_args()
    if a.calibrate:
        return calibrate()
    cells = br.tuned_cells([m.strip() for m in a.models.split(",")], [int(h) for h in a.H.split(",")],
                           a.tiers.split(","))
    if a.plan:
        for i, c in enumerate(cells, 1):
            print(f"{i:3d}. {c['tier']:<4} {c['model']:<14} {c['trial']:<5} {c['data']:<12} H={c['H']}")
        return
    tuned = br.P.tuned()
    with tempfile.TemporaryDirectory() as tmp:
        proj = Path(tmp) / "proj"
        shutil.copytree(bp.DEFAULT_PROJECT, proj, ignore=shutil.ignore_patterns("Debug", "Release"))
        hal = Path(tmp) / "hal"
        ld = proj / "STM32F103RBTX_FLASH.ld"             # the harness's golden vectors can overflow 128 K (not the model)
        ld.write_text(ld.read_text().replace("LENGTH = 128K", "LENGTH = 1024K"))
        for i, c in enumerate(cells, 1):
            if a.only and str(i) not in a.only.split(","):
                continue
            H, D, bm = c["H"], cv.DATA_D[c["data"]], c["bm"]
            bp.install(proj, bm, H, D, "none", config=c["config"], geo=c["geo"], stream_ch=bm in bp.STREAM_CH,
                       data=c["data"] if bm in bp.CONFIG_MODELS else None)
            port_files = {p.name for p in (proj / "Core" / "Src").glob("*.c")
                          if p.name not in ("main.c", "syscalls.c", "sysmem.c", "system_stm32f1xx.c")
                          and not p.name.startswith("stm32f1xx")}
            elf, _, _ = br.build(proj, hal)
            r = emulate(elf, proj, bm, H, D, port_files)
            t = torch_ops(c["tier"], c["model"], c["data"], H, tuned[f"{c['tier']}|{c['model']}|{c['data']}|{H}"]["knobs"])
            row = dict(tier=c["tier"], model=c["model"], trial=c["trial"], data=c["data"], H=H, D=D,
                       port="final", **r, **t_cols(t))
            row["ratio"] = round(r["C_ops"] / (row["T_ops"] + row["T_fft"]), 3)
            print(f"{i:3d}. {c['tier']:<4} {c['model']:<14} {c['data']:<12} H={H:<3} C {r['C_ops']:>11,}  "
                  f"torch {row['T_ops']:>11,} + fft {row['T_fft']:>9,}  ratio {row['ratio']}  stack {r['stack_B']} B  "
                  f"maxabs {r['golden_maxabs']}  ({r['emu_s']} s)", flush=True)
            if a.out:
                new = not a.out.exists()
                with a.out.open("a", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
                    if new:
                        w.writeheader()
                    w.writerow(row)


if __name__ == "__main__":
    main()
