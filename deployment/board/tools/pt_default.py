# -*- coding: utf-8 -*-
"""PyTorch 2.5.1 DEFAULT-build CPU kernels (ATEN_CPU_CAPABILITY=default, one thread), step by step in float32 / float64,
counting every arithmetic operation they execute (Vectorized<float> = 8 lanes, each lane counted).
Sources: PyTorch 2.5.1 aten/src/ATen/native/cpu/.  Checked bit-exact against torch by test_pt_default.py."""
from collections import Counter

import numpy as np

F, D, V = np.float32, np.float64, 8


class Cnt(Counter):
    def op(self, kind, n=1):
        self[kind] += n


# ------------------------------------------------------------------ TensorIterator (reduction, output first)
def _geometry(shape, in_st, red):
    """Permuted + coalesced shape and byte strides [out, in] as TensorIterator builds them."""
    n = len(shape)
    out_shape = [1 if d in red else s for d, s in enumerate(shape)]
    cs, acc = [0] * n, 1
    for d in reversed(range(n)):
        cs[d], acc = acc, acc * out_shape[d]
    st = [[0 if d in red else 4 * cs[d] for d in range(n)], [4 * s for s in in_st]]
    perm = list(reversed(range(n)))

    def swap(d0, d1):
        for k in (0, 1):
            s0, s1 = st[k][d0], st[k][d1]
            if k == 0 and (s0 == 0) != (s1 == 0):
                return 1 if s1 == 0 else -1
            if s0 == 0 or s1 == 0:
                continue
            if s0 < s1:
                return -1
            if s0 > s1:
                return 1
            if shape[d0] > shape[d1]:
                return 1
        return 0

    if n > 1:
        for i in range(1, n):
            d1 = i
            for d0 in range(i - 1, -1, -1):
                c = swap(perm[d0], perm[d1])
                if c > 0:
                    perm[d0], perm[d1] = perm[d1], perm[d0]
                    d1 = d0
                elif c < 0:
                    break
    sh = [shape[p] for p in perm]
    st = [[s[p] for p in perm] for s in st]
    if n > 1:
        prev = 0
        for d in range(1, n):
            ok = sh[prev] == 1 or sh[d] == 1 or all(sh[prev] * s[prev] == s[d] for s in st)
            if ok:
                if sh[prev] == 1:
                    for s in st:
                        s[prev] = s[d]
                sh[prev] *= sh[d]
            else:
                prev += 1
                if prev != d:
                    for s in st:
                        s[prev] = s[d]
                    sh[prev] = sh[d]
        sh, st = sh[:prev + 1], [s[:prev + 1] for s in st]
    return sh, st


def _for_each(sh, st, loop):
    """serial_for_each over the full range: loop(out_off, in_off, [s_out0, s_in0, s_out1, s_in1], size0, size1)."""
    if len(sh) <= 1:
        return loop(0, 0, [st[0][0], st[1][0], 0, 0], sh[0], 1)
    outer = sh[2:]
    idx = [0] * len(outer)
    while True:
        o = sum(i * st[0][2 + k] for k, i in enumerate(idx))
        x = sum(i * st[1][2 + k] for k, i in enumerate(idx))
        loop(o, x, [st[0][0], st[1][0], st[0][1], st[1][1]], sh[0], sh[1])
        k = 0
        while k < len(outer):
            idx[k] += 1
            if idx[k] < outer[k]:
                break
            idx[k] = 0
            k += 1
        if k == len(outer):
            return


# ------------------------------------------------------------------ sum / mean (SumKernel.cpp cascade_sum)
class _Sum:
    def __init__(self, buf, out, c):
        self.buf, self.out, self.c = buf, out, c

    def ld(self, off, vec):                                  # byte offset -> scalar or 8 lanes
        i = off // 4
        return self.buf[i:i + V].copy() if vec else self.buf[i]

    def add(self, a, b):
        self.c.op("add", V if isinstance(a, np.ndarray) else 1)
        return (a + b).astype(F) if isinstance(a, np.ndarray) else F(a + b)

    def multi_row(self, base, row_st, col_st, size, nrows, vec):
        zero = (lambda: np.zeros(V, F)) if vec else (lambda: F(0))
        lp = max(4, _ceil_log2(size) // 4)
        step, mask0 = 1 << lp, (1 << lp) - 1
        acc = [[zero() for _ in range(nrows)] for _ in range(4)]
        i = 0
        while i + step <= size:
            for _ in range(step):
                for k in range(nrows):
                    acc[0][k] = self.add(acc[0][k], self.ld(base + i * row_st + k * col_st, vec))
                i += 1
            for j in range(1, 4):
                for k in range(nrows):
                    acc[j][k] = self.add(acc[j][k], acc[j - 1][k])
                    acc[j - 1][k] = zero()
                if i & (mask0 << (j * lp)):
                    break
        while i < size:
            for k in range(nrows):
                acc[0][k] = self.add(acc[0][k], self.ld(base + i * row_st + k * col_st, vec))
            i += 1
        for j in range(1, 4):
            for k in range(nrows):
                acc[0][k] = self.add(acc[0][k], acc[j][k])
        return acc[0]

    def row(self, base, st, size, vec):
        p = self.multi_row(base, st * 4, st, size // 4, 4, vec)
        for i in range(size // 4 * 4, size):
            p[0] = self.add(p[0], self.ld(base + i * st, vec))
        for k in range(1, 4):
            p[0] = self.add(p[0], p[k])
        return p[0]

    def store(self, off, v):                                # CastStoreAccumulate: *ptr += value
        self.c.op("add")
        self.out[off // 4] = F(self.out[off // 4] + v)

    def loop(self, o, x, s, size0, size1):
        ins, outs = [s[1], s[3]], [s[0], s[2]]
        if outs[0] != 0 and outs[1] == 0:
            ins.reverse(), outs.reverse()
            size0, size1 = size1, size0
        if outs[0] != 0 and outs[1] != 0:                    # not a true reduction: out = out + in
            raise NotImplementedError("cascade_sum special case")
        ost = outs[1]
        if ins[0] == 4 and size0 >= V:                       # vectorized_inner_sum
            vs = size0 // V
            for j in range(size1):
                r = x + j * ins[1]
                va = self.row(r, 4 * V, vs, True)
                fa = F(0)
                for k in range(vs * V, size0):
                    fa = self.add(fa, self.ld(r + 4 * k, False))
                for k in range(V):
                    fa = self.add(fa, va[k])
                self.store(o + j * ost, fa)
        elif ins[1] == 4 and size1 >= V:                     # vectorized_outer_sum
            j = 0
            while j + 4 * V <= size1:
                sums = self.multi_row(x + 4 * j, ins[0], 4 * V, size0, 4, True)
                for i in range(4):
                    for k in range(V):
                        self.store(o + (j + i * V) * ost + k * ost, sums[i][k])
                j += 4 * V
            while j + V <= size1:
                sums = self.row(x + 4 * j, ins[0], size0, True)
                for k in range(V):
                    self.store(o + j * ost + k * ost, sums[k])
                j += V
            while j < size1:
                self.store(o + j * ost, self.row(x + 4 * j, ins[0], size0, False))
                j += 1
        elif ins[0] < ins[1]:                                # scalar_inner_sum
            for j in range(size1):
                self.store(o + j * ost, self.row(x + j * ins[1], ins[0], size0, False))
        else:                                                # scalar_outer_sum
            j = 0
            while j + 3 < size1:
                sums = self.multi_row(x + j * ins[1], ins[0], ins[1], size0, 4, False)
                for k in range(4):
                    self.store(o + j * ost + k * ost, sums[k])
                j += 4
            while j < size1:
                self.store(o + j * ost, self.row(x + j * ins[1], ins[0], size0, False))
                j += 1


def _ceil_log2(x):
    return 1 if x <= 2 else int(x - 1).bit_length()


def _view(t):
    """(flat storage as float32, shape, element strides) of a torch tensor, offset folded into the storage."""
    import torch
    st = t.untyped_storage()
    buf = torch.empty(0).set_(st, 0, (st.nbytes() // 4,), (1,)).numpy()[t.storage_offset():].astype(F)
    return np.concatenate([buf, np.zeros(V, F)]), list(t.shape), list(t.stride())


def _out_size(shape, red):
    return int(np.prod([1 if d in red else s for d, s in enumerate(shape)]))


def sum_(t, dims, c):
    buf, shape, st = _view(t)
    red = {d % len(shape) for d in dims} if dims else set(range(len(shape)))
    out = np.zeros(_out_size(shape, red), F)
    sh, sts = _geometry(shape, st, red)
    k = _Sum(buf, out, c)
    _for_each(sh, sts, k.loop)
    return out


def mean(t, dims, c):
    out = sum_(t, dims, c)
    shape = list(t.shape)
    red = {d % len(shape) for d in dims} if dims else set(range(len(shape)))
    n = F(int(np.prod([shape[d] for d in red])))
    c.op("div", out.size)
    return (out / n).astype(F)


# ------------------------------------------------------------------ var (WelfordOps in double, binary_kernel_reduce)
def var(t, dims, correction, c):
    x = t.detach().numpy()
    red = sorted(d % x.ndim for d in dims)
    keep = [d for d in range(x.ndim) if d not in red]
    xs = np.transpose(x, keep + red).reshape(int(np.prod([x.shape[d] for d in keep])), -1)
    if len(red) > 1:
        raise NotImplementedError("var over several dims: element order not modelled")
    out = np.empty(xs.shape[0], F)
    for r in range(xs.shape[0]):
        mean_, m2, nf = D(0), D(0), D(0)
        for n, v in enumerate(xs[r], 1):
            nf = D(n)
            delta = D(v) - mean_
            mean_ = mean_ + delta / nf
            m2 = m2 + delta * (D(v) - mean_)
        c.op("dadd", 4 * xs.shape[1]), c.op("ddiv", xs.shape[1]), c.op("dmul", xs.shape[1])
        c.op("dcmp"), c.op("dadd"), c.op("ddiv")
        div = nf - D(correction) if nf > D(correction) else D(0)
        out[r] = F(m2 / div)
    return out


# ------------------------------------------------------------------ layer norm (layer_norm_kernel.cpp + moments_utils.h)
def _add_moments_vec(m0_add, m1a, m2a, st, c):
    """AddMomentsVec into st = [m0, m1 vec, m2 vec]."""
    n = st[0] + m0_add
    cc = F(0) if n == 0 else F(F(m0_add) / F(n))
    c.op("div", 0 if n == 0 else 1)
    delta = (m1a - st[1]).astype(F)
    st[1] = (st[1] + (cc * delta).astype(F)).astype(F)
    st[2] = (st[2] + (m2a + (((delta * delta).astype(F) * cc).astype(F) * F(st[0])).astype(F)).astype(F)).astype(F)
    c.op("add", 4 * V), c.op("mul", 4 * V)
    st[0] = n


def _moments(X, c):
    """RowwiseMomentsImpl<float> (Welford + cascade, ddof 0): (mean, var)."""
    N = len(X)
    n = N // V
    m = -(-n // 16)
    depth = _ceil_log2(m)
    stk = [[0, np.zeros(V, F), np.zeros(V, F)] for _ in range(depth)]
    for i in range(m):
        m0 = min(16, n - i * 16)
        m1v, m2v = np.zeros(V, F), np.zeros(V, F)
        for j in range(m0):
            x = X[(i * 16 + j) * V:(i * 16 + j + 1) * V]
            d = (x - m1v).astype(F)
            m1v = (m1v + (d * F(F(1) / F(j + 1))).astype(F)).astype(F)      # c_vecs: a static table, built once
            m2v = (m2v + (d * (x - m1v).astype(F)).astype(F)).astype(F)
        c.op("add", 4 * V * m0), c.op("mul", 2 * V * m0)
        _add_moments_vec(m0, m1v, m2v, stk[0], c)
        mask = i + 1
        j = 1
        while j < depth and (mask & 1) == 0:
            _add_moments_vec(stk[j - 1][0], stk[j - 1][1], stk[j - 1][2], stk[j], c)
            stk[j - 1] = [0, np.zeros(V, F), np.zeros(V, F)]
            mask >>= 1
            j += 1
    for i in range(1, depth):
        _add_moments_vec(stk[i][0], stk[i][1], stk[i][2], stk[0], c)
    m1a, m2a = stk[0][1], stk[0][2]
    m0, m1, m2 = 0, F(0), F(0)
    for v in X[n * V:]:
        delta = F(v - m1)
        m0 += 1
        m1 = F(m1 + F(delta / F(m0)))
        m2 = F(m2 + F(delta * F(v - m1)))
    c.op("add", 4 * (N - n * V)), c.op("div", N - n * V), c.op("mul", N - n * V)
    for k in range(V):                                        # AddMoments per lane, m0_add = n
        nn = m0 + n
        cc = F(0) if nn == 0 else F(F(n) / F(nn))
        c.op("div", 0 if nn == 0 else 1)
        delta = F(m1a[k] - m1)
        m1 = F(m1 + F(cc * delta))
        m2 = F(m2 + F(m2a[k] + F(F(F(delta * delta) * cc) * F(m0))))
        c.op("add", 4), c.op("mul", 4)
        m0 = nn
    c.op("div")
    return m1, F(m2 / F(N))


def layer_norm(t, w, b, eps, c):
    x = t.detach().numpy().astype(F)
    N = x.shape[-1]
    rows = x.reshape(-1, N)
    g, bb = w.detach().numpy().astype(F), b.detach().numpy().astype(F)
    out = np.empty_like(rows)
    pad = -(-N // V) * V
    for r, row in enumerate(rows):
        mean_, var_ = _moments(row, c)
        rstd = F(F(1) / F(np.sqrt(F(var_ + F(eps)))))
        c.op("div"), c.op("add"), c.op("libm")
        lx, lg, lb = (np.zeros(pad, F) for _ in range(3))
        lx[:N], lg[:N], lb[:N] = row, g, bb
        y = ((((lx + -mean_).astype(F) * rstd).astype(F) * lg).astype(F) + lb).astype(F)
        c.op("add", 2 * pad), c.op("mul", 2 * pad)
        out[r] = y[:N]
    return out.reshape(x.shape)


# ------------------------------------------------------------------ softmax, last dim (SoftMaxKernel.cpp _vec_softmax_lastdim)
def _reduce_all(fun, data, c, kind, per):
    size = len(data)
    if size < V:
        acc = np.zeros(V, F)
        acc[:size] = data
        n = size
    else:
        acc = data[:V].astype(F).copy()
        d = V
        while d < size - size % V:
            acc = fun(acc, data[d:d + V])
            c.op(kind, per * V)
            d += V
        if size - d > 0:
            nxt = np.zeros(V, F)
            nxt[:size - d] = data[d:]
            r = fun(acc, nxt)
            c.op(kind, per * V)
            acc = np.concatenate([r[:size - d], acc[size - d:]])
        n = V
    arr = acc.copy()                                          # acc_vec.store(acc_arr) once, before the steps
    for i in range(1, n):
        nxt = np.zeros(V, F)
        nxt[0] = arr[i]
        acc = fun(acc, nxt)
        c.op(kind, per * V)
    return acc[0]


def _vmax(a, b):
    r = np.where(a > b, a, b).astype(F)
    return np.where(np.isnan(a), a, r).astype(F)


def _exp(a):          # expf correctly rounded; torch's own expf differs in about 1 row of 100 (library, not order)
    return np.exp(a.astype(D)).astype(F)


def softmax(t, c):
    x = t.detach().numpy().astype(F)
    K = x.shape[-1]
    rows = x.reshape(-1, K)
    out = np.empty_like(rows)
    pad = -(-K // V) * V
    for r, row in enumerate(rows):
        mx = _reduce_all(_vmax, row, c, "cmp", 2)
        lanes = np.zeros(pad, F)
        lanes[:K] = row
        e = _exp((lanes - mx).astype(F))
        c.op("add", pad), c.op("libm", pad)
        s = _reduce_all(lambda a, b: (a + b).astype(F), e[:K], c, "add", 1)
        s = F(F(1) / s)
        c.op("div")
        out[r] = (e * s).astype(F)[:K]
        c.op("mul", pad)
    return out.reshape(x.shape)


# ------------------------------------------------------------------ avg_pool2d (contiguous kernel), upsample_linear1d, vector norm
def avg_pool(t, k, c):
    x = t.detach().contiguous().numpy().astype(F)
    W = x.shape[-1] // k
    w = x.reshape(-1, W, k)
    s = np.zeros(w.shape[:2], F)
    for j in range(k):
        s = (s + w[:, :, j]).astype(F)
    c.op("add", w.shape[0] * W * (k + 1)), c.op("div", w.shape[0] * W)
    return (F(0) + (s / F(k)).astype(F)).astype(F).reshape(*x.shape[:-1], W)


def upsample_linear(t, size, c):
    x = t.detach().numpy().astype(F)
    n_in = x.shape[-1]
    scale = F(F(n_in) / F(size))
    c.op("div")
    i0, i1, l0, l1 = [], [], [], []
    for d in range(size):
        src = F(F(scale * F(F(d) + F(0.5))) - F(0.5))
        src = F(0) if src < F(0) else src
        idx = min(int(np.floor(src)), n_in - 1)
        lam = F(min(max(F(src - F(idx)), F(0)), F(1)))
        i0.append(idx), i1.append(idx + (1 if idx < n_in - 1 else 0)), l1.append(lam), l0.append(F(F(1) - lam))
    c.op("add", 4 * size), c.op("mul", size), c.op("cmp", 3 * size), c.op("floor", size)
    rows = x.reshape(-1, n_in)
    out = (rows[:, i0] * np.array(l0, F)).astype(F) + (rows[:, i1] * np.array(l1, F)).astype(F)
    c.op("mul", 2 * out.size), c.op("add", out.size)
    return out.astype(F).reshape(*x.shape[:-1], size)


def vector_norm_last2(t, c):
    """ord 2 over the last two dims of a contiguous tensor (binary_kernel_reduce_lastdim path)."""
    x = t.detach().numpy().astype(F)
    if x.shape[-1] == 1 and x.shape[-2] == 1:                # is_reduce_over_1D_vector: abs only
        return np.abs(x).reshape(x.shape[:-2])
    rows = x.reshape(x.shape[0], -1)
    n = rows.shape[1]
    out = np.empty(rows.shape[0], F)
    for r, row in enumerate(rows):
        acc = np.zeros(V, F)
        d = 0
        while d < n - n % V:
            acc = (acc + (row[d:d + V] * row[d:d + V]).astype(F)).astype(F)
            d += V
        b0 = acc[0]
        for j in range(1, V):
            b0 = F(b0 + acc[j])
        for v in row[d:]:
            b0 = F(b0 + F(v * v))
        out[r] = F(np.sqrt(b0))
    nv = (n - n % V)
    c.op("mul", rows.shape[0] * (nv + n % V)), c.op("add", rows.shape[0] * (nv + V - 1 + n % V))
    c.op("libm", rows.shape[0])
    return out
