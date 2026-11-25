/* PyTorch 2.5.1 DEFAULT-build CPU kernels, step by step (see mcu_pt.h).  Sources: SumKernel.cpp (cascade_sum),
 * TensorIterator.cpp (reorder / coalesce), SharedReduceOps.h (WelfordOps), moments_utils.h + layer_norm_kernel.cpp,
 * SoftMaxKernel.cpp + functional_base.h (reduce_all, map), ReduceOpsKernel.cpp (norm, last dim). */
#include <math.h>
#include "mcu_pt.h"
#include "mcu_emit.h"

#define V 8
typedef struct { float v[V]; } vec8;

static int ceil_log2(int x)
{
    int b = 0;
    if (x <= 2) return 1;
    for (unsigned u = (unsigned)(x - 1); u; u >>= 1) ++b;
    return b;
}

/* ------------------------------------------------------------------ cascade_sum
 * Every output is reduced on its own, so a caller may run only the outputs [lo, hi) of the whole tensor's loop
 * (a channel of a batch): the others are skipped, the arithmetic of each output stays the same.  x[g - xoff] is
 * element g of the whole tensor; out[o - lo] is output o. */
typedef struct {
    const float *x;
    int xoff, lo, hi;
    float *out, z;
} sum_ctx;

static void vload(vec8 *a, const float *p)
{
    for (int i = 0; i < V; ++i) a->v[i] = p[i];
}

#define ON(c, o) ((o) >= (c)->lo && (o) < (c)->hi)
#define X(c, g) ((c)->x[(g) - (c)->xoff])

static void store(const sum_ctx *c, int o, float v)          /* CastStoreAccumulate: *ptr += value */
{
    c->out[o - c->lo] += v;
}

/* multi_row_sum, 4 rows; lane mask m[k] (bit l = lane l of row k; a scalar row uses bit 0) */
static void mrow(const sum_ctx *c, int base, int rs, int cs, int size, int lanes, const unsigned m[4], float r[4][V])
{
    int lp = ceil_log2(size) / 4;
    if (lp < 4) lp = 4;
    const int step = 1 << lp, mask0 = step - 1;
    float acc[4][4][V];
    for (int j = 0; j < 4; ++j)
        for (int k = 0; k < 4; ++k)
            for (int l = 0; l < lanes; ++l) acc[j][k][l] = c->z;
    int i = 0;
    while (i + step <= size) {
        for (int s = 0; s < step; ++s, ++i)
            for (int k = 0; k < 4; ++k)
                for (int l = 0; l < lanes; ++l)
                    if (m[k] >> l & 1u) acc[0][k][l] += X(c, base + i * rs + k * cs + l);
        for (int j = 1; j < 4; ++j) {
            for (int k = 0; k < 4; ++k)
                for (int l = 0; l < lanes; ++l)
                    if (m[k] >> l & 1u) {
                        acc[j][k][l] += acc[j - 1][k][l];
                        acc[j - 1][k][l] = c->z;
                    }
            if (i & (mask0 << (j * lp))) break;
        }
    }
    for (; i < size; ++i)
        for (int k = 0; k < 4; ++k)
            for (int l = 0; l < lanes; ++l)
                if (m[k] >> l & 1u) acc[0][k][l] += X(c, base + i * rs + k * cs + l);
    for (int j = 1; j < 4; ++j)
        for (int k = 0; k < 4; ++k)
            for (int l = 0; l < lanes; ++l)
                if (m[k] >> l & 1u) acc[0][k][l] += acc[j][k][l];
    for (int k = 0; k < 4; ++k)
        for (int l = 0; l < lanes; ++l) r[k][l] = acc[0][k][l];
}

/* row_sum: ILP 4 over the row, then the partial sums */
static void row(const sum_ctx *c, int base, int st, int size, int lanes, unsigned m, float r[V])
{
    const unsigned mm[4] = {m, m, m, m};
    float p[4][V];
    mrow(c, base, 4 * st, st, size / 4, lanes, mm, p);
    for (int i = size / 4 * 4; i < size; ++i)
        for (int l = 0; l < lanes; ++l)
            if (m >> l & 1u) p[0][l] += X(c, base + i * st + l);
    for (int k = 1; k < 4; ++k)
        for (int l = 0; l < lanes; ++l)
            if (m >> l & 1u) p[0][l] += p[k][l];
    for (int l = 0; l < lanes; ++l) r[l] = p[0][l];
}

static unsigned lanes_on(const sum_ctx *c, int o, int ost, int n)
{
    unsigned m = 0;
    for (int l = 0; l < n; ++l)
        if (ON(c, o + l * ost)) m |= 1u << l;
    return m;
}

/* one call of the 2-d loop; o / xo = element offsets, s = {out0, in0, out1, in1} */
static void sum_loop(const sum_ctx *c, int o, int xo, const int s[4], int size0, int size1)
{
    int in0 = s[1], in1 = s[3], ost = s[2];
    float r[4][V];
    if (s[0] != 0 && s[2] == 0) {
        in0 = s[3]; in1 = s[1]; ost = s[0];
        int t = size0; size0 = size1; size1 = t;
    }
    if (in0 == 1 && size0 >= V) {                                   /* vectorized_inner_sum */
        const int vs = size0 / V;
        for (int j = 0; j < size1; ++j) {
            if (!ON(c, o + j * ost)) continue;
            const int b = xo + j * in1;
            row(c, b, V, vs, V, 0xFFu, r[0]);
            float fa = c->z;
            for (int k = vs * V; k < size0; ++k) fa += X(c, b + k);
            for (int k = 0; k < V; ++k) fa += r[0][k];
            store(c, o + j * ost, fa);
        }
    } else if (in1 == 1 && size1 >= V) {                            /* vectorized_outer_sum */
        int j = 0;
        for (; j + 4 * V <= size1; j += 4 * V) {
            unsigned m[4];
            for (int i = 0; i < 4; ++i) m[i] = lanes_on(c, o + (j + i * V) * ost, ost, V);
            if (!(m[0] | m[1] | m[2] | m[3])) continue;
            mrow(c, xo + j, in0, V, size0, V, m, r);
            for (int i = 0; i < 4; ++i)
                for (int k = 0; k < V; ++k)
                    if (m[i] >> k & 1u) store(c, o + (j + i * V + k) * ost, r[i][k]);
        }
        for (; j + V <= size1; j += V) {
            const unsigned m = lanes_on(c, o + j * ost, ost, V);
            if (!m) continue;
            row(c, xo + j, in0, size0, V, m, r[0]);
            for (int k = 0; k < V; ++k)
                if (m >> k & 1u) store(c, o + (j + k) * ost, r[0][k]);
        }
        for (; j < size1; ++j)
            if (ON(c, o + j * ost)) {
                row(c, xo + j, in0, size0, 1, 1u, r[0]);
                store(c, o + j * ost, r[0][0]);
            }
    } else if (in0 < in1) {                                         /* scalar_inner_sum */
        for (int j = 0; j < size1; ++j)
            if (ON(c, o + j * ost)) {
                row(c, xo + j * in1, in0, size0, 1, 1u, r[0]);
                store(c, o + j * ost, r[0][0]);
            }
    } else {                                                        /* scalar_outer_sum */
        int j = 0;
        for (; j + 3 < size1; j += 4) {
            unsigned m[4];
            for (int k = 0; k < 4; ++k) m[k] = ON(c, o + (j + k) * ost) ? 1u : 0u;
            if (!(m[0] | m[1] | m[2] | m[3])) continue;
            mrow(c, xo + j * in1, in0, in1, size0, 1, m, r);
            for (int k = 0; k < 4; ++k)
                if (m[k]) store(c, o + (j + k) * ost, r[k][0]);
        }
        for (; j < size1; ++j)
            if (ON(c, o + j * ost)) {
                row(c, xo + j * in1, in0, size0, 1, 1u, r[0]);
                store(c, o + j * ost, r[0][0]);
            }
    }
}

/* TensorIterator: output strides (keepdim, contiguous, 0 on reduced dims), reduced dims first, by stride, coalesced */
static int geometry(int nd, const int *shape, const int *stride, unsigned red, int *sh, int st[2][PT_MAXD])
{
    int so[PT_MAXD], perm[PT_MAXD], acc = 1;
    for (int d = nd - 1; d >= 0; --d) {
        so[d] = (red >> d & 1u) ? 0 : acc;
        acc *= (red >> d & 1u) ? 1 : shape[d];
    }
    for (int d = 0; d < nd; ++d) perm[d] = nd - 1 - d;
    for (int i = 1; i < nd; ++i) {
        int d1 = i;
        for (int d0 = i - 1; d0 >= 0; --d0) {
            const int a = perm[d0], b = perm[d1];
            int c = 0;
            if ((so[a] == 0) != (so[b] == 0)) c = so[b] == 0 ? 1 : -1;
            else if (so[a] != 0 && so[b] != 0 && so[a] != so[b]) c = so[a] < so[b] ? -1 : 1;
            else if (so[a] != 0 && so[b] != 0 && shape[a] > shape[b]) c = 1;
            else if (stride[a] != 0 && stride[b] != 0) {
                if (stride[a] < stride[b]) c = -1;
                else if (stride[a] > stride[b]) c = 1;
                else if (shape[a] > shape[b]) c = 1;
            }
            if (c > 0) {
                perm[d0] = b; perm[d1] = a;
                d1 = d0;
            } else if (c < 0) {
                break;
            }
        }
    }
    for (int d = 0; d < nd; ++d) {
        sh[d] = shape[perm[d]];
        st[0][d] = so[perm[d]];
        st[1][d] = stride[perm[d]];
    }
    int prev = 0;
    for (int d = 1; d < nd; ++d) {
        const int ok = sh[prev] == 1 || sh[d] == 1 ||
                       (sh[prev] * st[0][prev] == st[0][d] && sh[prev] * st[1][prev] == st[1][d]);
        if (ok) {
            if (sh[prev] == 1) { st[0][prev] = st[0][d]; st[1][prev] = st[1][d]; }
            sh[prev] *= sh[d];
        } else {
            ++prev;
            if (prev != d) { st[0][prev] = st[0][d]; st[1][prev] = st[1][d]; sh[prev] = sh[d]; }
        }
    }
    return nd > 0 ? prev + 1 : 0;
}

void pt_sum_part(const float *x, int xoff, float *out, int nd, const int *shape, const int *stride, unsigned red,
                 int lo, int hi)
{
    int sh[PT_MAXD], st[2][PT_MAXD];
    const sum_ctx c = {x, xoff, lo, hi, out, mcu_rt(0.0f)};
    for (int i = lo; i < hi; ++i) out[i - lo] = c.z;                /* output_base().fill_(0) */
    const int n = geometry(nd, shape, stride, red, sh, st);
    if (n <= 1) {
        const int s[4] = {st[0][0], st[1][0], 0, 0};
        sum_loop(&c, 0, 0, s, sh[0], 1);
        return;
    }
    int idx[PT_MAXD] = {0};
    const int s[4] = {st[0][0], st[1][0], st[0][1], st[1][1]};
    for (;;) {
        int o = 0, xo = 0;
        for (int k = 2; k < n; ++k) { o += idx[k] * st[0][k]; xo += idx[k] * st[1][k]; }
        sum_loop(&c, o, xo, s, sh[0], sh[1]);
        int k = 2;
        for (; k < n; ++k) {
            if (++idx[k] < sh[k]) break;
            idx[k] = 0;
        }
        if (k == n) return;
    }
}

void pt_mean_part(const float *x, int xoff, float *out, int nd, const int *shape, const int *stride, unsigned red,
                  int lo, int hi)
{
    int nred = 1;
    for (int d = 0; d < nd; ++d)
        if (red >> d & 1u) nred *= shape[d];
    pt_sum_part(x, xoff, out, nd, shape, stride, red, lo, hi);
    const float nf = mcu_rt((float)nred);
    for (int i = 0; i < hi - lo; ++i) out[i] = out[i] / nf;        /* sum_out(...).div_(dim_prod) */
}

static int n_out(int nd, const int *shape, unsigned red)
{
    int n = 1;
    for (int d = 0; d < nd; ++d)
        if (!(red >> d & 1u)) n *= shape[d];
    return n;
}

void pt_sum(const float *x, float *out, int nd, const int *shape, const int *stride, unsigned red)
{
    pt_sum_part(x, 0, out, nd, shape, stride, red, 0, n_out(nd, shape, red));
}

void pt_mean(const float *x, float *out, int nd, const int *shape, const int *stride, unsigned red)
{
    pt_mean_part(x, 0, out, nd, shape, stride, red, 0, n_out(nd, shape, red));
}

void pt_mean_t(const float *x, float *out, int l, int d)
{
    const int shape[3] = {1, l, d}, stride[3] = {l * d, d, 1};
    pt_mean(x, out, 3, shape, stride, 1u << 1);
}

/* ------------------------------------------------------------------ var: WelfordOps<float, double> */
void pt_var(const float *x, float *out, int rows, int rstep, int n, int stride, int correction)
{
    volatile double vc = (double)correction;
    const double corr = vc;
    for (int r = 0; r < rows; ++r) {
        const float *p = x + r * rstep;
        double mean = mcu_rt(0.0f), m2 = mean, nf = mean;
        for (long long i = 1; i <= n; ++i, p += stride) {
            nf = (double)i;
            const double delta = *p - mean;
            mean = mean + delta / nf;
            m2 = m2 + delta * (*p - mean);
        }
        mcu_drop((float)mean);                                      /* project() casts the mean too */
        const double div = nf > corr ? nf - corr : 0.0;
        out[r] = (float)(m2 / div);
    }
}

/* ------------------------------------------------------------------ layer norm: RowwiseMoments + affine (map3) */
static const float PT_CINV[16] = {1.0f / 1, 1.0f / 2, 1.0f / 3, 1.0f / 4, 1.0f / 5, 1.0f / 6, 1.0f / 7, 1.0f / 8,
                                  1.0f / 9, 1.0f / 10, 1.0f / 11, 1.0f / 12, 1.0f / 13, 1.0f / 14, 1.0f / 15, 1.0f / 16};

static void add_moments_vec(int m0_add, const vec8 *m1a, const vec8 *m2a, int *m0, vec8 *m1, vec8 *m2, float z)
{
    const int n = *m0 + m0_add;
    const float c = n == 0 ? z : (float)m0_add / (float)n;
    const float m0f = (float)*m0;
    for (int l = 0; l < V; ++l) {
        const float delta = m1a->v[l] - m1->v[l];
        m1->v[l] += c * delta;
        m2->v[l] += m2a->v[l] + delta * delta * c * m0f;
    }
    *m0 = n;
}

#define PT_DEPTH 8
static void moments(const float *X, int N, float z, float *mean, float *var)
{
    const int n = N / V, m = (n + 15) / 16, depth = ceil_log2(m);
    int s0[PT_DEPTH];
    vec8 s1[PT_DEPTH], s2[PT_DEPTH];
    for (int d = 0; d < depth; ++d) {
        s0[d] = 0;
        for (int l = 0; l < V; ++l) s1[d].v[l] = s2[d].v[l] = z;
    }
    for (int i = 0; i < m; ++i) {
        const int m0 = n - i * 16 < 16 ? n - i * 16 : 16;
        vec8 m1v, m2v;
        for (int l = 0; l < V; ++l) m1v.v[l] = m2v.v[l] = z;
        for (int j = 0; j < m0; ++j) {
            const float *xp = X + (i * 16 + j) * V;
            for (int l = 0; l < V; ++l) {
                const float d = xp[l] - m1v.v[l];
                m1v.v[l] += d * PT_CINV[j];
                m2v.v[l] += d * (xp[l] - m1v.v[l]);
            }
        }
        add_moments_vec(m0, &m1v, &m2v, &s0[0], &s1[0], &s2[0], z);
        int mask = i + 1;
        for (int j = 1; j < depth && (mask & 1) == 0; ++j) {
            add_moments_vec(s0[j - 1], &s1[j - 1], &s2[j - 1], &s0[j], &s1[j], &s2[j], z);
            s0[j - 1] = 0;
            for (int l = 0; l < V; ++l) s1[j - 1].v[l] = s2[j - 1].v[l] = z;
            mask >>= 1;
        }
    }
    for (int d = 1; d < depth; ++d) add_moments_vec(s0[d], &s1[d], &s2[d], &s0[0], &s1[0], &s2[0], z);
    int m0 = 0;
    float m1 = z, m2 = z;
    for (int i = n * V; i < N; ++i) {
        const float delta = X[i] - m1;
        ++m0;
        m1 += delta / (float)m0;
        m2 += delta * (X[i] - m1);
    }
    for (int l = 0; l < V; ++l) {                                   /* AddMoments per lane, m0_add = n */
        const int nn = m0 + n;
        const float c = nn == 0 ? z : (float)n / (float)nn;
        const float delta = s1[0].v[l] - m1;
        m1 += c * delta;
        m2 += s2[0].v[l] + delta * delta * c * (float)m0;
        m0 = nn;
    }
    *mean = m1;
    *var = m2 / (float)N;
}

void pt_layer_norm(const float *x, float *y, int rows, int n, const float *g, const float *b, float eps)
{
    const float z = mcu_rt(0.0f);
    for (int r = 0; r < rows; ++r, x += n, y += n) {
        float mean, var;
        moments(x, n, z, &mean, &var);
        const float rstd = 1.0f / sqrtf(var + eps);
        const float bias = -mean;
        for (int d = 0; d < n; d += V) {
            float lx[V], lg[V], lb[V];
            for (int l = 0; l < V; ++l) {
                const int ok = d + l < n;
                lx[l] = ok ? x[d + l] : z;
                lg[l] = ok ? g[d + l] : z;
                lb[l] = ok ? b[d + l] : z;
            }
            for (int l = 0; l < V; ++l) {
                const float v = (lx[l] + bias) * rstd * lg[l] + lb[l];
                if (d + l < n) y[d + l] = v;
                else mcu_drop(v);
            }
        }
    }
}

/* ------------------------------------------------------------------ softmax, last dim */
static void vfun(int add, vec8 *a, const vec8 *b)
{
    for (int l = 0; l < V; ++l) {
        if (add) {
            a->v[l] = a->v[l] + b->v[l];
        } else {
            float c = a->v[l] > b->v[l] ? a->v[l] : b->v[l];
            if (a->v[l] != a->v[l]) c = a->v[l];                /* _isnan(a): a float compare */
            a->v[l] = c;
        }
    }
}

static float reduce_all(int add, const float *d, int size, float z)
{
    vec8 acc;
    int n;
    if (size < V) {
        for (int l = 0; l < V; ++l) acc.v[l] = l < size ? d[l] : z;
        n = size;
    } else {
        vload(&acc, d);
        int k = V;
        for (; k < size - size % V; k += V) {
            vec8 t;
            vload(&t, d + k);
            vfun(add, &acc, &t);
        }
        if (size - k > 0) {
            vec8 t, r = acc;
            for (int l = 0; l < V; ++l) t.v[l] = l < size - k ? d[k + l] : z;
            vfun(add, &r, &t);
            for (int l = 0; l < size - k; ++l) acc.v[l] = r.v[l];
        }
        n = V;
    }
    vec8 arr = acc;
    for (int i = 1; i < n; ++i) {
        vec8 nx;
        for (int l = 0; l < V; ++l) nx.v[l] = z;
        nx.v[0] = arr.v[i];
        vfun(add, &acc, &nx);
    }
    return acc.v[0];
}

void pt_softmax(const float *x, float *y, int rows, int k)
{
    const float z = mcu_rt(0.0f);
    for (int r = 0; r < rows; ++r, x += k, y += k) {
        const float mx = reduce_all(0, x, k, z);
        for (int d = 0; d < k; d += V)
            for (int l = 0; l < V; ++l) {
                const float v = expf((d + l < k ? x[d + l] : z) - mx);
                if (d + l < k) y[d + l] = v;
                else mcu_drop(v);
            }
        const float s = 1 / reduce_all(1, y, k, z);
        for (int d = 0; d < k; d += V)
            for (int l = 0; l < V; ++l) {
                const float v = (d + l < k ? y[d + l] : z) * s;
                if (d + l < k) y[d + l] = v;
                else mcu_drop(v);
            }
    }
}

/* ------------------------------------------------------------------ vector norm, ord 2, last dim */
float pt_norm2(const float *x, int n)
{
    const float z = mcu_rt(0.0f);
    float acc[V];
    for (int l = 0; l < V; ++l) acc[l] = z;
    int d = 0;
    for (; d < n - n % V; d += V)
        for (int l = 0; l < V; ++l) acc[l] += x[d + l] * x[d + l];
    for (int j = 1; j < V; ++j) acc[0] = acc[0] + acc[j];
    for (; d < n; ++d) acc[0] += x[d] * x[d];
    return sqrtf(acc[0]);
}
