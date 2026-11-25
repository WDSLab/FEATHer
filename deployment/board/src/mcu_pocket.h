/* pocketfft's transforms with const plans (tables from export/pocket_plan.py).  c = the data, transformed in
 * place (real: n floats, halfcomplex r0 r1 i1 r2 i2 ...; complex: 2 n floats), ch = work of the same size.
 * A plan is made for one direction and names the pass of each factor. */
#ifndef MCU_POCKET_H
#define MCU_POCKET_H
#include <stddef.h>

typedef struct cmplx {
  float r,i;
} cmplx;

typedef void pk_cpass(size_t ido, size_t l1, const cmplx * restrict cc, cmplx * restrict ch,
                      const cmplx * restrict wa);
typedef void pk_cpass_s(size_t ido, size_t l1, const cmplx * restrict cc, cmplx * restrict ch,
                        const cmplx * restrict wa, const int sign);
typedef void pk_rpass(size_t ido, size_t l1, const float * restrict cc, float * restrict ch,
                      const float * restrict wa);
typedef void pk_rpass_g(size_t ido, size_t ip, size_t l1, float * restrict cc, float * restrict ch,
                        const float * restrict wa, const float * restrict csarr);
pk_cpass pk_pass2b, pk_pass2f, pk_pass3b, pk_pass3f, pk_pass4b, pk_pass4f, pk_pass5b, pk_pass5f, pk_pass8b, pk_pass8f;
pk_cpass_s pk_pass7, pk_pass11;
pk_rpass pk_radf2, pk_radf3, pk_radf4, pk_radf5, pk_radb2, pk_radb3, pk_radb4, pk_radb5;
pk_rpass_g pk_radfg, pk_radbg;

#define NFCT 8
typedef struct cfftp_fctdata
  {
  size_t fct;
  const cmplx *tw, *tws;
  pk_cpass *fn;
  pk_cpass_s *fns;
  } cfftp_fctdata;

typedef struct cfftp_plan_i
  {
  size_t length, nfct;
  int sign;                       /* -1 forward, +1 backward */
  cfftp_fctdata fct[NFCT];
  } cfftp_plan_i;

typedef struct rfftp_fctdata
  {
  size_t fct;
  const float *tw, *tws;
  pk_rpass *fn;
  pk_rpass_g *fng;
  } rfftp_fctdata;

typedef struct rfftp_plan_i
  {
  size_t length, nfct;
  rfftp_fctdata fct[NFCT];
  } rfftp_plan_i;

void pk_rfft_forward(const rfftp_plan_i *plan, float c[], float ch[], float fct);    /* a forward plan */
void pk_rfft_backward(const rfftp_plan_i *plan, float c[], float ch[], float fct);   /* a backward plan */
void pk_cfft(const cfftp_plan_i *plan, float c[], float ch[], float fct);

#endif
