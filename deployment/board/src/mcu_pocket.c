/* The transform code of pocketfft (C, https://gitlab.mpcdf.mpg.de/mtr/pocketfft, float), trimmed:
 * radix passes as in the original; plans = const tables that name each factor's pass (so a build links only the
 * passes its lengths use), work buffer = an argument, complex radix 8 added from the C++ pocketfft.
 * Original notice:
 *
 * This file is part of pocketfft.
 * Licensed under a 3-clause BSD style license - see LICENSE_pocketfft.md
 * Copyright (C) 2004-2018 Max-Planck-Society, author Martin Reinecke
 */
#include <string.h>
#include "mcu_pocket.h"

#define SWAP(a,b,type) \
  do { type tmp_=(a); (a)=(b); (b)=tmp_; } while(0)

#ifdef __GNUC__
#define NOINLINE __attribute__((noinline))
#define WARN_UNUSED_RESULT __attribute__ ((warn_unused_result))
#else
#define NOINLINE
#define WARN_UNUSED_RESULT
#endif

#define PMC(a,b,c,d) { a.r=c.r+d.r; a.i=c.i+d.i; b.r=c.r-d.r; b.i=c.i-d.i; }
#define ADDC(a,b,c) { a.r=b.r+c.r; a.i=b.i+c.i; }
#define SCALEC(a,b) { a.r*=b; a.i*=b; }
#define ROT90(a) { float tmp_=a.r; a.r=-a.i; a.i=tmp_; }
#define ROTM90(a) { float tmp_=-a.r; a.r=a.i; a.i=tmp_; }
#define CH(a,b,c) ch[(a)+ido*((b)+l1*(c))]
#define CC(a,b,c) cc[(a)+ido*((b)+cdim*(c))]
#define WA(x,i) wa[(i)-1+(x)*(ido-1)]
/* a = b*c */
#define A_EQ_B_MUL_C(a,b,c) { a.r=b.r*c.r-b.i*c.i; a.i=b.r*c.i+b.i*c.r; }
/* a = conj(b)*c*/
#define A_EQ_CB_MUL_C(a,b,c) { a.r=b.r*c.r+b.i*c.i; a.i=b.r*c.i-b.i*c.r; }

#define PMSIGNC(a,b,c,d) { a.r=c.r+sign*d.r; a.i=c.i+sign*d.i; b.r=c.r-sign*d.r; b.i=c.i-sign*d.i; }
/* a = b*c */
#define MULPMSIGNC(a,b,c) { a.r=b.r*c.r-sign*b.i*c.i; a.i=b.r*c.i+sign*b.i*c.r; }
/* a *= b */
#define MULPMSIGNCEQ(a,b) { float xtmp=a.r; a.r=b.r*a.r-sign*b.i*a.i; a.i=b.r*a.i+sign*b.i*xtmp; }

NOINLINE void pk_pass2b (size_t ido, size_t l1, const cmplx * restrict cc,
  cmplx * restrict ch, const cmplx * restrict wa)
  {
  const size_t cdim=2;

  if (ido==1)
    for (size_t k=0; k<l1; ++k)
      PMC (CH(0,k,0),CH(0,k,1),CC(0,0,k),CC(0,1,k))
  else
    for (size_t k=0; k<l1; ++k)
      {
      PMC (CH(0,k,0),CH(0,k,1),CC(0,0,k),CC(0,1,k))
      for (size_t i=1; i<ido; ++i)
        {
        cmplx t;
        PMC (CH(i,k,0),t,CC(i,0,k),CC(i,1,k))
        A_EQ_B_MUL_C (CH(i,k,1),WA(0,i),t)
        }
      }
  }

NOINLINE void pk_pass2f (size_t ido, size_t l1, const cmplx * restrict cc,
  cmplx * restrict ch, const cmplx * restrict wa)
  {
  const size_t cdim=2;

  if (ido==1)
    for (size_t k=0; k<l1; ++k)
      PMC (CH(0,k,0),CH(0,k,1),CC(0,0,k),CC(0,1,k))
  else
    for (size_t k=0; k<l1; ++k)
      {
      PMC (CH(0,k,0),CH(0,k,1),CC(0,0,k),CC(0,1,k))
      for (size_t i=1; i<ido; ++i)
        {
        cmplx t;
        PMC (CH(i,k,0),t,CC(i,0,k),CC(i,1,k))
        A_EQ_CB_MUL_C (CH(i,k,1),WA(0,i),t)
        }
      }
  }

#define PREP3(idx) \
        cmplx t0 = CC(idx,0,k), t1, t2; \
        PMC (t1,t2,CC(idx,1,k),CC(idx,2,k)) \
        CH(idx,k,0).r=t0.r+t1.r; \
        CH(idx,k,0).i=t0.i+t1.i;
#define PARTSTEP3a(u1,u2,twr,twi) \
        { \
        cmplx ca,cb; \
        ca.r=t0.r+twr*t1.r; \
        ca.i=t0.i+twr*t1.i; \
        cb.i=twi*t2.r; \
        cb.r=-(twi*t2.i); \
        PMC(CH(0,k,u1),CH(0,k,u2),ca,cb) \
        }

#define PARTSTEP3b(u1,u2,twr,twi) \
        { \
        cmplx ca,cb,da,db; \
        ca.r=t0.r+twr*t1.r; \
        ca.i=t0.i+twr*t1.i; \
        cb.i=twi*t2.r; \
        cb.r=-(twi*t2.i); \
        PMC(da,db,ca,cb) \
        A_EQ_B_MUL_C (CH(i,k,u1),WA(u1-1,i),da) \
        A_EQ_B_MUL_C (CH(i,k,u2),WA(u2-1,i),db) \
        }
NOINLINE void pk_pass3b (size_t ido, size_t l1, const cmplx * restrict cc,
  cmplx * restrict ch, const cmplx * restrict wa)
  {
  const size_t cdim=3;
  const float tw1r=-0.5, tw1i= 0.86602540378443864676;

  if (ido==1)
    for (size_t k=0; k<l1; ++k)
      {
      PREP3(0)
      PARTSTEP3a(1,2,tw1r,tw1i)
      }
  else
    for (size_t k=0; k<l1; ++k)
      {
      {
      PREP3(0)
      PARTSTEP3a(1,2,tw1r,tw1i)
      }
      for (size_t i=1; i<ido; ++i)
        {
        PREP3(i)
        PARTSTEP3b(1,2,tw1r,tw1i)
        }
      }
  }
#define PARTSTEP3f(u1,u2,twr,twi) \
        { \
        cmplx ca,cb,da,db; \
        ca.r=t0.r+twr*t1.r; \
        ca.i=t0.i+twr*t1.i; \
        cb.i=twi*t2.r; \
        cb.r=-(twi*t2.i); \
        PMC(da,db,ca,cb) \
        A_EQ_CB_MUL_C (CH(i,k,u1),WA(u1-1,i),da) \
        A_EQ_CB_MUL_C (CH(i,k,u2),WA(u2-1,i),db) \
        }
NOINLINE void pk_pass3f (size_t ido, size_t l1, const cmplx * restrict cc,
  cmplx * restrict ch, const cmplx * restrict wa)
  {
  const size_t cdim=3;
  const float tw1r=-0.5, tw1i= -0.86602540378443864676;

  if (ido==1)
    for (size_t k=0; k<l1; ++k)
      {
      PREP3(0)
      PARTSTEP3a(1,2,tw1r,tw1i)
      }
  else
    for (size_t k=0; k<l1; ++k)
      {
      {
      PREP3(0)
      PARTSTEP3a(1,2,tw1r,tw1i)
      }
      for (size_t i=1; i<ido; ++i)
        {
        PREP3(i)
        PARTSTEP3f(1,2,tw1r,tw1i)
        }
      }
  }

NOINLINE void pk_pass4b (size_t ido, size_t l1, const cmplx * restrict cc,
  cmplx * restrict ch, const cmplx * restrict wa)
  {
  const size_t cdim=4;

  if (ido==1)
    for (size_t k=0; k<l1; ++k)
      {
      cmplx t1, t2, t3, t4;
      PMC(t2,t1,CC(0,0,k),CC(0,2,k))
      PMC(t3,t4,CC(0,1,k),CC(0,3,k))
      ROT90(t4)
      PMC(CH(0,k,0),CH(0,k,2),t2,t3)
      PMC(CH(0,k,1),CH(0,k,3),t1,t4)
      }
  else
    for (size_t k=0; k<l1; ++k)
      {
      {
      cmplx t1, t2, t3, t4;
      PMC(t2,t1,CC(0,0,k),CC(0,2,k))
      PMC(t3,t4,CC(0,1,k),CC(0,3,k))
      ROT90(t4)
      PMC(CH(0,k,0),CH(0,k,2),t2,t3)
      PMC(CH(0,k,1),CH(0,k,3),t1,t4)
      }
      for (size_t i=1; i<ido; ++i)
        {
        cmplx c2, c3, c4, t1, t2, t3, t4;
        cmplx cc0=CC(i,0,k), cc1=CC(i,1,k),cc2=CC(i,2,k),cc3=CC(i,3,k);
        PMC(t2,t1,cc0,cc2)
        PMC(t3,t4,cc1,cc3)
        ROT90(t4)
        cmplx wa0=WA(0,i), wa1=WA(1,i),wa2=WA(2,i);
        PMC(CH(i,k,0),c3,t2,t3)
        PMC(c2,c4,t1,t4)
        A_EQ_B_MUL_C (CH(i,k,1),wa0,c2)
        A_EQ_B_MUL_C (CH(i,k,2),wa1,c3)
        A_EQ_B_MUL_C (CH(i,k,3),wa2,c4)
        }
      }
  }
NOINLINE void pk_pass4f (size_t ido, size_t l1, const cmplx * restrict cc,
  cmplx * restrict ch, const cmplx * restrict wa)
  {
  const size_t cdim=4;

  if (ido==1)
    for (size_t k=0; k<l1; ++k)
      {
      cmplx t1, t2, t3, t4;
      PMC(t2,t1,CC(0,0,k),CC(0,2,k))
      PMC(t3,t4,CC(0,1,k),CC(0,3,k))
      ROTM90(t4)
      PMC(CH(0,k,0),CH(0,k,2),t2,t3)
      PMC(CH(0,k,1),CH(0,k,3),t1,t4)
      }
  else
    for (size_t k=0; k<l1; ++k)
      {
      {
      cmplx t1, t2, t3, t4;
      PMC(t2,t1,CC(0,0,k),CC(0,2,k))
      PMC(t3,t4,CC(0,1,k),CC(0,3,k))
      ROTM90(t4)
      PMC(CH(0,k,0),CH(0,k,2),t2,t3)
      PMC (CH(0,k,1),CH(0,k,3),t1,t4)
      }
      for (size_t i=1; i<ido; ++i)
        {
        cmplx c2, c3, c4, t1, t2, t3, t4;
        cmplx cc0=CC(i,0,k), cc1=CC(i,1,k),cc2=CC(i,2,k),cc3=CC(i,3,k);
        PMC(t2,t1,cc0,cc2)
        PMC(t3,t4,cc1,cc3)
        ROTM90(t4)
        cmplx wa0=WA(0,i), wa1=WA(1,i),wa2=WA(2,i);
        PMC(CH(i,k,0),c3,t2,t3)
        PMC(c2,c4,t1,t4)
        A_EQ_CB_MUL_C (CH(i,k,1),wa0,c2)
        A_EQ_CB_MUL_C (CH(i,k,2),wa1,c3)
        A_EQ_CB_MUL_C (CH(i,k,3),wa2,c4)
        }
      }
  }

#define PREP5(idx) \
        cmplx t0 = CC(idx,0,k), t1, t2, t3, t4; \
        PMC (t1,t4,CC(idx,1,k),CC(idx,4,k)) \
        PMC (t2,t3,CC(idx,2,k),CC(idx,3,k)) \
        CH(idx,k,0).r=t0.r+t1.r+t2.r; \
        CH(idx,k,0).i=t0.i+t1.i+t2.i;

#define PARTSTEP5a(u1,u2,twar,twbr,twai,twbi) \
        { \
        cmplx ca,cb; \
        ca.r=t0.r+twar*t1.r+twbr*t2.r; \
        ca.i=t0.i+twar*t1.i+twbr*t2.i; \
        cb.i=twai*t4.r twbi*t3.r; \
        cb.r=-(twai*t4.i twbi*t3.i); \
        PMC(CH(0,k,u1),CH(0,k,u2),ca,cb) \
        }

#define PARTSTEP5b(u1,u2,twar,twbr,twai,twbi) \
        { \
        cmplx ca,cb,da,db; \
        ca.r=t0.r+twar*t1.r+twbr*t2.r; \
        ca.i=t0.i+twar*t1.i+twbr*t2.i; \
        cb.i=twai*t4.r twbi*t3.r; \
        cb.r=-(twai*t4.i twbi*t3.i); \
        PMC(da,db,ca,cb) \
        A_EQ_B_MUL_C (CH(i,k,u1),WA(u1-1,i),da) \
        A_EQ_B_MUL_C (CH(i,k,u2),WA(u2-1,i),db) \
        }
NOINLINE void pk_pass5b (size_t ido, size_t l1, const cmplx * restrict cc,
  cmplx * restrict ch, const cmplx * restrict wa)
  {
  const size_t cdim=5;
  const float tw1r= 0.3090169943749474241,
               tw1i= 0.95105651629515357212,
               tw2r= -0.8090169943749474241,
               tw2i= 0.58778525229247312917;

  if (ido==1)
    for (size_t k=0; k<l1; ++k)
      {
      PREP5(0)
      PARTSTEP5a(1,4,tw1r,tw2r,+tw1i,+tw2i)
      PARTSTEP5a(2,3,tw2r,tw1r,+tw2i,-tw1i)
      }
  else
    for (size_t k=0; k<l1; ++k)
      {
      {
      PREP5(0)
      PARTSTEP5a(1,4,tw1r,tw2r,+tw1i,+tw2i)
      PARTSTEP5a(2,3,tw2r,tw1r,+tw2i,-tw1i)
      }
      for (size_t i=1; i<ido; ++i)
        {
        PREP5(i)
        PARTSTEP5b(1,4,tw1r,tw2r,+tw1i,+tw2i)
        PARTSTEP5b(2,3,tw2r,tw1r,+tw2i,-tw1i)
        }
      }
  }
#define PARTSTEP5f(u1,u2,twar,twbr,twai,twbi) \
        { \
        cmplx ca,cb,da,db; \
        ca.r=t0.r+twar*t1.r+twbr*t2.r; \
        ca.i=t0.i+twar*t1.i+twbr*t2.i; \
        cb.i=twai*t4.r twbi*t3.r; \
        cb.r=-(twai*t4.i twbi*t3.i); \
        PMC(da,db,ca,cb) \
        A_EQ_CB_MUL_C (CH(i,k,u1),WA(u1-1,i),da) \
        A_EQ_CB_MUL_C (CH(i,k,u2),WA(u2-1,i),db) \
        }
NOINLINE void pk_pass5f (size_t ido, size_t l1, const cmplx * restrict cc,
  cmplx * restrict ch, const cmplx * restrict wa)
  {
  const size_t cdim=5;
  const float tw1r= 0.3090169943749474241,
               tw1i= -0.95105651629515357212,
               tw2r= -0.8090169943749474241,
               tw2i= -0.58778525229247312917;

  if (ido==1)
    for (size_t k=0; k<l1; ++k)
      {
      PREP5(0)
      PARTSTEP5a(1,4,tw1r,tw2r,+tw1i,+tw2i)
      PARTSTEP5a(2,3,tw2r,tw1r,+tw2i,-tw1i)
      }
  else
    for (size_t k=0; k<l1; ++k)
      {
      {
      PREP5(0)
      PARTSTEP5a(1,4,tw1r,tw2r,+tw1i,+tw2i)
      PARTSTEP5a(2,3,tw2r,tw1r,+tw2i,-tw1i)
      }
      for (size_t i=1; i<ido; ++i)
        {
        PREP5(i)
        PARTSTEP5f(1,4,tw1r,tw2r,+tw1i,+tw2i)
        PARTSTEP5f(2,3,tw2r,tw1r,+tw2i,-tw1i)
        }
      }
  }

#define PREP7(idx) \
        cmplx t1 = CC(idx,0,k), t2, t3, t4, t5, t6, t7; \
        PMC (t2,t7,CC(idx,1,k),CC(idx,6,k)) \
        PMC (t3,t6,CC(idx,2,k),CC(idx,5,k)) \
        PMC (t4,t5,CC(idx,3,k),CC(idx,4,k)) \
        CH(idx,k,0).r=t1.r+t2.r+t3.r+t4.r; \
        CH(idx,k,0).i=t1.i+t2.i+t3.i+t4.i;

#define PARTSTEP7a0(u1,u2,x1,x2,x3,y1,y2,y3,out1,out2) \
        { \
        cmplx ca,cb; \
        ca.r=t1.r+x1*t2.r+x2*t3.r+x3*t4.r; \
        ca.i=t1.i+x1*t2.i+x2*t3.i+x3*t4.i; \
        cb.i=y1*t7.r y2*t6.r y3*t5.r; \
        cb.r=-(y1*t7.i y2*t6.i y3*t5.i); \
        PMC(out1,out2,ca,cb) \
        }
#define PARTSTEP7a(u1,u2,x1,x2,x3,y1,y2,y3) \
        PARTSTEP7a0(u1,u2,x1,x2,x3,y1,y2,y3,CH(0,k,u1),CH(0,k,u2))
#define PARTSTEP7(u1,u2,x1,x2,x3,y1,y2,y3) \
        { \
        cmplx da,db; \
        PARTSTEP7a0(u1,u2,x1,x2,x3,y1,y2,y3,da,db) \
        MULPMSIGNC (CH(i,k,u1),WA(u1-1,i),da) \
        MULPMSIGNC (CH(i,k,u2),WA(u2-1,i),db) \
        }

NOINLINE void pk_pass7 (size_t ido, size_t l1, const cmplx * restrict cc,
  cmplx * restrict ch, const cmplx * restrict wa, const int sign)
  {
  const size_t cdim=7;
  const float tw1r= 0.623489801858733530525,
               tw1i= (sign>0) ? 0.7818314824680298087084f : -0.7818314824680298087084f,
               tw2r= -0.222520933956314404289,
               tw2i= (sign>0) ? 0.9749279121818236070181f : -0.9749279121818236070181f,
               tw3r= -0.9009688679024191262361,
               tw3i= (sign>0) ? 0.4338837391175581204758f : -0.4338837391175581204758f;

  if (ido==1)
    for (size_t k=0; k<l1; ++k)
      {
      PREP7(0)
      PARTSTEP7a(1,6,tw1r,tw2r,tw3r,+tw1i,+tw2i,+tw3i)
      PARTSTEP7a(2,5,tw2r,tw3r,tw1r,+tw2i,-tw3i,-tw1i)
      PARTSTEP7a(3,4,tw3r,tw1r,tw2r,+tw3i,-tw1i,+tw2i)
      }
  else
    for (size_t k=0; k<l1; ++k)
      {
      {
      PREP7(0)
      PARTSTEP7a(1,6,tw1r,tw2r,tw3r,+tw1i,+tw2i,+tw3i)
      PARTSTEP7a(2,5,tw2r,tw3r,tw1r,+tw2i,-tw3i,-tw1i)
      PARTSTEP7a(3,4,tw3r,tw1r,tw2r,+tw3i,-tw1i,+tw2i)
      }
      for (size_t i=1; i<ido; ++i)
        {
        PREP7(i)
        PARTSTEP7(1,6,tw1r,tw2r,tw3r,+tw1i,+tw2i,+tw3i)
        PARTSTEP7(2,5,tw2r,tw3r,tw1r,+tw2i,-tw3i,-tw1i)
        PARTSTEP7(3,4,tw3r,tw1r,tw2r,+tw3i,-tw1i,+tw2i)
        }
      }
  }

#define PREP11(idx) \
        cmplx t1 = CC(idx,0,k), t2, t3, t4, t5, t6, t7, t8, t9, t10, t11; \
        PMC (t2,t11,CC(idx,1,k),CC(idx,10,k)) \
        PMC (t3,t10,CC(idx,2,k),CC(idx, 9,k)) \
        PMC (t4,t9 ,CC(idx,3,k),CC(idx, 8,k)) \
        PMC (t5,t8 ,CC(idx,4,k),CC(idx, 7,k)) \
        PMC (t6,t7 ,CC(idx,5,k),CC(idx, 6,k)) \
        CH(idx,k,0).r=t1.r+t2.r+t3.r+t4.r+t5.r+t6.r; \
        CH(idx,k,0).i=t1.i+t2.i+t3.i+t4.i+t5.i+t6.i;

#define PARTSTEP11a0(u1,u2,x1,x2,x3,x4,x5,y1,y2,y3,y4,y5,out1,out2) \
        { \
        cmplx ca,cb; \
        ca.r=t1.r+x1*t2.r+x2*t3.r+x3*t4.r+x4*t5.r+x5*t6.r; \
        ca.i=t1.i+x1*t2.i+x2*t3.i+x3*t4.i+x4*t5.i+x5*t6.i; \
        cb.i=y1*t11.r y2*t10.r y3*t9.r y4*t8.r y5*t7.r; \
        cb.r=-(y1*t11.i y2*t10.i y3*t9.i y4*t8.i y5*t7.i ); \
        PMC(out1,out2,ca,cb) \
        }
#define PARTSTEP11a(u1,u2,x1,x2,x3,x4,x5,y1,y2,y3,y4,y5) \
        PARTSTEP11a0(u1,u2,x1,x2,x3,x4,x5,y1,y2,y3,y4,y5,CH(0,k,u1),CH(0,k,u2))
#define PARTSTEP11(u1,u2,x1,x2,x3,x4,x5,y1,y2,y3,y4,y5) \
        { \
        cmplx da,db; \
        PARTSTEP11a0(u1,u2,x1,x2,x3,x4,x5,y1,y2,y3,y4,y5,da,db) \
        MULPMSIGNC (CH(i,k,u1),WA(u1-1,i),da) \
        MULPMSIGNC (CH(i,k,u2),WA(u2-1,i),db) \
        }

NOINLINE void pk_pass11 (size_t ido, size_t l1, const cmplx * restrict cc,
  cmplx * restrict ch, const cmplx * restrict wa, const int sign)
  {
  const size_t cdim=11;
  const float tw1r =        0.8412535328311811688618,
               tw1i = (sign>0) ? 0.5406408174555975821076f : -0.5406408174555975821076f,
               tw2r =        0.4154150130018864255293,
               tw2i = (sign>0) ? 0.9096319953545183714117f : -0.9096319953545183714117f,
               tw3r =       -0.1423148382732851404438,
               tw3i = (sign>0) ? 0.9898214418809327323761f : -0.9898214418809327323761f,
               tw4r =       -0.6548607339452850640569,
               tw4i = (sign>0) ? 0.755749574354258283774f : -0.755749574354258283774f,
               tw5r =       -0.9594929736144973898904,
               tw5i = (sign>0) ? 0.2817325568414296977114f : -0.2817325568414296977114f;

  if (ido==1)
    for (size_t k=0; k<l1; ++k)
      {
      PREP11(0)
      PARTSTEP11a(1,10,tw1r,tw2r,tw3r,tw4r,tw5r,+tw1i,+tw2i,+tw3i,+tw4i,+tw5i)
      PARTSTEP11a(2, 9,tw2r,tw4r,tw5r,tw3r,tw1r,+tw2i,+tw4i,-tw5i,-tw3i,-tw1i)
      PARTSTEP11a(3, 8,tw3r,tw5r,tw2r,tw1r,tw4r,+tw3i,-tw5i,-tw2i,+tw1i,+tw4i)
      PARTSTEP11a(4, 7,tw4r,tw3r,tw1r,tw5r,tw2r,+tw4i,-tw3i,+tw1i,+tw5i,-tw2i)
      PARTSTEP11a(5, 6,tw5r,tw1r,tw4r,tw2r,tw3r,+tw5i,-tw1i,+tw4i,-tw2i,+tw3i)
      }
  else
    for (size_t k=0; k<l1; ++k)
      {
      {
      PREP11(0)
      PARTSTEP11a(1,10,tw1r,tw2r,tw3r,tw4r,tw5r,+tw1i,+tw2i,+tw3i,+tw4i,+tw5i)
      PARTSTEP11a(2, 9,tw2r,tw4r,tw5r,tw3r,tw1r,+tw2i,+tw4i,-tw5i,-tw3i,-tw1i)
      PARTSTEP11a(3, 8,tw3r,tw5r,tw2r,tw1r,tw4r,+tw3i,-tw5i,-tw2i,+tw1i,+tw4i)
      PARTSTEP11a(4, 7,tw4r,tw3r,tw1r,tw5r,tw2r,+tw4i,-tw3i,+tw1i,+tw5i,-tw2i)
      PARTSTEP11a(5, 6,tw5r,tw1r,tw4r,tw2r,tw3r,+tw5i,-tw1i,+tw4i,-tw2i,+tw3i)
      }
      for (size_t i=1; i<ido; ++i)
        {
        PREP11(i)
        PARTSTEP11(1,10,tw1r,tw2r,tw3r,tw4r,tw5r,+tw1i,+tw2i,+tw3i,+tw4i,+tw5i)
        PARTSTEP11(2, 9,tw2r,tw4r,tw5r,tw3r,tw1r,+tw2i,+tw4i,-tw5i,-tw3i,-tw1i)
        PARTSTEP11(3, 8,tw3r,tw5r,tw2r,tw1r,tw4r,+tw3i,-tw5i,-tw2i,+tw1i,+tw4i)
        PARTSTEP11(4, 7,tw4r,tw3r,tw1r,tw5r,tw2r,+tw4i,-tw3i,+tw1i,+tw5i,-tw2i)
        PARTSTEP11(5, 6,tw5r,tw1r,tw4r,tw2r,tw3r,+tw5i,-tw1i,+tw4i,-tw2i,+tw3i)
        }
      }
  }

/* radix 8: pocketfft_hdronly.h (C++) cfftp::pass8, written with this file's macros */
#define PMINPLACE(a,b) { cmplx t_=a; a.r+=b.r; a.i+=b.i; b.r=t_.r-b.r; b.i=t_.i-b.i; }
#define HSQT2 0.707106781186547524400844362104849f
#define ROTX45F(a) { float tmp_=a.r; a.r=HSQT2*(a.r+a.i); a.i=HSQT2*(a.i-tmp_); }
#define ROTX45B(a) { float tmp_=a.r; a.r=HSQT2*(a.r-a.i); a.i=HSQT2*(a.i+tmp_); }
#define ROTX135F(a) { float tmp_=a.r; a.r=HSQT2*(a.i-a.r); a.i=HSQT2*(-tmp_-a.i); }
#define ROTX135B(a) { float tmp_=a.r; a.r=HSQT2*(-a.r-a.i); a.i=HSQT2*(tmp_-a.i); }
#define SUMC(a,b,c) { a.r=b.r+c.r; a.i=b.i+c.i; }
#define DIFC(a,b,c) { a.r=b.r-c.r; a.i=b.i-c.i; }
#define PASS8(NAME, ROTX90, ROTX45, ROTX135, SMUL) \
NOINLINE void NAME (size_t ido, size_t l1, const cmplx * restrict cc, \
  cmplx * restrict ch, const cmplx * restrict wa) \
  { \
  const size_t cdim=8; \
  for (size_t k=0; k<l1; ++k) \
    { \
      { \
      cmplx a0, a1, a2, a3, a4, a5, a6, a7, s_, d_; \
      PMC(a1,a5,CC(0,1,k),CC(0,5,k)) \
      PMC(a3,a7,CC(0,3,k),CC(0,7,k)) \
      PMINPLACE(a1,a3) \
      ROTX90(a3) \
      ROTX90(a7) \
      PMINPLACE(a5,a7) \
      ROTX45(a5) \
      ROTX135(a7) \
      PMC(a0,a4,CC(0,0,k),CC(0,4,k)) \
      PMC(a2,a6,CC(0,2,k),CC(0,6,k)) \
      SUMC(s_,a0,a2) DIFC(d_,a0,a2) \
      PMC(CH(0,k,0),CH(0,k,4),s_,a1) \
      PMC(CH(0,k,2),CH(0,k,6),d_,a3) \
      ROTX90(a6) \
      SUMC(s_,a4,a6) DIFC(d_,a4,a6) \
      PMC(CH(0,k,1),CH(0,k,5),s_,a5) \
      PMC(CH(0,k,3),CH(0,k,7),d_,a7) \
      } \
    for (size_t i=1; i<ido; ++i) \
      { \
      cmplx a0, a1, a2, a3, a4, a5, a6, a7, t; \
      PMC(a1,a5,CC(i,1,k),CC(i,5,k)) \
      PMC(a3,a7,CC(i,3,k),CC(i,7,k)) \
      ROTX90(a7) \
      PMINPLACE(a1,a3) \
      ROTX90(a3) \
      PMINPLACE(a5,a7) \
      ROTX45(a5) \
      ROTX135(a7) \
      PMC(a0,a4,CC(i,0,k),CC(i,4,k)) \
      PMC(a2,a6,CC(i,2,k),CC(i,6,k)) \
      PMINPLACE(a0,a2) \
      SUMC(CH(i,k,0),a0,a1) \
      DIFC(t,a0,a1) SMUL(CH(i,k,4),WA(3,i),t) \
      SUMC(t,a2,a3) SMUL(CH(i,k,2),WA(1,i),t) \
      DIFC(t,a2,a3) SMUL(CH(i,k,6),WA(5,i),t) \
      ROTX90(a6) \
      PMINPLACE(a4,a6) \
      SUMC(t,a4,a5) SMUL(CH(i,k,1),WA(0,i),t) \
      DIFC(t,a4,a5) SMUL(CH(i,k,5),WA(4,i),t) \
      SUMC(t,a6,a7) SMUL(CH(i,k,3),WA(2,i),t) \
      DIFC(t,a6,a7) SMUL(CH(i,k,7),WA(6,i),t) \
      } \
    } \
  }
PASS8(pk_pass8b, ROT90, ROTX45B, ROTX135B, A_EQ_B_MUL_C)
PASS8(pk_pass8f, ROTM90, ROTX45F, ROTX135F, A_EQ_CB_MUL_C)

NOINLINE static void pass_all(const cfftp_plan_i *plan, cmplx c[], cmplx ch[], float fct)
  {
  const int sign=plan->sign;
  if (plan->length==1) return;
  size_t len=plan->length;
  size_t l1=1, nf=plan->nfct;
  cmplx *p1=c, *p2=ch;

  for(size_t k1=0; k1<nf; k1++)
    {
    size_t ip=plan->fct[k1].fct;
    size_t l2=ip*l1;
    size_t ido = len/l2;
    if (plan->fct[k1].fns) plan->fct[k1].fns(ido, l1, p1, p2, plan->fct[k1].tw, sign);   /* 7, 11 */
    else plan->fct[k1].fn(ido, l1, p1, p2, plan->fct[k1].tw);
    SWAP(p1,p2,cmplx *);
    l1=l2;
    }
  if (p1!=c)
    {
    if (fct!=1.)
      for (size_t i=0; i<len; ++i)
        {
        c[i].r = ch[i].r*fct;
        c[i].i = ch[i].i*fct;
        }
    else
      memcpy (c,p1,len*sizeof(cmplx));
    }
  else
    if (fct!=1.)
      for (size_t i=0; i<len; ++i)
        {
        c[i].r *= fct;
        c[i].i *= fct;
        }
  }

#undef PMSIGNC
#undef A_EQ_B_MUL_C
#undef A_EQ_CB_MUL_C
#undef MULPMSIGNC
#undef MULPMSIGNCEQ

#undef WA
#undef CC
#undef CH
#undef ROT90
#undef SCALEC
#undef ADDC
#undef PMC

void pk_cfft(const cfftp_plan_i *plan, float c[], float ch[], float fct)
  { pass_all(plan,(cmplx *)c,(cmplx *)ch, fct); }

#define WA(x,i) wa[(i)+(x)*(ido-1)]
#define PM(a,b,c,d) { a=c+d; b=c-d; }
/* (a+ib) = conj(c+id) * (e+if) */
#define MULPM(a,b,c,d,e,f) { a=c*e+d*f; b=c*f-d*e; }

#define CC(a,b,c) cc[(a)+ido*((b)+l1*(c))]
#define CH(a,b,c) ch[(a)+ido*((b)+cdim*(c))]

NOINLINE void pk_radf2 (size_t ido, size_t l1, const float * restrict cc,
  float * restrict ch, const float * restrict wa)
  {
  const size_t cdim=2;

  for (size_t k=0; k<l1; k++)
    PM (CH(0,0,k),CH(ido-1,1,k),CC(0,k,0),CC(0,k,1))
  if ((ido&1)==0)
    for (size_t k=0; k<l1; k++)
      {
      CH(    0,1,k) = -CC(ido-1,k,1);
      CH(ido-1,0,k) =  CC(ido-1,k,0);
      }
  if (ido<=2) return;
  for (size_t k=0; k<l1; k++)
    for (size_t i=2; i<ido; i+=2)
      {
      size_t ic=ido-i;
      float tr2, ti2;
      MULPM (tr2,ti2,WA(0,i-2),WA(0,i-1),CC(i-1,k,1),CC(i,k,1))
      PM (CH(i-1,0,k),CH(ic-1,1,k),CC(i-1,k,0),tr2)
      PM (CH(i  ,0,k),CH(ic  ,1,k),ti2,CC(i  ,k,0))
      }
  }

NOINLINE void pk_radf3 (size_t ido, size_t l1, const float * restrict cc,
  float * restrict ch, const float * restrict wa)
  {
  const size_t cdim=3;
  static const float taur=-0.5, taui=0.86602540378443864676;

  for (size_t k=0; k<l1; k++)
    {
    float cr2=CC(0,k,1)+CC(0,k,2);
    CH(0,0,k) = CC(0,k,0)+cr2;
    CH(0,2,k) = taui*(CC(0,k,2)-CC(0,k,1));
    CH(ido-1,1,k) = CC(0,k,0)+taur*cr2;
    }
  if (ido==1) return;
  for (size_t k=0; k<l1; k++)
    for (size_t i=2; i<ido; i+=2)
      {
      size_t ic=ido-i;
      float di2, di3, dr2, dr3;
      MULPM (dr2,di2,WA(0,i-2),WA(0,i-1),CC(i-1,k,1),CC(i,k,1)) // d2=conj(WA0)*CC1
      MULPM (dr3,di3,WA(1,i-2),WA(1,i-1),CC(i-1,k,2),CC(i,k,2)) // d3=conj(WA1)*CC2
      float cr2=dr2+dr3; // c add
      float ci2=di2+di3;
      CH(i-1,0,k) = CC(i-1,k,0)+cr2; // c add
      CH(i  ,0,k) = CC(i  ,k,0)+ci2;
      float tr2 = CC(i-1,k,0)+taur*cr2; // c add
      float ti2 = CC(i  ,k,0)+taur*ci2;
      float tr3 = taui*(di2-di3);  // t3 = taui*i*(d3-d2)?
      float ti3 = taui*(dr3-dr2);
      PM(CH(i-1,2,k),CH(ic-1,1,k),tr2,tr3) // PM(i) = t2+t3
      PM(CH(i  ,2,k),CH(ic  ,1,k),ti3,ti2) // PM(ic) = conj(t2-t3)
      }
  }

NOINLINE void pk_radf4 (size_t ido, size_t l1, const float * restrict cc,
  float * restrict ch, const float * restrict wa)
  {
  const size_t cdim=4;
  static const float hsqt2=0.70710678118654752440;

  for (size_t k=0; k<l1; k++)
    {
    float tr1,tr2;
    PM (tr1,CH(0,2,k),CC(0,k,3),CC(0,k,1))
    PM (tr2,CH(ido-1,1,k),CC(0,k,0),CC(0,k,2))
    PM (CH(0,0,k),CH(ido-1,3,k),tr2,tr1)
    }
  if ((ido&1)==0)
    for (size_t k=0; k<l1; k++)
      {
      float ti1=-hsqt2*(CC(ido-1,k,1)+CC(ido-1,k,3));
      float tr1= hsqt2*(CC(ido-1,k,1)-CC(ido-1,k,3));
      PM (CH(ido-1,0,k),CH(ido-1,2,k),CC(ido-1,k,0),tr1)
      PM (CH(    0,3,k),CH(    0,1,k),ti1,CC(ido-1,k,2))
      }
  if (ido<=2) return;
  for (size_t k=0; k<l1; k++)
    for (size_t i=2; i<ido; i+=2)
      {
      size_t ic=ido-i;
      float ci2, ci3, ci4, cr2, cr3, cr4, ti1, ti2, ti3, ti4, tr1, tr2, tr3, tr4;
      MULPM(cr2,ci2,WA(0,i-2),WA(0,i-1),CC(i-1,k,1),CC(i,k,1))
      MULPM(cr3,ci3,WA(1,i-2),WA(1,i-1),CC(i-1,k,2),CC(i,k,2))
      MULPM(cr4,ci4,WA(2,i-2),WA(2,i-1),CC(i-1,k,3),CC(i,k,3))
      PM(tr1,tr4,cr4,cr2)
      PM(ti1,ti4,ci2,ci4)
      PM(tr2,tr3,CC(i-1,k,0),cr3)
      PM(ti2,ti3,CC(i  ,k,0),ci3)
      PM(CH(i-1,0,k),CH(ic-1,3,k),tr2,tr1)
      PM(CH(i  ,0,k),CH(ic  ,3,k),ti1,ti2)
      PM(CH(i-1,2,k),CH(ic-1,1,k),tr3,ti4)
      PM(CH(i  ,2,k),CH(ic  ,1,k),tr4,ti3)
      }
  }

NOINLINE void pk_radf5 (size_t ido, size_t l1, const float * restrict cc,
  float * restrict ch, const float * restrict wa)
  {
  const size_t cdim=5;
  static const float tr11= 0.3090169943749474241, ti11=0.95105651629515357212,
                      tr12=-0.8090169943749474241, ti12=0.58778525229247312917;

  for (size_t k=0; k<l1; k++)
    {
    float cr2, cr3, ci4, ci5;
    PM (cr2,ci5,CC(0,k,4),CC(0,k,1))
    PM (cr3,ci4,CC(0,k,3),CC(0,k,2))
    CH(0,0,k)=CC(0,k,0)+cr2+cr3;
    CH(ido-1,1,k)=CC(0,k,0)+tr11*cr2+tr12*cr3;
    CH(0,2,k)=ti11*ci5+ti12*ci4;
    CH(ido-1,3,k)=CC(0,k,0)+tr12*cr2+tr11*cr3;
    CH(0,4,k)=ti12*ci5-ti11*ci4;
    }
  if (ido==1) return;
  for (size_t k=0; k<l1;++k)
    for (size_t i=2; i<ido; i+=2)
      {
      float ci2, di2, ci4, ci5, di3, di4, di5, ci3, cr2, cr3, dr2, dr3,
         dr4, dr5, cr5, cr4, ti2, ti3, ti5, ti4, tr2, tr3, tr4, tr5;
      size_t ic=ido-i;
      MULPM (dr2,di2,WA(0,i-2),WA(0,i-1),CC(i-1,k,1),CC(i,k,1))
      MULPM (dr3,di3,WA(1,i-2),WA(1,i-1),CC(i-1,k,2),CC(i,k,2))
      MULPM (dr4,di4,WA(2,i-2),WA(2,i-1),CC(i-1,k,3),CC(i,k,3))
      MULPM (dr5,di5,WA(3,i-2),WA(3,i-1),CC(i-1,k,4),CC(i,k,4))
      PM(cr2,ci5,dr5,dr2)
      PM(ci2,cr5,di2,di5)
      PM(cr3,ci4,dr4,dr3)
      PM(ci3,cr4,di3,di4)
      CH(i-1,0,k)=CC(i-1,k,0)+cr2+cr3;
      CH(i  ,0,k)=CC(i  ,k,0)+ci2+ci3;
      tr2=CC(i-1,k,0)+tr11*cr2+tr12*cr3;
      ti2=CC(i  ,k,0)+tr11*ci2+tr12*ci3;
      tr3=CC(i-1,k,0)+tr12*cr2+tr11*cr3;
      ti3=CC(i  ,k,0)+tr12*ci2+tr11*ci3;
      MULPM(tr5,tr4,cr5,cr4,ti11,ti12)
      MULPM(ti5,ti4,ci5,ci4,ti11,ti12)
      PM(CH(i-1,2,k),CH(ic-1,1,k),tr2,tr5)
      PM(CH(i  ,2,k),CH(ic  ,1,k),ti5,ti2)
      PM(CH(i-1,4,k),CH(ic-1,3,k),tr3,tr4)
      PM(CH(i  ,4,k),CH(ic  ,3,k),ti4,ti3)
      }
  }

#undef CC
#undef CH
#define C1(a,b,c) cc[(a)+ido*((b)+l1*(c))]
#define C2(a,b) cc[(a)+idl1*(b)]
#define CH2(a,b) ch[(a)+idl1*(b)]
#define CC(a,b,c) cc[(a)+ido*((b)+cdim*(c))]
#define CH(a,b,c) ch[(a)+ido*((b)+l1*(c))]
NOINLINE void pk_radfg (size_t ido, size_t ip, size_t l1,
  float * restrict cc, float * restrict ch, const float * restrict wa,
  const float * restrict csarr)
  {
  const size_t cdim=ip;
  size_t ipph=(ip+1)/2;
  size_t idl1 = ido*l1;

  if (ido>1)
    {
    for (size_t j=1, jc=ip-1; j<ipph; ++j,--jc)              // 114
      {
      size_t is=(j-1)*(ido-1),
             is2=(jc-1)*(ido-1);
      for (size_t k=0; k<l1; ++k)                            // 113
        {
        size_t idij=is;
        size_t idij2=is2;
        for (size_t i=1; i<=ido-2; i+=2)                      // 112
          {
          float t1=C1(i,k,j ), t2=C1(i+1,k,j ),
                 t3=C1(i,k,jc), t4=C1(i+1,k,jc);
          float x1=wa[idij]*t1 + wa[idij+1]*t2,
                 x2=wa[idij]*t2 - wa[idij+1]*t1,
                 x3=wa[idij2]*t3 + wa[idij2+1]*t4,
                 x4=wa[idij2]*t4 - wa[idij2+1]*t3;
          C1(i  ,k,j ) = x1+x3;
          C1(i  ,k,jc) = x2-x4;
          C1(i+1,k,j ) = x2+x4;
          C1(i+1,k,jc) = x3-x1;
          idij+=2;
          idij2+=2;
          }
        }
      }
    }

  for (size_t j=1, jc=ip-1; j<ipph; ++j,--jc)                // 123
    for (size_t k=0; k<l1; ++k)                              // 122
      {
      float t1=C1(0,k,j), t2=C1(0,k,jc);
      C1(0,k,j ) = t1+t2;
      C1(0,k,jc) = t2-t1;
      }

//everything in C
//memset(ch,0,ip*l1*ido*sizeof(float));

  for (size_t l=1,lc=ip-1; l<ipph; ++l,--lc)                 // 127
    {
    for (size_t ik=0; ik<idl1; ++ik)                         // 124
      {
      CH2(ik,l ) = C2(ik,0)+csarr[2*l]*C2(ik,1)+csarr[4*l]*C2(ik,2);
      CH2(ik,lc) = csarr[2*l+1]*C2(ik,ip-1)+csarr[4*l+1]*C2(ik,ip-2);
      }
    size_t iang = 2*l;
    size_t j=3, jc=ip-3;
    for (; j<ipph-3; j+=4,jc-=4)              // 126
      {
      iang+=l; if (iang>=ip) iang-=ip;
      float ar1=csarr[2*iang], ai1=csarr[2*iang+1];
      iang+=l; if (iang>=ip) iang-=ip;
      float ar2=csarr[2*iang], ai2=csarr[2*iang+1];
      iang+=l; if (iang>=ip) iang-=ip;
      float ar3=csarr[2*iang], ai3=csarr[2*iang+1];
      iang+=l; if (iang>=ip) iang-=ip;
      float ar4=csarr[2*iang], ai4=csarr[2*iang+1];
      for (size_t ik=0; ik<idl1; ++ik)                       // 125
        {
        CH2(ik,l ) += ar1*C2(ik,j )+ar2*C2(ik,j +1)
                     +ar3*C2(ik,j +2)+ar4*C2(ik,j +3);
        CH2(ik,lc) += ai1*C2(ik,jc)+ai2*C2(ik,jc-1)
                     +ai3*C2(ik,jc-2)+ai4*C2(ik,jc-3);
        }
      }
    for (; j<ipph-1; j+=2,jc-=2)              // 126
      {
      iang+=l; if (iang>=ip) iang-=ip;
      float ar1=csarr[2*iang], ai1=csarr[2*iang+1];
      iang+=l; if (iang>=ip) iang-=ip;
      float ar2=csarr[2*iang], ai2=csarr[2*iang+1];
      for (size_t ik=0; ik<idl1; ++ik)                       // 125
        {
        CH2(ik,l ) += ar1*C2(ik,j )+ar2*C2(ik,j +1);
        CH2(ik,lc) += ai1*C2(ik,jc)+ai2*C2(ik,jc-1);
        }
      }
    for (; j<ipph; ++j,--jc)              // 126
      {
      iang+=l; if (iang>=ip) iang-=ip;
      float ar=csarr[2*iang], ai=csarr[2*iang+1];
      for (size_t ik=0; ik<idl1; ++ik)                       // 125
        {
        CH2(ik,l ) += ar*C2(ik,j );
        CH2(ik,lc) += ai*C2(ik,jc);
        }
      }
    }
  for (size_t ik=0; ik<idl1; ++ik)                         // 101
    CH2(ik,0) = C2(ik,0);
  for (size_t j=1; j<ipph; ++j)                              // 129
    for (size_t ik=0; ik<idl1; ++ik)                         // 128
      CH2(ik,0) += C2(ik,j);

// everything in CH at this point!
//memset(cc,0,ip*l1*ido*sizeof(float));

  for (size_t k=0; k<l1; ++k)                                // 131
    for (size_t i=0; i<ido; ++i)                             // 130
      CC(i,0,k) = CH(i,k,0);

  for (size_t j=1, jc=ip-1; j<ipph; ++j,--jc)                // 137
    {
    size_t j2=2*j-1;
    for (size_t k=0; k<l1; ++k)                              // 136
      {
      CC(ido-1,j2,k) = CH(0,k,j);
      CC(0,j2+1,k) = CH(0,k,jc);
      }
    }

  if (ido==1) return;

  for (size_t j=1, jc=ip-1; j<ipph; ++j,--jc)                // 140
    {
    size_t j2=2*j-1;
    for(size_t k=0; k<l1; ++k)                               // 139
      for(size_t i=1, ic=ido-i-2; i<=ido-2; i+=2, ic-=2)      // 138
        {
        CC(i   ,j2+1,k) = CH(i  ,k,j )+CH(i  ,k,jc);
        CC(ic  ,j2  ,k) = CH(i  ,k,j )-CH(i  ,k,jc);
        CC(i+1 ,j2+1,k) = CH(i+1,k,j )+CH(i+1,k,jc);
        CC(ic+1,j2  ,k) = CH(i+1,k,jc)-CH(i+1,k,j );
        }
    }
  }
#undef C1
#undef C2
#undef CH2

#undef CH
#undef CC
#define CH(a,b,c) ch[(a)+ido*((b)+l1*(c))]
#define CC(a,b,c) cc[(a)+ido*((b)+cdim*(c))]

NOINLINE void pk_radb2 (size_t ido, size_t l1, const float * restrict cc,
  float * restrict ch, const float * restrict wa)
  {
  const size_t cdim=2;

  for (size_t k=0; k<l1; k++)
    PM (CH(0,k,0),CH(0,k,1),CC(0,0,k),CC(ido-1,1,k))
  if ((ido&1)==0)
    for (size_t k=0; k<l1; k++)
      {
      CH(ido-1,k,0) = 2.*CC(ido-1,0,k);
      CH(ido-1,k,1) =-2.*CC(0    ,1,k);
      }
  if (ido<=2) return;
  for (size_t k=0; k<l1;++k)
    for (size_t i=2; i<ido; i+=2)
      {
      size_t ic=ido-i;
      float ti2, tr2;
      PM (CH(i-1,k,0),tr2,CC(i-1,0,k),CC(ic-1,1,k))
      PM (ti2,CH(i  ,k,0),CC(i  ,0,k),CC(ic  ,1,k))
      MULPM (CH(i,k,1),CH(i-1,k,1),WA(0,i-2),WA(0,i-1),ti2,tr2)
      }
  }

NOINLINE void pk_radb3 (size_t ido, size_t l1, const float * restrict cc,
  float * restrict ch, const float * restrict wa)
  {
  const size_t cdim=3;
  static const float taur=-0.5, taui=0.86602540378443864676;

  for (size_t k=0; k<l1; k++)
    {
    float tr2=2.*CC(ido-1,1,k);
    float cr2=CC(0,0,k)+taur*tr2;
    CH(0,k,0)=CC(0,0,k)+tr2;
    float ci3=2.*taui*CC(0,2,k);
    PM (CH(0,k,2),CH(0,k,1),cr2,ci3);
    }
  if (ido==1) return;
  for (size_t k=0; k<l1; k++)
    for (size_t i=2; i<ido; i+=2)
      {
      size_t ic=ido-i;
      float tr2=CC(i-1,2,k)+CC(ic-1,1,k); // t2=CC(I) + conj(CC(ic))
      float ti2=CC(i  ,2,k)-CC(ic  ,1,k);
      float cr2=CC(i-1,0,k)+taur*tr2;     // c2=CC +taur*t2
      float ci2=CC(i  ,0,k)+taur*ti2;
      CH(i-1,k,0)=CC(i-1,0,k)+tr2;         // CH=CC+t2
      CH(i  ,k,0)=CC(i  ,0,k)+ti2;
      float cr3=taui*(CC(i-1,2,k)-CC(ic-1,1,k));// c3=taui*(CC(i)-conj(CC(ic)))
      float ci3=taui*(CC(i  ,2,k)+CC(ic  ,1,k));
      float di2, di3, dr2, dr3;
      PM(dr3,dr2,cr2,ci3) // d2= (cr2-ci3, ci2+cr3) = c2+i*c3
      PM(di2,di3,ci2,cr3) // d3= (cr2+ci3, ci2-cr3) = c2-i*c3
      MULPM(CH(i,k,1),CH(i-1,k,1),WA(0,i-2),WA(0,i-1),di2,dr2) // ch = WA*d2
      MULPM(CH(i,k,2),CH(i-1,k,2),WA(1,i-2),WA(1,i-1),di3,dr3)
      }
  }

NOINLINE void pk_radb4 (size_t ido, size_t l1, const float * restrict cc,
  float * restrict ch, const float * restrict wa)
  {
  const size_t cdim=4;
  static const float sqrt2=1.41421356237309504880;

  for (size_t k=0; k<l1; k++)
    {
    float tr1, tr2;
    PM (tr2,tr1,CC(0,0,k),CC(ido-1,3,k))
    float tr3=2.*CC(ido-1,1,k);
    float tr4=2.*CC(0,2,k);
    PM (CH(0,k,0),CH(0,k,2),tr2,tr3)
    PM (CH(0,k,3),CH(0,k,1),tr1,tr4)
    }
  if ((ido&1)==0)
    for (size_t k=0; k<l1; k++)
      {
      float tr1,tr2,ti1,ti2;
      PM (ti1,ti2,CC(0    ,3,k),CC(0    ,1,k))
      PM (tr2,tr1,CC(ido-1,0,k),CC(ido-1,2,k))
      CH(ido-1,k,0)=tr2+tr2;
      CH(ido-1,k,1)=sqrt2*(tr1-ti1);
      CH(ido-1,k,2)=ti2+ti2;
      CH(ido-1,k,3)=-sqrt2*(tr1+ti1);
      }
  if (ido<=2) return;
  for (size_t k=0; k<l1;++k)
    for (size_t i=2; i<ido; i+=2)
      {
      float ci2, ci3, ci4, cr2, cr3, cr4, ti1, ti2, ti3, ti4, tr1, tr2, tr3, tr4;
      size_t ic=ido-i;
      PM (tr2,tr1,CC(i-1,0,k),CC(ic-1,3,k))
      PM (ti1,ti2,CC(i  ,0,k),CC(ic  ,3,k))
      PM (tr4,ti3,CC(i  ,2,k),CC(ic  ,1,k))
      PM (tr3,ti4,CC(i-1,2,k),CC(ic-1,1,k))
      PM (CH(i-1,k,0),cr3,tr2,tr3)
      PM (CH(i  ,k,0),ci3,ti2,ti3)
      PM (cr4,cr2,tr1,tr4)
      PM (ci2,ci4,ti1,ti4)
      MULPM (CH(i,k,1),CH(i-1,k,1),WA(0,i-2),WA(0,i-1),ci2,cr2)
      MULPM (CH(i,k,2),CH(i-1,k,2),WA(1,i-2),WA(1,i-1),ci3,cr3)
      MULPM (CH(i,k,3),CH(i-1,k,3),WA(2,i-2),WA(2,i-1),ci4,cr4)
      }
  }

NOINLINE void pk_radb5 (size_t ido, size_t l1, const float * restrict cc,
  float * restrict ch, const float * restrict wa)
  {
  const size_t cdim=5;
  static const float tr11= 0.3090169943749474241, ti11=0.95105651629515357212,
                      tr12=-0.8090169943749474241, ti12=0.58778525229247312917;

  for (size_t k=0; k<l1; k++)
    {
    float ti5=CC(0,2,k)+CC(0,2,k);
    float ti4=CC(0,4,k)+CC(0,4,k);
    float tr2=CC(ido-1,1,k)+CC(ido-1,1,k);
    float tr3=CC(ido-1,3,k)+CC(ido-1,3,k);
    CH(0,k,0)=CC(0,0,k)+tr2+tr3;
    float cr2=CC(0,0,k)+tr11*tr2+tr12*tr3;
    float cr3=CC(0,0,k)+tr12*tr2+tr11*tr3;
    float ci4, ci5;
    MULPM(ci5,ci4,ti5,ti4,ti11,ti12)
    PM(CH(0,k,4),CH(0,k,1),cr2,ci5)
    PM(CH(0,k,3),CH(0,k,2),cr3,ci4)
    }
  if (ido==1) return;
  for (size_t k=0; k<l1;++k)
    for (size_t i=2; i<ido; i+=2)
      {
      size_t ic=ido-i;
      float tr2, tr3, tr4, tr5, ti2, ti3, ti4, ti5;
      PM(tr2,tr5,CC(i-1,2,k),CC(ic-1,1,k))
      PM(ti5,ti2,CC(i  ,2,k),CC(ic  ,1,k))
      PM(tr3,tr4,CC(i-1,4,k),CC(ic-1,3,k))
      PM(ti4,ti3,CC(i  ,4,k),CC(ic  ,3,k))
      CH(i-1,k,0)=CC(i-1,0,k)+tr2+tr3;
      CH(i  ,k,0)=CC(i  ,0,k)+ti2+ti3;
      float cr2=CC(i-1,0,k)+tr11*tr2+tr12*tr3;
      float ci2=CC(i  ,0,k)+tr11*ti2+tr12*ti3;
      float cr3=CC(i-1,0,k)+tr12*tr2+tr11*tr3;
      float ci3=CC(i  ,0,k)+tr12*ti2+tr11*ti3;
      float ci4, ci5, cr5, cr4;
      MULPM(cr5,cr4,tr5,tr4,ti11,ti12)
      MULPM(ci5,ci4,ti5,ti4,ti11,ti12)
      float dr2, dr3, dr4, dr5, di2, di3, di4, di5;
      PM(dr4,dr3,cr3,ci4)
      PM(di3,di4,ci3,cr4)
      PM(dr5,dr2,cr2,ci5)
      PM(di2,di5,ci2,cr5)
      MULPM(CH(i,k,1),CH(i-1,k,1),WA(0,i-2),WA(0,i-1),di2,dr2)
      MULPM(CH(i,k,2),CH(i-1,k,2),WA(1,i-2),WA(1,i-1),di3,dr3)
      MULPM(CH(i,k,3),CH(i-1,k,3),WA(2,i-2),WA(2,i-1),di4,dr4)
      MULPM(CH(i,k,4),CH(i-1,k,4),WA(3,i-2),WA(3,i-1),di5,dr5)
      }
  }

#undef CC
#undef CH
#define CC(a,b,c) cc[(a)+ido*((b)+cdim*(c))]
#define CH(a,b,c) ch[(a)+ido*((b)+l1*(c))]
#define C1(a,b,c) cc[(a)+ido*((b)+l1*(c))]
#define C2(a,b) cc[(a)+idl1*(b)]
#define CH2(a,b) ch[(a)+idl1*(b)]

NOINLINE void pk_radbg (size_t ido, size_t ip, size_t l1,
  float * restrict cc, float * restrict ch, const float * restrict wa,
  const float * restrict csarr)
  {
  const size_t cdim=ip;
  size_t ipph=(ip+1)/ 2;
  size_t idl1 = ido*l1;

  for (size_t k=0; k<l1; ++k)        // 102
    for (size_t i=0; i<ido; ++i)     // 101
      CH(i,k,0) = CC(i,0,k);
  for (size_t j=1, jc=ip-1; j<ipph; ++j, --jc)   // 108
    {
    size_t j2=2*j-1;
    for (size_t k=0; k<l1; ++k)
      {
      CH(0,k,j ) = 2*CC(ido-1,j2,k);
      CH(0,k,jc) = 2*CC(0,j2+1,k);
      }
    }

  if (ido!=1)
    {
    for (size_t j=1, jc=ip-1; j<ipph; ++j,--jc)   // 111
      {
      size_t j2=2*j-1;
      for (size_t k=0; k<l1; ++k)
        for (size_t i=1, ic=ido-i-2; i<=ido-2; i+=2, ic-=2)      // 109
          {
          CH(i  ,k,j ) = CC(i  ,j2+1,k)+CC(ic  ,j2,k);
          CH(i  ,k,jc) = CC(i  ,j2+1,k)-CC(ic  ,j2,k);
          CH(i+1,k,j ) = CC(i+1,j2+1,k)-CC(ic+1,j2,k);
          CH(i+1,k,jc) = CC(i+1,j2+1,k)+CC(ic+1,j2,k);
          }
      }
    }
  for (size_t l=1,lc=ip-1; l<ipph; ++l,--lc)
    {
    for (size_t ik=0; ik<idl1; ++ik)
      {
      C2(ik,l ) = CH2(ik,0)+csarr[2*l]*CH2(ik,1)+csarr[4*l]*CH2(ik,2);
      C2(ik,lc) = csarr[2*l+1]*CH2(ik,ip-1)+csarr[4*l+1]*CH2(ik,ip-2);
      }
    size_t iang=2*l;
    size_t j=3,jc=ip-3;
    for(; j<ipph-3; j+=4,jc-=4)
      {
      iang+=l; if(iang>ip) iang-=ip;
      float ar1=csarr[2*iang], ai1=csarr[2*iang+1];
      iang+=l; if(iang>ip) iang-=ip;
      float ar2=csarr[2*iang], ai2=csarr[2*iang+1];
      iang+=l; if(iang>ip) iang-=ip;
      float ar3=csarr[2*iang], ai3=csarr[2*iang+1];
      iang+=l; if(iang>ip) iang-=ip;
      float ar4=csarr[2*iang], ai4=csarr[2*iang+1];
      for (size_t ik=0; ik<idl1; ++ik)
        {
        C2(ik,l ) += ar1*CH2(ik,j )+ar2*CH2(ik,j +1)
                    +ar3*CH2(ik,j +2)+ar4*CH2(ik,j +3);
        C2(ik,lc) += ai1*CH2(ik,jc)+ai2*CH2(ik,jc-1)
                    +ai3*CH2(ik,jc-2)+ai4*CH2(ik,jc-3);
        }
      }
    for(; j<ipph-1; j+=2,jc-=2)
      {
      iang+=l; if(iang>ip) iang-=ip;
      float ar1=csarr[2*iang], ai1=csarr[2*iang+1];
      iang+=l; if(iang>ip) iang-=ip;
      float ar2=csarr[2*iang], ai2=csarr[2*iang+1];
      for (size_t ik=0; ik<idl1; ++ik)
        {
        C2(ik,l ) += ar1*CH2(ik,j )+ar2*CH2(ik,j +1);
        C2(ik,lc) += ai1*CH2(ik,jc)+ai2*CH2(ik,jc-1);
        }
      }
    for(; j<ipph; ++j,--jc)
      {
      iang+=l; if(iang>ip) iang-=ip;
      float war=csarr[2*iang], wai=csarr[2*iang+1];
      for (size_t ik=0; ik<idl1; ++ik)
        {
        C2(ik,l ) += war*CH2(ik,j );
        C2(ik,lc) += wai*CH2(ik,jc);
        }
      }
    }
  for (size_t j=1; j<ipph; ++j)
    for (size_t ik=0; ik<idl1; ++ik)
      CH2(ik,0) += CH2(ik,j);
  for (size_t j=1, jc=ip-1; j<ipph; ++j,--jc)   // 124
    for (size_t k=0; k<l1; ++k)
      {
      CH(0,k,j ) = C1(0,k,j)-C1(0,k,jc);
      CH(0,k,jc) = C1(0,k,j)+C1(0,k,jc);
      }

  if (ido==1) return;

  for (size_t j=1, jc=ip-1; j<ipph; ++j, --jc)  // 127
    for (size_t k=0; k<l1; ++k)
      for (size_t i=1; i<=ido-2; i+=2)
        {
        CH(i  ,k,j ) = C1(i  ,k,j)-C1(i+1,k,jc);
        CH(i  ,k,jc) = C1(i  ,k,j)+C1(i+1,k,jc);
        CH(i+1,k,j ) = C1(i+1,k,j)+C1(i  ,k,jc);
        CH(i+1,k,jc) = C1(i+1,k,j)-C1(i  ,k,jc);
        }

// All in CH

  for (size_t j=1; j<ip; ++j)
    {
    size_t is = (j-1)*(ido-1);
    for (size_t k=0; k<l1; ++k)
      {
      size_t idij = is;
      for (size_t i=1; i<=ido-2; i+=2)
        {
        float t1=CH(i,k,j), t2=CH(i+1,k,j);
        CH(i  ,k,j) = wa[idij]*t1-wa[idij+1]*t2;
        CH(i+1,k,j) = wa[idij]*t2+wa[idij+1]*t1;
        idij+=2;
        }
      }
    }
  }
#undef C1
#undef C2
#undef CH2

#undef CC
#undef CH
#undef PM
#undef MULPM
#undef WA

static void copy_and_norm(float *c, float *p1, size_t n, float fct)
  {
  if (p1!=c)
    {
    if (fct!=1.)
      for (size_t i=0; i<n; ++i)
        c[i] = fct*p1[i];
    else
      memcpy (c,p1,n*sizeof(float));
    }
  else
    if (fct!=1.)
      for (size_t i=0; i<n; ++i)
        c[i] *= fct;
  }

void pk_rfft_forward(const rfftp_plan_i *plan, float c[], float ch[], float fct)
  {
  if (plan->length==1) return;
  size_t n=plan->length;
  size_t l1=n, nf=plan->nfct;
  float *p1=c, *p2=ch;

  for(size_t k1=0; k1<nf;++k1)
    {
    size_t k=nf-k1-1;
    size_t ip=plan->fct[k].fct;
    size_t ido=n / l1;
    l1 /= ip;
    if (plan->fct[k].fn)                     /* 2 3 4 5 */
      plan->fct[k].fn(ido, l1, p1, p2, plan->fct[k].tw);
    else
      {
      plan->fct[k].fng(ido, ip, l1, p1, p2, plan->fct[k].tw, plan->fct[k].tws);
      SWAP (p1,p2,float *);
      }
    SWAP (p1,p2,float *);
    }
  copy_and_norm(c,p1,n,fct);
  }

void pk_rfft_backward(const rfftp_plan_i *plan, float c[], float ch[], float fct)
  {
  if (plan->length==1) return;
  size_t n=plan->length;
  size_t l1=1, nf=plan->nfct;
  float *p1=c, *p2=ch;

  for(size_t k=0; k<nf; k++)
    {
    size_t ip = plan->fct[k].fct,
           ido= n/(ip*l1);
    if (plan->fct[k].fn)                     /* 2 3 4 5 */
      plan->fct[k].fn(ido, l1, p1, p2, plan->fct[k].tw);
    else
      plan->fct[k].fng(ido, ip, l1, p1, p2, plan->fct[k].tw, plan->fct[k].tws);
    SWAP (p1,p2,float *);
    l1*=ip;
    }
  copy_and_norm(c,p1,n,fct);
  }
