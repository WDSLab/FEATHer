"""Fast screening loop for FEATHer architecture arms.

Why a second protocol exists
----------------------------
The shipping protocol (batch_size=32) costs ~10 min per run on a 4090 and is
dominated by CUDA launch overhead, not arithmetic: a 822-parameter model spends
94% of an epoch in kernel launches for 760 tiny batches. Raising the batch to
512 cuts an epoch 7.4x with the same samples seen. Measured calibration on
Steel H=96 (2026-08-26):

    canonical, shipping (bs 32,  lr 1e-3, 3 seeds)   MSE 0.4721   seed std 1.40%
    canonical, dev      (bs 512, lr 8e-3, 3 seeds)   MSE 0.4696   seed std 0.35%
    redA delta, shipping   -2.67% (1 seed)
    redA delta, dev        -1.76% (3 seeds)

so the dev protocol lands within 0.5% of the shipping number, reproduces the
sign and rough size of a known delta, and has a 4x LOWER noise floor (the
larger batch removes gradient noise). It is a ranking tool: promote winners to
the shipping protocol before believing an absolute number.

lr is swept per arm and selected on VAL, never test, because the dev batch size
sits near a cliff (Steel H96: lr 4e-3 -> 0.465, lr 3e-3 -> 0.517) and a fixed lr
would penalise arms whose gradient scale differs from the canonical model's.

Usage
-----
    python tools/dev/run_dev.py --arms canon,redA,gatestatic
    python tools/dev/run_dev.py --arms all --data Steel,BlastFurnace --pred_len 96,336
    python tools/dev/dev_summary.py --prefix dev --ref canon
"""
import argparse
import itertools
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# arm name -> extra CLI flags appended to the canonical FEATHer config.
# The canonical mfg config (_FEATHER_MFG_N: mix / residual r16 / d_state 16 /
# kernel 7 / 4 bands) is what an empty string means.
ARMS = {
    "canon":       "",

    # --- redundancy: the whole band bank + gate + DTK as ONE real filter -----
    # 2026-08-25 screening found this ties the canonical model on mfg and LTSF
    # at -30% parameters. Re-run here with seeds because the original was n=1.
    "redA":        "--num_bands 1 --point_kernel 7 --dtk_mode none",
    "redA_s8":     "--num_bands 1 --point_kernel 7 --dtk_mode none --d_state 8",
    "redA_r8":     "--num_bands 1 --point_kernel 7 --dtk_mode none --mlp_rank 8",
    "redA_k11":    "--num_bands 1 --point_kernel 11 --dtk_mode none",

    # --- the FFT gate is a learned constant (2026-08-14) -> price removing it -
    # gate_mode=static keeps the learned band mixture but drops the transform,
    # which on the MCU also drops the twiddle table and the rfft buffers.
    "gatestatic":  "--gate_mode static",

    # --- nonlinearity: the one axis that has paid every time it was tried ----
    "dtkgelu":     "--dtk_act gelu",

    # --- channel axis: three options, measured against each other ----------
    #   canon      mix + in_proj + residual MLP   = TWO mixers (linear + nonlinear)
    #   noinproj   mix + residual MLP             = one nonlinear mixer, no D^2
    #   ci         channel-independent            = no mixing, D-INDEPENDENT params
    # ci is the one that was never re-measured on the current architecture: the
    # "mix beats CI by 5.5% on mfg" number comes from the 2026-08-05 ceiling
    # scan, which predates the residual MLP and the shipping config, and the
    # 08-24 ablation has no ci arm. Params (mfg-6 total): 12,154 / 10,584 /
    # 1,656 — and ci also lets an MCU port process one channel at a time, so
    # its working buffers are L*1 instead of L*D.
    "noinproj":    "--no_in_proj",
    "ci":          "--channel_mode ci_full",
    "ci_nores":    "--channel_mode ci_full --no_residual_mlp",

    # --- period: Steel/EWELD's real acf peak is 96, not the configured 24, and
    # P must divide both seq_len and pred_len. P=48 is the largest legal value
    # at H=96 and was never tried.
    "period48":    "--period 48",
    "period12":    "--period 12",

    # --- capacity knobs that cut parameters -------------------------------
    "s8":          "--d_state 8",
    "r8":          "--mlp_rank 8",
    "r4":          "--mlp_rank 4",

    # --- stacked reductions: the two arms that beat canonical with zero failed
    # seeds (redA -1.45% at -42% params, noinproj -1.56% at -5%) act on
    # different axes (time vs channel), so they should compose.
    "redA_nip":    "--num_bands 1 --point_kernel 7 --dtk_mode none --no_in_proj",
    "redA_nip_r8": "--num_bands 1 --point_kernel 7 --dtk_mode none --no_in_proj "
                   "--mlp_rank 8",
    "redA_nip_r4": "--num_bands 1 --point_kernel 7 --dtk_mode none --no_in_proj "
                   "--mlp_rank 4",

    # --- head nonlinearity on top of the reduced model ---------------------
    # Once redA collapses the band bank and the DTK, the residual MLP is the
    # model's only nonlinearity and the SPK head is a plain linear map. Every
    # nonlinearity added to this family has paid; a hidden layer in the
    # cross-period backbone costs n*h + h + h*m + m (24/60/132 params at
    # h=4/8/16), which is the cheapest place left to put one.
    "redA_nip_spk4":  "--num_bands 1 --point_kernel 7 --dtk_mode none "
                      "--no_in_proj --spk_hidden 4",
    "redA_nip_spk8":  "--num_bands 1 --point_kernel 7 --dtk_mode none "
                      "--no_in_proj --spk_hidden 8",
    "redA_nip_spk16": "--num_bands 1 --point_kernel 7 --dtk_mode none "
                      "--no_in_proj --spk_hidden 16",
    # trade the channel-axis rank for head nonlinearity at equal parameters
    "redA_nip_r4_spk8": "--num_bands 1 --point_kernel 7 --dtk_mode none "
                        "--no_in_proj --mlp_rank 4 --spk_hidden 8",

    # --- REVIVE the band bank instead of removing it (2026-08-27) -----------
    # The band bank measured inert because two independent defects cancel any
    # effect it could have, and every past attempt fixed one at a time:
    #   (1) the "global" gate is blind to spectral SHAPE — measured live, a slow
    #       and a fast sinusoid get gate weights identical to 4 decimals;
    #   (2) the band path is linear, so even a gate that DID move would only be
    #       re-mixing one collapsed filter.
    # Fixing (2) alone made things worse (dtkgelu +6.8% shipping); fixing (1)
    # alone did nothing (gate_mode=banded +1.1%). These arms fix them together.
    "bandse":        "--band_act se",
    "bandgelu":      "--band_act gelu",
    "gatebanded":    "--gate_mode banded",
    "revive":        "--band_act se --gate_mode banded",
    "revive_gelu":   "--band_act gelu --gate_mode banded",
    # does a revived bank actually want MORE bands? (flat until now by construction)
    "revive_b2":     "--band_act se --gate_mode banded --num_bands 2",
    "revive_b4":     "--band_act se --gate_mode banded --num_bands 4",

    # --- BAND-AXIS DECISION RUN (2026-08-31) --------------------------------
    # The channel axis is already settled on B (--no_in_proj): accuracy tie,
    # 5-18% fewer params, and EWELD failures 5/10 -> 0/10 (Fisher p=0.033).
    # So every band-axis candidate is stacked ON TOP of it, and "noinproj" is
    # the baseline they are measured against — not "canon".
    "nip_gatebanded": "--no_in_proj --gate_mode banded",
    "nip_revive":     "--no_in_proj --band_act se --gate_mode banded",

    # 2x2 를 채우는 두 칸 (2026-08-31). 지금까지는 대각선 두 끝만 돌렸다:
    #   noinproj = 밴드4 + 게이트 + DTK,  redA_nip = DWConv1 + DTK없음.
    # 가운데를 채워야 "밴드가 한 일"과 "DTK가 한 일"이 분리된다.
    # 가설: DTK 는 4개 밴드에 공유로 적용되므로 밴드들을 서로 비슷하게 만든다
    # -> 게이트가 할 일이 없어진다. DTK 를 빼면 밴드가 뚜렷해져 살아날 수 있다.
    "nip_dtknone":    "--no_in_proj --dtk_mode none",
    "nip_b1k7":       "--no_in_proj --num_bands 1 --point_kernel 7",

    # 밴드를 남기되 죽은 가지를 없앤 구성 (2026-08-31). nip_dtknone 은 DTK 를 빼면서
    # POINT 가 커널 1(= 채널별 스칼라, 시간축 필터 0)로 남아 EWELD/Steel 에서 시드
    # 실패가 되살아났다. 커널 사다리를 3 -> 5 -> 풀링(실효폭 ~8)로 올리면 모든 가지가
    # 실제로 필터링하면서 "미세 -> 거친" 순서도 지켜진다. (POINT 를 7 로 넓히는 것은
    # 순서를 뒤집어 다중스케일 개념을 깨므로 쓰지 않는다.)
    "nip_b3k3":       "--no_in_proj --num_bands 3 --point_kernel 3 --dtk_mode none",

    # 확정 후보(B) 위에서 게이트만 비활성 — 가중치를 1/S 로 고정한다(계산은 동일,
    # 적응성만 제거). "주파수 게이트가 정확도를 사는가"의 직접 대조군.
    # 기존 측정: gateuni +1.32%(08-24) · gatesm +0.21%(08-24) · gatebanded +0.05%(08-31)
    # → 어느 방향으로 건드려도 변화 없음. 확정 구성 위에서 다시 확인한다.
    "nip_b3k3_gateuni": "--no_in_proj --num_bands 3 --point_kernel 3 --dtk_mode none "
                        "--gate_uniform",

    # 확정 후보(B) 위에서 게이트를 "실제로 움직이는" 버전으로 교체 (2026-08-31 밤).
    # 08-27 에 gate_mode=banded + band_act=se 로 게이트가 실제로 움직임을 학습된
    # 체크포인트에서 확인했으나(변동폭 0.17~0.21, se 게인 zero-init 이탈 0.81),
    # 그 측정은 전부 **DTK 가 있고 밴드가 4개인 옛 구조** 위에서였다. DTK 는 모든
    # 밴드에 공유로 걸려 밴드들을 서로 비슷하게 만들므로 게이트가 할 일을 지운다.
    # DTK 를 제거한 지금 구조에서는 밴드가 더 뚜렷해져 게이트가 값을 할 수도 있다.
    "nip_b3k3_banded": "--no_in_proj --num_bands 3 --point_kernel 3 --dtk_mode none "
                       "--gate_mode banded",
    "nip_b3k3_revive": "--no_in_proj --num_bands 3 --point_kernel 3 --dtk_mode none "
                       "--gate_mode banded --band_act se",

    # 커널 사다리 폭 실험 (2026-08-31 밤). 지금까지의 "밴드는 기여 0" 결론은 전부
    # 사다리 {1,3,5,풀링} 하나만 재본 것이다. L=96 인데 수용영역이 전부 좁은 쪽에
    # 몰려 있어(최대/최소 2.7배) 밴드끼리 구별이 안 되고, 그러면 게이트가 고를
    # 대상이 없다. 옥타브 간격으로 넓히면 밴드가 실제로 다른 스케일을 잡는다.
    # (LOW=4배 풀링이라 실효폭 ~8. 사다리를 그 너머까지 늘리는 것이 요점.)
    "nip_b4_357":   "--no_in_proj --num_bands 4 --band_kernels 3,5,7 "
                    "--dtk_mode none --gate_mode banded",
    "nip_b4_3715":  "--no_in_proj --num_bands 4 --band_kernels 3,7,15 "
                    "--dtk_mode none --gate_mode banded",
    "nip_b4_31131": "--no_in_proj --num_bands 4 --band_kernels 3,11,31 "
                    "--dtk_mode none --gate_mode banded",

    # ★ 대조군 (2026-08-31 밤). 넓은 사다리가 이겼는데(−2.06%), 그게 "다중 스케일"
    # 때문인지 "그냥 수용영역이 커서"인지 구분이 안 된다. 밴드 없이 넓은 conv 하나만
    # 쓰는 arm 이 같은 이득을 내면 밴드 이야기는 성립하지 않는다.
    # 리뷰어의 첫 질문("큰 커널 하나만 써봤나?")에 대한 답이기도 하다.
    "nip_b4_31563": "--no_in_proj --num_bands 4 --band_kernels 3,15,63 "
                    "--dtk_mode none --gate_mode banded",
    "nip_b1k15":    "--no_in_proj --num_bands 1 --point_kernel 15 --dtk_mode none",
    "nip_b1k31":    "--no_in_proj --num_bands 1 --point_kernel 31 --dtk_mode none",

    # LTSF 를 mfg 와 같은 채널모드(mix)로 맞춘 버전 (2026-08-31 밤). ETTh1/ETTh2 는
    # _DATASET_OVERRIDES 가 ci_full 을 주므로, 그대로 돌리면 "데이터셋 차이"와
    # "채널모드 차이"가 섞인다. --channel_mode mix 로 고정해 사다리 효과만 분리한다.
    "mix_b3k3_banded": "--no_in_proj --num_bands 3 --point_kernel 3 --dtk_mode none "
                       "--gate_mode banded --channel_mode mix",
    "mix_b4_3715":     "--no_in_proj --num_bands 4 --band_kernels 3,7,15 "
                       "--dtk_mode none --gate_mode banded --channel_mode mix",
    "mix_b4_31131":    "--no_in_proj --num_bands 4 --band_kernels 3,11,31 "
                       "--dtk_mode none --gate_mode banded --channel_mode mix",
    "mix_b1k31":       "--no_in_proj --num_bands 1 --point_kernel 31 --dtk_mode none "
                       "--channel_mode mix",

    # 넓은 사다리 위에서의 게이트 ablation (2026-08-31 밤). 좁은 사다리({3,5,pool})
    # 위에서는 게이트가 -0.45% 였다. 사다리를 {3,11,31}로 넓혀 밴드가 서로 뚜렷해진
    # 지금은 게이트가 고를 대상이 실제로 생겼으므로 역할이 커졌을 수 있다.
    # gateuni = 가중치를 1/S 로 고정(계산 동일, 적응성만 제거) = 순수 대조군.
    "wide_banded":  "--no_in_proj --num_bands 4 --band_kernels 3,11,31 "
                    "--dtk_mode none --gate_mode banded",
    "wide_gateuni": "--no_in_proj --num_bands 4 --band_kernels 3,11,31 "
                    "--dtk_mode none --gate_uniform",
    "wide_global":  "--no_in_proj --num_bands 4 --band_kernels 3,11,31 "
                    "--dtk_mode none",

    # 팽창으로 같은 수용영역을 더 싸게 (2026-08-31 밤). 수용영역 = d*(k-1)+1 이므로
    # k=3,d=15 는 탭 3개로 수용영역 31 을 만든다. 커널 넓힘 {3,11,31}(688p) 대비
    # 팽창 {3,9,31}(472p)은 conv 파라미터가 288 -> 72 로 줄어든다(-31% 총합).
    # ⚠ 탭 사이에 구멍이 생기므로 실제로 같은 성능을 내는지는 재봐야 안다.
    "dil_3_9_31":  "--no_in_proj --num_bands 4 --band_kernels 3,3,3 "
                   "--band_dilations 1,4,15 --dtk_mode none --gate_mode banded",
    "dil_3_7_15":  "--no_in_proj --num_bands 4 --band_kernels 3,3,3 "
                   "--band_dilations 1,3,7 --dtk_mode none --gate_mode banded",
    "dil_mix":     "--no_in_proj --num_bands 4 --band_kernels 3,5,5 "
                   "--band_dilations 1,2,7 --dtk_mode none --gate_mode banded",

    # 팽창 사다리 위에서의 게이트 대조군 (LTSF 용). mfg 에서는 사다리 {3,11,31} 위에서
    # 밴드별 게이트가 -0.73% 였는데, LTSF 에서는 게이트 ablation 을 한 번도 안 했다.
    "dil_gateuni": "--no_in_proj --num_bands 4 --band_kernels 3,3,3 "
                   "--band_dilations 1,4,15 --dtk_mode none --gate_uniform",
    "dil_global":  "--no_in_proj --num_bands 4 --band_kernels 3,3,3 "
                   "--band_dilations 1,4,15 --dtk_mode none",

    # LTSF per-dataset 서치 (2026-09-02). mfg 는 "설정 하나" 프로토콜이라 못 하지만
    # LTSF 는 베이스라인이 전부 per-dataset 튜닝이라 대칭을 위해 허용된다(08-14).
    # DTK 를 뺀 뒤 살아있는 축은 num_bands 와 point_kernel 뿐이다
    # (d_state/kernel_size 는 DTK 전용이라 죽음). val_loss 로만 선택할 것.
    "ls_b2k3": "--no_in_proj --num_bands 2 --point_kernel 3 --dtk_mode none --gate_mode banded",
    "ls_b2k5": "--no_in_proj --num_bands 2 --point_kernel 5 --dtk_mode none --gate_mode banded",
    "ls_b3k3": "--no_in_proj --num_bands 3 --point_kernel 3 --dtk_mode none --gate_mode banded",
    "ls_b3k5": "--no_in_proj --num_bands 3 --point_kernel 5 --dtk_mode none --gate_mode banded",
    "ls_b4k3": "--no_in_proj --num_bands 4 --point_kernel 3 --dtk_mode none --gate_mode banded",
    "ls_b4k5": "--no_in_proj --num_bands 4 --point_kernel 5 --dtk_mode none --gate_mode banded",

    # ↑ 위 ls_* 격자는 잘못 짰다: HIGH=3 / MID=5 가 코드에 박혀 있어 point_kernel 만
    # 바꾸면 커널이 중복된다(b3k5 = 5,5 / b4k3 = 3,3,5). 수용영역 사다리를 통제하는
    # 실험이 못 되므로 --band_kernels 로 다시 짠다. 아래가 유효한 격자.
    "lk_b2_3":   "--no_in_proj --num_bands 2 --band_kernels 3 --dtk_mode none --gate_mode banded",
    "lk_b2_5":   "--no_in_proj --num_bands 2 --band_kernels 5 --dtk_mode none --gate_mode banded",
    "lk_b3_35":  "--no_in_proj --num_bands 3 --band_kernels 3,5 --dtk_mode none --gate_mode banded",
    "lk_b3_37":  "--no_in_proj --num_bands 3 --band_kernels 3,7 --dtk_mode none --gate_mode banded",
    "lk_b4_357": "--no_in_proj --num_bands 4 --band_kernels 3,5,7 --dtk_mode none --gate_mode banded",
    "lk_b4_359": "--no_in_proj --num_bands 4 --band_kernels 3,5,9 --dtk_mode none --gate_mode banded",

    # Steel 나쁜 분지의 원인 진단 (2026-09-03). 확정 구성(lk_b4_357)에서 부품을
    # 하나씩 끊어 실패율이 어디서 사라지는지 본다. 20시드로 돌릴 것.
    #   base   = 확정 구성 그대로
    #   nores  = 잔차 MLP 제거      (모델의 유일한 비선형 + 채널믹서)
    #   norevin= RevIN 제거          (half-RevIN 의 1/stdev 폭발 가설)
    #   gateuni= 게이트 고정 1/4     (softmax 포화 가설)
    "dx_base":    "--no_in_proj --num_bands 4 --band_kernels 3,5,7 --dtk_mode none --gate_mode banded",
    "dx_nores":   "--no_in_proj --num_bands 4 --band_kernels 3,5,7 --dtk_mode none --gate_mode banded "
                  "--no_residual_mlp",
    "dx_norevin": "--no_in_proj --num_bands 4 --band_kernels 3,5,7 --dtk_mode none --gate_mode banded "
                  "--no_revin",
    "dx_gateuni": "--no_in_proj --num_bands 4 --band_kernels 3,5,7 --dtk_mode none --gate_uniform",

    # --- REAL band splits: the matched-frequency control (2026-08-27) --------
    # The first revival round kept the learned conv bank, so a flat result could
    # still be answered with "those were not real frequency bands". These modes
    # replace the bank with splits that are disjoint by construction, reconstruct
    # the input exactly, and cost NEGATIVE parameters (822 -> 750 at D=6/4 bands
    # because the learned filters disappear):
    #   diff  Laplacian differences of moving averages (cross-corr 0.66)
    #   fft   exact rFFT slices           (cross-corr 0.0009 = orthogonal)
    "banddiff":      "--band_mode diff",
    "bandfft":       "--band_mode fft",
    # ... and the same with the collapse + gate fixes, which is the strongest
    # form of the original FEATHer thesis that can be built.
    "fft_revive":    "--band_mode fft  --band_act se --gate_mode banded",
    "diff_revive":   "--band_mode diff --band_act se --gate_mode banded",
    "fft_revive_b4": "--band_mode fft  --band_act se --gate_mode banded --num_bands 4",
    "fft_revive_b2": "--band_mode fft  --band_act se --gate_mode banded --num_bands 2",

    # --- octave-spaced bands: the equal-width split was the wrong split ------
    # Measured on training windows (instance-normalized, L=96): equal-width
    # quarters give the lowest slice 81.7% of the energy on ETTh1 and 87.4% on
    # Steel, so "four bands" was one real band plus three nearly empty ones.
    # Halving the bin index at each step balances them to 64/17/10/8 and
    # 71/17/8/4 — the standard wavelet / mel-spacing remedy.
    # --- spectral-separation loss (lambda_spec) ----------------------------
    # This is the term that PUSHES each band conv to own its frequency slice
    # (utils/losses.spectral_separation_loss_scales). It has been called "flat"
    # since 2026-07-06, but that measurement ran under the band-order bug fixed
    # in the 07-14 audit (P1-1) and was explicitly declared confounded; the
    # 08-24 re-run left the specHi arm unfinished. So it is currently UNTESTED.
    #
    # The collapse argument makes a falsifiable prediction: while the band path
    # is linear the term must be inert (pushing bands apart cannot change their
    # weighted sum), but with the collapse broken it should start to matter.
    # Hence each lambda is run twice, with and without band_act=se.
    # --- gate initialisation (2026-08-27 root-cause candidate) --------------
    # The canonical "global" gate is randomly initialised and its softmax is
    # already saturated before training: on EWELD the initial band mixture runs
    # [0.0002, 0.9954, 0.0002, 0.0042] (max/min 5,077). The ratio tracks the
    # outcome exactly — the two seeds that fail on EWELD start at 5,077x and
    # 1,795x, the one that succeeds at 17x — and the failing runs are worse on
    # TRAIN loss too, i.e. it is an optimisation failure decided at init, with
    # near-zero bands acting as dead branches. gatezero starts every band at
    # 1/S while keeping the canonical gate structure and parameter count.
    "gatezero":      "--gate_zero_init",

    "spec0":         "--lambda_spec 0.0",
    "specHi":        "--lambda_spec 0.1",
    "specVHi":       "--lambda_spec 1.0",
    "se_spec0":      "--band_act se --lambda_spec 0.0",
    "se_spec":       "--band_act se --lambda_spec 0.01",
    "se_specHi":     "--band_act se --lambda_spec 0.1",
    "se_specVHi":    "--band_act se --lambda_spec 1.0",

    "bandoct":       "--band_mode fftoct",
    "oct_revive":    "--band_mode fftoct --band_act se --gate_mode banded",
    "oct_revive_b2": "--band_mode fftoct --band_act se --gate_mode banded --num_bands 2",
    "oct_revive_b4": "--band_mode fftoct --band_act se --gate_mode banded --num_bands 4",
    "oct_se":        "--band_mode fftoct --band_act se",

    # 믹서 경량화 스크린 (2026-09-14). mfg 데이터에서는 _FEATHER_MFG_N 이 확정 mix
    # 모델을 이미 적용하므로 빈 문자열 = 확정 모델이다. 판정 지표는 "섞기 이득을
    # 몇 % 지키냐" = (ci - arm) / (ci - base).
    #   tie        W_up = W_down^T (아이디어 1: 가중치 절반, 연산 동일)
    #   r4 / r8    공유 요약 개수만 줄이기 (이미 있는 옵션, 참고용)
    #   in1        믹서 입력을 밴드 출력 -> 정규화 입력으로만 바꿈 (풀링 효과와 분리)
    #   in4/12/96  입력을 시간축으로 p 칸씩 평균 내서 섞고 보정값을 보간 (아이디어 2).
    #              MCU 에서 L x D hidden state 를 없앨 수 있는 형태.
    "mx_base": "",
    "mx_ci":   "--channel_mode ci_full",
    "mx_tie":  "--mlp_tie",
    "mx_r4":   "--mlp_rank 4",
    "mx_r8":   "--mlp_rank 8",
    "mx_in1":  "--mix_mode input --mix_pool 1",
    "mx_in4":  "--mix_mode input --mix_pool 4",
    "mx_in12": "--mix_mode input --mix_pool 12",
    "mx_in96": "--mix_mode input --mix_pool 96",
    # in1 의 공유 요약 크기 R 을 줄이기 (2026-09-14). in1 은 MCU 에서 L x R 만 들고 있으면
    # 되므로 R 이 곧 믹서의 추가 SRAM 이다 (R=4 -> 1.5KB, R=2 -> 0.8KB, 센서 수 무관).
    "mx_in1_r4": "--mix_mode input --mix_pool 1 --mlp_rank 4",
    "mx_in1_r2": "--mix_mode input --mix_pool 1 --mlp_rank 2",
    # 게이트 교환 실험(09-15)에서 창마다 달라지는 게이트의 효과가 1% 안팎이었음 ->
    # 입력과 무관한 학습된 고정 비율로 처음부터 학습해 비교한다. static_rng 는
    # banded 와 난수 소비를 맞춰 SPK 헤드 초기값이 같다(차이 = 게이트만).
    "mx_in1_r4_static": "--mix_mode input --mix_pool 1 --mlp_rank 4 --gate_mode static_rng",
    # FFT 게이트를 유지하되 센서마다 자기 스펙트럼으로 비율을 정한다(09-15). 파라미터·
    # 초기값은 banded 와 같다. 칩에서는 센서 d 를 처리할 때 게이트도 같이 계산되므로
    # 센서 전체를 먼저 훑을 필요가 없다(정규화 1번).
    "mx_in1_r4_gsensor": "--mix_mode input --mix_pool 1 --mlp_rank 4 --gate_mode banded_sensor",
    # 2026-09-18: the candidate with POINT/MID taps summing to 0 (band_check found
    # the learned bank only "mostly" separated; this separates it by construction).
    "mx_in1_r4_gsensor_zs": "--mix_mode input --mix_pool 1 --mlp_rank 4 --gate_mode banded_sensor "
                            "--band_zero_sum",
    # ...and the same with the FFT gate replaced by fixed learned ratios: does the
    # gate start to matter once the bands are actually different?
    # 2026-09-18 FFT-gate redesigns, each with its input-independent control
    # (static band mixture + static G2/G4 weight): the gate earns its place
    # only if *_fft beats *_static.
    "g2_fft":    "--mix_mode input --mix_pool 1 --mlp_rank 4 --gate_mode banded_sensor --gate_mix fft",
    "g2_static": "--mix_mode input --mix_pool 1 --mlp_rank 4 --gate_mode static_rng --gate_mix static",
    "g4_fft":    "--mix_mode input --mix_pool 1 --mlp_rank 4 --gate_mode banded_sensor --head2 fft",
    "g4_static": "--mix_mode input --mix_pool 1 --mlp_rank 4 --gate_mode static_rng --head2 static",
    "mx_in1_r4_static_zs": "--mix_mode input --mix_pool 1 --mlp_rank 4 --gate_mode static_rng "
                           "--band_zero_sum",
    # 게이트 필요성 확인 2단계(09-16): 가중치 없이 세 밴드를 그냥 더한다. none_rng 는
    # static_rng 처럼 banded 와 난수 소비를 맞춰 SPK 헤드 초기값이 같다.
    "mx_in1_r4_gnone": "--mix_mode input --mix_pool 1 --mlp_rank 4 --gate_mode none_rng",
    # 09-21 old-paper-style gate ablation: uniform 1/S weights (gate built, then ignored)
    # 09-21 mixing diagnostics: the candidate with the mixer turned off (same seeds, ckpts kept)
    "mx_in1_r4_gsensor_ci": "--channel_mode ci_full --mix_mode input --mix_pool 1 --mlp_rank 4 --gate_mode banded_sensor",
    # 09-21 light-mixer fixes (A listen / B lag / C zero-init) on the candidate
    "mxA": "--mix_mode input --mix_pool 1 --mlp_rank 4 --gate_mode banded_sensor --mix_listen",
    "mxB": "--mix_mode input --mix_pool 1 --mlp_rank 4 --gate_mode banded_sensor --mix_lag 9,4",
    "mxC": "--mix_mode input --mix_pool 1 --mlp_rank 4 --gate_mode banded_sensor --mix_zero_init",
    "mxABC": "--mix_mode input --mix_pool 1 --mlp_rank 4 --gate_mode banded_sensor --mix_listen --mix_lag 9,4 --mix_zero_init",
    # 09-22 test-free re-decision of input_raw: train on the first 70 % of the train split
    # so VAL sits about as far ahead of training as TEST normally does (read VAL only).
    "gap_cand": "--mix_mode input --mix_pool 1 --mlp_rank 4 --gate_mode banded_sensor --train_frac 0.7 --train_frac_mode first",
    "gap_raw": "--mix_mode input_raw --mix_pool 1 --mlp_rank 4 --gate_mode banded_sensor --train_frac 0.7 --train_frac_mode first",
    "mx_in1_r4_guni": "--mix_mode input --mix_pool 1 --mlp_rank 4 --gate_mode banded_sensor --gate_uniform",
    # 정규화 1번(09-16): 후보(센서별 게이트)와 같고, 요약 u 만 정규화 전 입력으로 만든다.
    # 칩에서 요약이 원래 입력 창만 읽으므로 센서마다 정규화를 한 번만 하면 된다.
    "mx_in1_r4_gsensor_raw": "--mix_mode input_raw --mix_pool 1 --mlp_rank 4 --gate_mode banded_sensor",
    # 위 센서별 게이트에서 구간 경계만 바꾼다(09-15, A안): 같은 폭(0-15/16-31/32-48)
    # -> 옥타브(0-12/13-24/25-48). LOW 구간을 AvgPool4 브랜치가 담을 수 있는 범위에 맞춤.
    "mx_in1_r4_gsensor_oct": "--mix_mode input --mix_pool 1 --mlp_rank 4 "
                             "--gate_mode banded_sensor --gate_bounds octave",
    # B안(09-15): 학습 데이터 평균 스펙트럼의 에너지가 1/3 씩 들어가도록 경계를 워커가
    # 학습 split 에서만 계산한다(예: Steel 2,6 / EWELD 2,4 / WindSCADA 2,5 / BF 3,9).
    "mx_in1_r4_gsensor_ter": "--mix_mode input --mix_pool 1 --mlp_rank 4 "
                             "--gate_mode banded_sensor --gate_bounds tercile",
    # 경계를 센서마다 따로 (09-16): 센서별 평균 스펙트럼으로 각자 에너지 3등분.
    # 학습 파라미터·작업 메모리는 그대로, 상수만 센서당 2개.
    "mx_in1_r4_gsensor_ters": "--mix_mode input --mix_pool 1 --mlp_rank 4 "
                              "--gate_mode banded_sensor --gate_bounds tercile_sensor",

    # 같은 비교를 LTSF 에서 (2026-09-15). LTSF 는 _FEATHER_LTSF_CI 가 ci_full 을 주므로
    # 빈 문자열 = CI 이고, mix 계열은 --channel_mode mix 를 명시해야 한다
    # (mx_* 를 LTSF 에 그대로 쓰면 mx_base 가 CI 가 된다).
    # 질문: in1 R=4 가 LTSF 에서도 CI 를 대신할 수 있나.
    "lt_ci":     "",
    "lt_mix":    "--channel_mode mix",
    # 09-23 frozen model on LTSF: CI with the summary cut to 4 (hidden mixing)
    "lt_ci_r4":  "--mlp_rank 4",
    "lt_in1":    "--channel_mode mix --mix_mode input --mix_pool 1",
    "lt_in1_r4": "--channel_mode mix --mix_mode input --mix_pool 1 --mlp_rank 4",
    # (09-16) mfg 후보(in1 R=4 + 센서별 게이트)를 LTSF 에서도 확인. 두 변종 모두:
    #   lt_in1_r4_gsensor  섞는 버전의 새 설정
    #   lt_ci_in1_r4       CI 를 같은 설정으로 통일한 버전 (기존 lt_ci 와 비교; CI 에서는
    #                      센서별 게이트 = 기존 게이트지만 R=4 와 입력 섞기는 계산이 달라짐)
    "lt_in1_r4_gsensor": "--channel_mode mix --mix_mode input --mix_pool 1 --mlp_rank 4 "
                         "--gate_mode banded_sensor",
    "lt_ci_in1_r4":      "--channel_mode ci_full --mix_mode input --mix_pool 1 --mlp_rank 4 "
                         "--gate_mode banded_sensor",

    # 입력 길이(룩백) 확인 (2026-10-01). 동결 모델(canon = mfg mix, R4) 을 L 만 바꿔 돌린다.
    # 베이스라인 룩백(results/_archive/baselines/fcst_results_baselines.csv, exp_tag lookback)
    # 은 L96->192 에서 5~6%, L96->336 에서 9~11% 좋아졌다 -> FEATHer 도 그만큼 좋아지는지.
    # L720 은 안 함: 7채널이면 입력 버퍼만 20,160 B + 스택 552 B > SRAM 20,480 B (모든 모델 불가).
    # 데이터별 주기(24/48) 는 192·336 을 나누고, 시험 구간의 예측 목표 시점은 L 과 무관.
    "L96":  "",
    "L192": "--seq_len 192",
    "L336": "--seq_len 336",
    # 같은 L 에서 섞기를 끈 것 (2026-10-02). Han et al. TKDE 2024: "입력이 길면 CI 는 좋아지고
    # CD 는 나빠질 수 있다" -> mix 의 이득(L96 +5.3%)이 L192·336 에서도 남는지. CLI 의
    # --channel_mode 가 _DATASET_OVERRIDES 의 mix 를 덮어쓴다(run_forecast.py cli_switches).
    "L96ci":  "--channel_mode ci_full",
    "L192ci": "--seq_len 192 --channel_mode ci_full",
    "L336ci": "--seq_len 336 --channel_mode ci_full",
    # 근거리/원거리 헤드 (2026-10-04, 아이디어 A). 느린 밴드만으로 H 전체를 1/4 해상도로 예측해
    # 보간 + 처음 96칸은 전 밴드 SPK 로 자세히. 사전 판정: H336·H720 평균 ≥2% 개선, 데이터 ≥5/6,
    # H96 악화 ≤0.5%, 칩당 ≥42채널 (lr 은 val 로 선택). 기준선 = 같은 CSV 의 L96 arm.
    "nf4": "--far_coarse 4 --near_len 96",
    # 공유 인자 예측 (2026-10-06, Fable 제안 P1, 사용자 승인). "센서는 각자 예측하고, 공정은 하나로
    # 예측한다": 섞기 MLP 를 R=4 인자 모델로 바꿔 인자 궤적을 따로 한 번 예측해 나눠 준다.
    #   p1 = 인자를 게이트 거친 밴드에서 / p2 = 느린(풀링) 밴드에서만 / p3 = 밴드·FFT 게이트 제거.
    # 사전 판정 규칙은 manuscript/drafts/p1_factor_plan.md (결과 보기 전에 작성). 기준선 = L96.
    "p1": "--factor_head 1",
    "p2": "--factor_head 1 --factor_src low",
    "p3": "--factor_head 1 --front none",
}

DEFAULT_ARMS = ["canon", "redA", "gatestatic", "dtkgelu", "noinproj",
                "period48", "s8", "r8"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arms", default=",".join(DEFAULT_ARMS),
                   help="comma list of arm names, or 'all'")
    p.add_argument("--data", default="Steel")
    p.add_argument("--pred_len", default="96")
    p.add_argument("--lrs", default="4e-3,8e-3")
    p.add_argument("--num_seeds", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=512)
    # A dev epoch is only 47 optimizer steps at bs=512 (vs 760 at bs=32), so the
    # first suspicion about the bad basin (see README) was undertraining. It is
    # not: at 200 epochs / patience 30 the failing seeds still land there and
    # still early-stop at epoch 12-67. So the default stays at the cheap 50/10
    # and the knobs remain available for a targeted re-check.
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--csv", default="results/dev/dev_loop.csv")
    p.add_argument("--prefix", default="dev")
    p.add_argument("--save_model", action="store_true",
                   help="Keep checkpoints, so the trained weights can be "
                        "inspected afterwards (e.g. did the gate actually "
                        "learn to move, did the band gains leave zero-init).")
    p.add_argument("--jobs", type=int, default=6,
                   help="concurrent worker processes on the single GPU")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--dry_run", action="store_true")
    a = p.parse_args()

    names = list(ARMS) if a.arms == "all" else [s.strip() for s in a.arms.split(",")]
    unknown = [n for n in names if n not in ARMS]
    if unknown:
        raise SystemExit("unknown arm(s): %s\nknown: %s"
                         % (", ".join(unknown), ", ".join(sorted(ARMS))))

    # The orchestrator takes a comma list for --data but only a SINGLE
    # --pred_len, so horizons are fanned out into separate jobs here (which
    # also parallelises them). The exp_tag stays shared: the CSV carries
    # data/pred_len columns and dev_summary filters on them.
    jobs = []
    horizons = [h.strip() for h in str(a.pred_len).split(",")]
    for arm, lr, H in itertools.product(names, a.lrs.split(","), horizons):
        tag = "%s_%s_%s" % (a.prefix, arm, lr)
        cmd = [a.python, "run_forecast.py", "--model", "FEATHer",
               "--data", a.data, "--pred_len", H,
               "--num_seeds", str(a.num_seeds), "--exp_tag", tag,
               "--batch_size", str(a.batch_size), "--lr", lr,
               "--num_epochs", str(a.num_epochs), "--patience", str(a.patience),
               "--results_csv", a.csv]
        if a.save_model:
            cmd += ["--save_model"]
        if ARMS[arm]:
            cmd += ARMS[arm].split()
        # log name carries the horizon so same-tag jobs don't clobber each other
        jobs.append(("%s_h%s" % (tag, H), cmd))

    n_runs = len(jobs) * a.num_seeds * len(a.data.split(","))
    print("%d job(s) = %d run(s); %d concurrent" % (len(jobs), n_runs, a.jobs))
    if a.dry_run:
        for tag, cmd in jobs:
            print(" ", tag, "|", " ".join(cmd[2:]))
        return

    t0 = time.time()
    logdir = os.path.join(REPO, "results", "_devlogs")
    os.makedirs(logdir, exist_ok=True)
    running, failed = [], []
    queue = list(jobs)
    while queue or running:
        while queue and len(running) < a.jobs:
            tag, cmd = queue.pop(0)
            fh = open(os.path.join(logdir, tag + ".log"), "w", encoding="utf-8")
            running.append((tag, subprocess.Popen(cmd, cwd=REPO, stdout=fh,
                                                  stderr=subprocess.STDOUT), fh))
            print("  [start] %s  (%d left)" % (tag, len(queue)), flush=True)
        time.sleep(2)
        for item in running[:]:
            tag, proc, fh = item
            if proc.poll() is not None:
                fh.close()
                running.remove(item)
                if proc.returncode != 0:
                    failed.append(tag)
                    print("  [FAIL ] %s  rc=%d  -> results/_devlogs/%s.log"
                          % (tag, proc.returncode, tag), flush=True)
                else:
                    print("  [done ] %s" % tag, flush=True)

    print("\nwall %.1f min%s" % ((time.time() - t0) / 60,
                                 "" if not failed else "   FAILED: " + ",".join(failed)))
    print("next: python tools/dev/dev_summary.py --csv %s --prefix %s --ref canon"
          % (a.csv, a.prefix))


if __name__ == "__main__":
    main()
