# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FEATHer (Fourier-Efficient Adaptive Temporal Hierarchy Forecaster) is an
ultra-lightweight model for long-term time-series forecasting under sub-1K
parameter / edge-MCU constraints. It combines multi-scale frequency
decomposition with a shared temporal kernel and a period-aware sparse
forecasting head.

## Current status (2026-07-14e)

**?¡âš¡?¡âš¡?¡âš¡??2026-08-14 (ìµœì¢… HP ?•ì • + ë°°í¬ ?„ë ˆ?´ë° ?•ë¦½ + mfg final ?„ë£Œ;
?ì„¸ = `memory/project_final_hp_framing_0814.md`):**
- **??ëª¨ë¸ ?¨ë?ë¦??•ì • = 2ë³€ì¢?ì±„ë„ëª¨ë“œ ?¤ìœ„ì¹?**: **mfg = FEATHer-N**(channel_mode `mix`
  + ?”ì°¨MLP rank16, canonical **`ds16_k7_B4`** = d_state16/kernel7/num_bands4, **??mfg
  ?µì¼ 1ê°?*, periodë§??°ì´?°ë³„ FFT) / **LTSF = FEATHer-CI**(`ci_full` + ?”ì°¨, **per-dataset**
  6ê°?config). **????MSE ?™ìŠµ**, ì°¨ì´ = ì±„ë„ëª¨ë“œ ?˜ë‚˜. "ì±„ë„?ê? ê°•í•œ ?°ì—…?¼ì„œ=mix, ?½í•œ
  ?€ì±„ë„=?…ë¦½" = inductive bias(DLinear/NLinear ? ë?). ?¨ì¼ configë¡?????ëª»ì¡???„ì—°).
- **??ê·¸ë¦¬?œì„œì¹?ê°?27config, val-rank)**: **d_stateë§?? ì˜ë¯?*(d16>d8>d4), **kernel/num_bands
  flat**(?¸ì´ì¦?. mfg mean-rank 1??ds16_k3_B3?´ë‚˜ **num_bands=4 ì±„íƒ**(4ë°´ë“œ ?œê·¸?ˆì²˜ +
  d16??test-best) ??**ds16_k7_B4**. LTSF: **CIN(0.3588) < L(0.3647)** ??CI ?•ì •.
- **????mfg final 3seed ?„ë£Œ = ë°°í¬ê°€??1??*: `feather_final.csv` 72?? **FEATHer-N OVERALL
  0.6940** > CycleNet 0.720 > FITS 0.726 > DiPE 0.729 > Mix/Sparse 0.749. ?€??=
  `analysis/01_accuracy_mfg.xlsx`(FEATHer 822p). **LTSF final(--group ltsf) ?„ì§** ??03
  FEATHer blank(?Œë¦¬ë©?ì±„ì?). LTSF CIN val=0.3588 ??FITS 0.3582(**4/6 ?°ì´????*, ?™ë¥ ).
- **???…â˜… ?„ë ˆ?´ë° = "128KB MCU ë°°í¬ê°€??ì¤?ìµœì •??(sub-1K ?ê¸°)**: sub-1K ê°•ì œ?˜ë©´ **DiPE
  (339p,0.729)?œí…Œ ì§?*(sub-1K ëª¨ë¸ ì¤?DiPE 1?„Â·CIN 2??=?ë©¸ ?„ë ˆ?? MCU-deployableë¡?ê°€ë©?  N 1??+ ë¬¼ë¦¬??ì°?N ìµœë? 4522p=18KB<<128KB). **ë°°í¬ì§‘í•© 6ê°?*(SparseTSF/MixLinear/DiPE/
  FITS/FEATHer-N/CIN); **CycleNet(H720 288KB)Â·DLinear(545KB) flash ??°œë¡?ë°°ì œ**(FITS 20KB??  CycleNet 288KB=14ë°°ê°­). **?…ë ¥ë²„í¼ ë°©ì–´**: L=720 ?ˆë„??720Ã—DÃ—4B)>20KB SRAM(Steel??19.7KB)
  ??L=96????ë³´ë“œ ìµœë? ë£©ë°±, **ëª¨ë¸ë¬´ê?**(ì±„ë„?…ë¦½ ?ˆì™¸ ?†ìŒ).
- **??configë¥?`_DATASET_OVERRIDES`???™ê²°**(`_FEATHER_MFG_N`/`_FEATHER_LTSF_CI` ê³µìœ dict +
  per-dataset period/arch) ??**`run_forecast.py --model FEATHer --group {mfg,ltsf}`ë¡?ì§ì ‘**
  (???„êµ¬ ??ë§Œë“¦; arch???ë™ `--model_overrides` forward; ê·¸ë¦¬?œëŠ” ?Œì»¤ ì§ì ‘?¸ì¶œ?´ë¼ ë¬´ì˜??.
- **????FASTer ë¯¸ìŠ¤?°ë¦¬ ì¢…ê²°**: ??ì½”ë“œ **test-peeking**(val?†ì´ min-over-epochs test) ??  0.373 ?¬í˜„Â·?¸ìš© ë¶ˆê?(?´ì¤‘?¤ë¼?´ë¡œ??ETT ëª»ë”°?¼ê°=ì½”ë“œ ?„í‹°?©íŠ¸). ?˜ì´ë¸Œë¦¬??????0.347)=
  cherry-picking ê¸ˆì?. `--oracle_test` ?Œë˜ê·¸ë¡œ ?°ë¦¬??ì²œì¥ ë³????ˆê²Œ ??ì§„ë‹¨?„ìš©).
  [[project-old-faster-test-peeking-0809]]
- **??ì»¤ë°‹(personal/ijaehun; ?œë²„ origin=ijaehun)**: b925a12(--oracle_test)Â·529f56e(NaN-tolerant
  resume torn-row)Â·ef1cdc8(mfg N override)Â·7689840(LTSF CI override)Â·76b80d1(--best_only)Â·
  cf704f9(build_paper_tables?’final). ???œë²„ VSCode ?…ëƒ ??askpass ê¹¨ì§ ?????°ë????ëŠ”
  `GIT_ASKPASS= git pull origin main`.
- **???¤ìŒ**: (1) **LTSF final** ë°œì‚¬(`--group ltsf --exp_tag final`, config ì½”ë“œ???ˆìŒ)??  03 ì±„ì? + best_only ì²œì¥ ?•ë¦¬ (2) **ë³´ë“œ ?¬ì¸¡??*(ìµœì¢… ds16_k7_B4 golden ?¬ìƒ?? 07-27?€
  ??453p) = ë°°í¬?¹ì„±??3ì¢?6ëª¨ë¸ latency/energy/memory ?¤ì¸¡ + ë°°í„°ë¦?projection + ?¸ë¼?´ì¦Œ
  flashë²? (3) **Ablation**(?¤ê³„??: A ë¶€???”ì°¨/ê²Œì´??ë°´ë“œ/DTK/SPK/RevIN)Â·B ì±„ë„ëª¨ë“œÂ·C HPë¯¼ê°??  (??ê²Œì´??ë°´ë“œ ?½í• ê²??•ì§ë³´ê³ ) (4) GRU/XGBoost=?‘ìŒ(ë¦¬ë¹„??ì¹´ë“œ). [[project-final-hp-framing-0814]]

**?¡âš¡?¡âš¡?¡âš¡ 2026-08-05 (ë°°í¬ì§‘í•© ?„ë ˆ?´ë° + L/N/CI ?¨ë?ë¦?+ ì²œì¥?¤ìº” ë°œì‚¬ +
MixLinear ë³´ë“œ?¬íŒ… ?„ë£Œ; ?ì„¸ = `memory/project_deployment_family_pivot_0805.md`):**
- **???„ë ˆ?´ë° ?•ì • = "MCU ë°°í¬ê°€??ëª¨ë¸ ì¤?ìµœì •??** (ê²½ëŸ‰tier ?„ë‹˜). DLinear??ë°°í¬?œì„œ
  **sizeë¡??ì—°?ˆë½**(ETTh1 H720 = 139,680 params Ã—4B = **545KB > 128KB flash**, ?„ë? ë¡œê·¸ê°?.
  MCU params ?í•œ ??**2.7ë§?float32)** = (128KB??15KB?Œì›¨?´âˆ’~1.5KB?Œì´ë¸?Ã·4B. robustness ?œê±°
  ?¬í™•??
- **??FEATHer ?¨ë?ë¦?= 2ì¶?4ì¹?* (DLinear Linear/NLinear/DLinear ë¯¸ëŸ¬): ì±„ë„ì²˜ë¦¬(mix/CI) Ã—
  ë¹„ì„ ?•ì”ì°?off/on) ??**L**(base canonical)Â·**N**(mix+?”ì°¨, ?€ì±„ë„??Â·**CI**(ci_full=ì±„ë„?…ë¦½,
  **Dë¬´ê? ~140~370 params sub-1K**)Â·**CIN**. **DÂ²ë²”ì¸ = `in_proj=nn.Linear(eff_d,eff_d)`
  (FEATHer.py line338)**; CIê°€ ?´ê±¸ ?œê±°(Electricity 125K??55). **CI??params???€??MACs??*
  (ì±„ë„ë°˜ë³µ). ?”ì°¨ ?¬ì„¤ê³? ?€??¬(rank16)?’ë‹¨??ë¹„ì„ ??**D??D?’D**(ì²œì¥?¤ìº”?€ ??rank16). **??  ?¨ë?ë¦?2ê°?L/N) vs 3ê°?L/N/CI) = PENDING**, ê²Œì´??= **mfg BlastFurnace(D=28)??CIê°€
  FITS(1.005) ?´ê¸°??* (ì§€ê¸?mixê°€ ì§€??? ì¼ mfg). `--channel_mode {mix,ci_full}`(2330b9c)Â·
  `--use_residual_mlp` ?´ë? ?ˆê³  orchestrator ?˜ë‹¤ forward(ê²€ì¦?.
- **??ì²œì¥?¤ìº” ë°œì‚¬ (?œë²„, ?€ê¸°ì¤‘)**: 4ì¹?Ã— 2loss(mse/l1) Ã— 12?°ì´??mfg6+LTSF?€ì±„ë„6) Ã— 4H Ã—
  **1seed = 384??*, ê²©ë¦¬ CSV `results/fcst_FEATHer_results.csv`(WDSLab?±í¬), exp_tag
  `feather_{L,N,CI,CIN}`+`_l1`. 1seedë¡?ì²œì¥(test-MSE) ë³¸ë’¤ bestë§?3seed. **?¸ì…˜??64/384**
  (mfg L ?„ë£Œ, N ì§„í–‰). **??mfg=MSE ?•ì •**(L: MSE?™ìŠµ 0.713 < MAE 0.744 ?¬í™•?? MAE??LTSFë§?.
  ?ë¦¼ = N2O/MetroPT3 ?€??+ oversubscription(MSE+MAE+lookback 2GPU) ??**FORCE_OMP/?™ì‹œsweep??*.
- **??RÂ² ?œë¡­**: mfg/LTSF ?„ë? -?˜ì¡°(r2_mean???ˆë„?°ë‚´ë¶€ Tì¶•ë¶„?°ìœ¼ë¡??˜ëˆ” ??near-constant
  ì±„ë„ ??°œ). **MSE/MAEë§?*(ë¬¸í—Œ?œì?). êµìˆ˜?í•˜ë©?RÂ²=1?’MSE/Var(?„ì—­) ?„ì²˜ë¦??¬ì‹¤?‰ë¶ˆ??
  ???’MSE=?œìœ„?™ì¼).
- **??ë³µì¡?„í‘œ T5 = `analysis/04_complexity.xlsx` ?„ì„±**(ë² ì´?¤ë¼?? FEATHer placeholder):
  **FFT-aware MACs ì¹´ìš´??* `tools/paper/measure_complexity.py`(matmul=FlopCounterModeÃ·2 +
  FFT=NÂ·log2N; **DiPE 0??1K**; params CSV?¼ì¹˜). `results/complexity.csv`(288). **??FEATHer??  MACs ìµœì†Œ ?„ë‹˜**(FEATHer-L 79K vs FITS 17K); ??ì°¸ê³ ??"min 1.67M"=?¤ë¥¸ê·œì•½. **ì°¨ë³„??=
  ë³´ë“œ latency**(FITS 796ms > FEATHer 349ms = MACs?ì–´???¤ì¹©?ë¦¼).
- **????MixLinear ë³´ë“œ?¬íŒ… ?„ë£Œ + x86 golden PASS** (ë°°í¬ì§‘í•© ? ì¼ êµ¬ë©?´ì—ˆ?? 49p=FEATHer
  ì§ì ‘ sub-1K ?¼ì´ë²?: `deployment/board/export_golden_mixlinear.py`(period24 SparseTSF?€ì¹?
  staged **0 diff**) + `src/mixlinear_infer.{c,h}`(ê³µìš©ì»¤ë„+ë³µì†Œlinear; forward=mean-sub?’shared
  conv1d(k25)?”ì°¨?’segment?’time(factored Lin2Ã—2)?’freq(len4 FFT+ë³µì†ŒLin4????+iFFT-real)?’mix
  Î±0.5) + golden 3?¤ë” + `test/test_mixlinear.c` + `_mixlinear_scalar.py`. **x86 golden PASS**
  (**WSL gcc 11.4 ë¡œì»¬**, 4?¨ê³„ ~1e-7). **ë³´ë“œì½”ë“œ 6ê°??„ë? ?„ë£Œ**; ?¨ì?ê±?MixLinear ë¬¼ë¦¬ì¸¡ì •
  (main.c 4?¸ì§‘=`#define RUN_MIXLINEAR`+ë¶„ê¸°, BOARD_RUNBOOK 51-60 ?¨í„´) + FEATHerìµœì¢…ë³€???¬ì¸¡??
- **??GRU/XGBoost = mfgë§?* (LTSF ?¤í‚µ). **?¹ê?ì¦?*: ?œì¡°PdM ?¼ë¬¸ RF/XGBoost/LSTM/GRU ê´€?‰ì 
  ?¬í•¨(XGBoost ?Œê·œëª¨mfg ê°•ì+?´ì„?? = practitioner-tier credibility + "generic RNN sub-1K
  ?ˆë¨" ?¼ê±°. **ë³´ë“œ ?ˆì˜¬ë¦?*(DLinearì²˜ëŸ¼ sizeë°°ì œ). **ê³µí†µì¶?= flashë°”ì´??*(XGBoost=?¸ë¦¬??  params N/A ??KBë¡?ë¹„êµ). ?œë‹ = GRU hidden{32,64,128}Â·XGBoost depth{4,6,8} mean-rank 1ê°?+
  val? íƒ. ì½”ë“œ ë¯¸ì»¤ë°?list_models15=?¤ìœ•?•ì¥ ?¸ë¦¬ê±? ë³„ë„ì»¤ë°‹).
- **???œë²„ CSV ?•ë¦¬**: `fcst_results.csv` FEATHer 192???œê±°?’ì•„ì¹´ì´ë¸?+ 5seed?¬ë¶„(2028/2029)
  352??ë¶„ë¦¬??_cleanup_20260805/`(ë°±ì—…+README ë¨¸ì?ë°±ë²•). ë©”ì¸ = ë² ì´?¤ë¼??3 ê· ì¼3?œë“œ 3429??
  `baseline_organized.csv`+`organize_baseline_results.py`(?¬ì‹¤?‰ìš©). lookback `--exclude
  FEATHer,ì²œì¥7` seq336/720 ?´ì–´ê°?
- **??ë¯¸ì»¤ë°?ë¡œì»¬**: MixLinear ?¬íŒ… ?„ë?(export/C/golden/test) + ë³µì¡?„ë„êµ?measure_complexity/
  build_complexity_xlsx) + organize_baseline_results + run_grid_ltsf_v2. **???¤ìŒ ?¸ì…˜**:
  ì²œì¥?¤ìº” ?„ì£¼ ?€ê¸?CI/CIN ?˜ì˜¤ë©?**?¨ë?ë¦?L/N vs L/N/CI ê²°ì •**) ???”ì°¨ ìµœì¢…?????¬íŠœ??3seed ??  ??T1 ?¼ì¸??flashì¶•Â·T4 ë°°í¬Â·01/03 ê°±ì‹ Â·Pareto) ??MixLinear ë³´ë“œì¸¡ì • + GRU/XGBoost.
  [[project-deployment-family-pivot-0805]]

**??2026-08-04b (lit_review vault ?„ë©´?•ë¦¬ + baseline provenance/?¬í˜„ë¶ˆê? ?•ëŸ‰ì¦ê±°;
?ì„¸ = `memory/project_litreview_vault_provenance_0804b.md`):** ?¤í—˜ ?€ê¸?ì¤?**ë¬¸ì„œÂ·ë¶„ì„
?¸ì…˜**(ì½”ë“œ/?œë²„ ë³€ê²?0, gitignored vault).
- **?°ì¶œ**: `analysis/lit_review/experiments/`(? ê·œ) foundation 11ê°?protocolÂ·lineupÂ·
  datasetsÂ·**master_table**Â·**number_crosscheck**Â·**datasets_credibility** ?? +
  `papers/`??baseline method ?¸íŠ¸ **9ê°?? ê·œ**(iTransformerÂ·PatchTSTÂ·TQNetÂ·TimesNetÂ·
  TimeMixerÂ·MDMLP_EIAÂ·LMSÂ·CycleNetÂ·DiPE) + ??21??`category::` ?œê·¸ + baseline PDF
  13ê°?TimeXer ?¤ìš´(pdfs/ 22ê°? arXiv IDë¡??°ë„ ?•ì •).
- **??lookback 2?íƒœê³?*: L=96 native(TSLib ê³„ë³´) vs **L=720 native(?£ì?-linear=
  SparseTSF/FITS/MixLinear/DiPE)** ??uniform L=96ê°€ **?°ë¦¬ê°€ ?´ê²¨????sub-1K ?¼ì´ë²Œì„
  ?¸ë””ìº?* ??ë°©ì–´ = ë°°í¬ ë°”ì´?¸ì˜ˆ??+ **SparseTSF@720 ë¶€ë¡?rank-stability**.
- **??ë³µì‚¬-??*(PDF ì§ì ‘?•ì¸): MixLinear?FITS(byte-identical) Â· DLinear??Transformer??  FEDformer(ìº¡ì…˜ ?ë°±) Â· DiPEë¹„ì„ ?•Â·CycleNet?iTf+TimeMixer Â· iTf/PatchTST?Zeng2022 Â·
  TimesNet/LMS/TimeMixer=TSLib ?¬run(ë¿Œë¦¬) Â· TQNet=?¸ìš© Â· MDMLP=ë¶ˆëª….
- **?…â˜… ?¬í˜„ë¶ˆê? ?•ëŸ‰ì¦ê±°(ë°©ë²•ë¡?ìµœê°• ë°©ì–´)**: **DLinear ETTh1@720, ?„ë? nominally L=96Â·
  ?ë‹¹??ê°™ì? TSLib repo?¸ë° 0.513(TimeXer/? ê·œ?´ëŸ¬?¤í„°)~0.695(TimesNet)=35% ?¤í”„?ˆë“œ.**
  ìµœì‹  3??LMS/TQNet/MDMLP)=?œë¡œ byte-identical(?´ë?ê³µìœ ), TimeXer=endpointsë§?ë¶€ë¶„ì¤‘ë³?
  ?ŒìŠ¤ ?„ë‹˜); TimeXerÂ·LMS ?˜ë‹¤ "TSLib ?¬run" ëª…ì‹œ?¸ë° ê°??¤ë¦„ ??**ê°™ì? repo ?¬run??  ?¬í˜„ ????** ???¼ë¬¸ ë°©ë²•ë¡ ì— "ê³µí‘œ?«ì ì§ì ‘ë¹„êµ ë¶ˆê? ???µì¼ L=96 from-scratch ?„ë©´
  ?¬í˜„ë§Œì´ ê³µì •" ë¬¸ì¥?¼ë¡œ ì§í–‰.
- **??mfg-6 ê³µì‹ ???¹ê?ì¦?4ê°?? ê·œ vet)**: ?„ë? credible ??Steel(UCI#851/BRI Q1)Â·
  WindSCADA(Zenodo/Wind Energy Q1)Â·BlastFurnace(**Data in Brief'25** DOI)Â·N2O(descriptor=
  arXiv, **?µì»¤=Water Research'25**)Â·MetroPT3(**Nature Sci Data'22**/UCI#791)Â·EWELD(**Nature
  Sci Data'23**/Figshare). ?„ë¬´??96??20 LTSF ?ˆì????œì¡° ?£ì? LTSF ìµœì´ˆ" ? íš¨. **?°ë¦¬
  ?¬ìš©=ê¶Œê³  ?•í•©**(BlastFurnace 29,602??raw[200??ë¶„ë¥˜?Œì¼ ?„ë‹˜]Â·MetroPT3 ?„ë‚ ë¡œê·¸7ë§?.
  ?¼ë¬¸ to-do: N2O?’Water Research ?¸ìš©Â·EWELD seasonal-naive ?„ìˆ˜Â·MetroPT3 ?„ë‚ ë¡œê·¸7+ê²°ì¸¡
  17% ë§ˆìŠ¤??ëª…ì‹œÂ·Steel ?„ì±„??ê³µë™?ˆì¸¡ ëª…ì‹œ.
- **??ê²°ì •(ì½”ë“œ 0)**: (1) **ë°°í¬ ?¤í—˜ ?¬í”„?ˆì´ë°?* ??FEATHer??ìµœì†ŒÂ·ìµœì†Â·ìµœì??ë„ˆì§€ ??  ?„ë‹˜(SparseTSFê°€ ?´ê?) ??**?¨ìœ¨ê²½ìŸ ?„ë ˆ?´ë° ê¸ˆì?, accuracy-per-resource Paretoë¡?*:
  FEATHerê°€ **FITS/DiPE???ë„ˆì§€Ã—?•í™•??ì§€ë°?*(??ë¹ ë¥´ê³????•í™•), SparseTSF/MixLinear??  ?¨ìœ¨ë§?ì§?ê±?ëª…í™•?????•í™•). **?¤ë“œ?¼ì¸=?•í™•??sub-1K ìµœì •??MCU ?‘ì¬), ë°°í¬=feasibility
  +FITS/DiPEì§€ë°?ë³´ì¡°.** (2) **???‘ì? ë³´ë“œ êµ¬ë§¤ NO**(?…ë ¥ë²„í¼ ëª¨ë¸ë¬´ê?Â·FEATHer ?¬ì „??  ?¬ê³  ?ë¦¼Â·SparseTSF ?ë„ˆì§€ ëª»ì´ê¹€=ë³´ë“œë¬´ê? ??? ë§¤????ê³ ì¹¨; ë°°í„°ë¦?ê°ë„ë§??í•˜ë©?ê·¸ë•Œ
  L053R8 ULP 1ê°?. (3) **?¼ì¸??ì¶”ê? NO**(S4D/SegRNN/MambaSimple/Koopa: Mamba=?ì¹™?œì™¸,
  ?˜ë¨¸ì§?Related Work ?¸ìš©+escalation; 14 ? ì?).
- **???¤ìŒ**: ?œë²„ ?ê?(CycleNet/MixLinear LTSF + baseline ë£©ë°± ?¤ìœ• `--exp_tag lookback`
  seq48/192/336/720 edge-6) + **FEATHer v2(MLP 640p) ë³´ë“œ ?¬ì¸¡??*(?•í™•??v2 / ë°°í¬?¤ì¸¡=
  canonical 453p = ë¶ˆì¼ì¹?êµ¬ë©). [[project-litreview-vault-provenance-0804b]]

**?¡âš¡?¡âš¡?¡âš¡ 2026-08-04 (JMS ë°°í¬-ì¤‘ì‹¬ ?¼ë²— + robustness ?¬ë§?•ì • + FEATHer-MLP v2;
?ì„¸ = `memory/project_jms_deployment_pivot_0804.md`):**
- **??robustness thesis ?°ì´?°ë¡œ ?¬ë§**: `robust_feather_v2_mlp.csv` 6 mfg ?„ì£¼ ??  FEATHer-MLP(canonical+MLP+RevIN)???£ì?-7 ì¤?**ê¼´ì°Œ** ??Dw(?ë?) 1.211 vs peer
  1.006~1.034 = **7/7**, corruptMSE(?ˆë?) 0.824 vs 0.688~0.749 = **7/7**(clean ì¢‹ì•„??  ëª?êµ¬ì œ). no_revin(EWELD ??°œ ë¹„ë³´??Â·robust-RevIN(median/MAD ?¤íŒ¨)Â·center(N2O ?…í™”)
  = **?‘ë™?˜ëŠ” ë³´í¸ fix ?†ìŒ.** ??**JMS??robustness ??ì£¼ì¥ ?µì§¸ ?œê±°**(?£ìœ¼ë©?"?œì¼
  ??ê°•ê±´"?´ë¼ ??š¨ê³? ì£¼ì¥ ???˜ë©´ ???ŒìŠ¤?¸í•´????. robustness = ?¥í›„ ???•ê·œ???™ê¸°ë¡œë§Œ.
- **??JMS = ë°°í¬-ì¤‘ì‹¬ ?¼ë²—(?•ì •)**: ê¸°ë‘¥ = accuracy-per-param + Cortex-M3 ?¤ì¸¡. ?Œë¼ë¯¸í„°
  ?•ë‹¹??"453p>41p??robustness"(ì£½ìŒ) ??**"?•í™•??**(FEATHerê°€ SparseTSF/MixLinearë³´ë‹¤
  ?•í™•)ë¡??„í™˜.
- **???•í™•??= FEATHer-MLP(v2) #2**: `feather_v2_mlp.csv` ?„ì£¼(14?°ì´??6mfg+8LTSF,
  3seed, **MLP rank16 + loss mse**). mfg ?£ì?-7 OVERALL MSE: DLinear 0.668 > **FEATHer
  0.694(640p)** > CycleNet 0.720 > FITS 0.726 > DiPE 0.729 > MixLinear/SparseTSF 0.749.
  = **#2, 29ë°???DLinear ë¹¼ê³¤ ???´ê?, Steel 1??DLinear ?œì¹¨).** DLinear ê°?˜ **79%ê°€
  BlastFurnace ?˜ë‚˜**(D=28 ê³ ì±„?? MLP ì±„ë„?¼í•© ë¬´ìµ; CIê°€ ?ˆë²„ ?„ë³´??ë¯¸ê?ì¦?.
- **??2-??êµ¬ì¡°(DLinear ì²˜ë¦¬)**: ?•í™•?„í‘œ=**?£ì?-7 ?„ë? ? ì?**(DLinear ?¬í•¨ ??ë¹¼ë©´ "?´ê¸°??  ???¨ê?"=ë¦¬ë·°??ì¦‰ì‚´) / ë°°í¬??**MCU ?¤ì–´ê°€??ê²ƒë§Œ**(F103RB 128KB flash): ?¤ì¸¡ 5ê°?  (persistence/SparseTSF/DiPE/FITS/FEATHer)+MixLinear(?¬íŒ… ??. **DLinear ?ˆë½**(H720??  138K??52KB>128KB = ë¬¼ë¦¬??ë¯¸íƒ‘??= dodging ?„ë‹ˆ??ë°œê²¬).
- **???´ë˜??ë² ì´?¤ë¼??*(GRU/XGBoost) ë°°í¬??ì¶”ê? = ?œì¡° ë¦¬ë·°??credibility. ê·œì¹™ =
  ?¬ë¦¬?Œë§ ê¸ˆì?(**budget ??best config**) + **?ì—°?¬ê¸° ë³‘ê¸°**(ì¡°ì‘ ?„ë‹˜ ì¦ëª… + "generic
  RNN?€ sub-1Kë¡???ì¤„ì–´??=?°ë¦¬ ?¼ê±°) + **XGBoost??flash ë°”ì´?¸ì¶•**(?¸ë¦¬??params ë¬´ì˜ë¯?.
  ì½”ë“œ ?´ë? ?ˆìŒ(ë¯¸ì»¤ë°? [[project-jms-baseline-repositioning]]).
- **??LTSF = FEATHer #5(?½í•¨)**: 0.406 vs FITS 0.376/SparseTSF 0.380/DiPE 0.380/DLinear
  0.399. MLP ?£ì–´???½í•¨ = ?œì¡° ?€ê²??¤ê³„, LTSF ???„ë‹˜. ??LTSF??accuracy-per-param??  ë¶ˆë¦¬(SparseTSF 41p<640p?¸ë° ???•í™•). **ê³µì •??OK**(FEATHer LTSF per-dataset ?œë‹??
  ë² ì´?¤ë¼?¸ë³´????. ê·¼ë° **?¬íŠœ???™ê¸° ?•ë‹¹ = ?´ì¤‘ ë¯¸ìŠ¤ë§¤ì¹˜**(archê°€ L1+MLP?†ì´ ?œë‹?ëŠ”??  ì§€ê¸?MSE+MLP). **ëª©í‘œ="?™ê¸‰ ?¼ë°˜?”ì¦ëª? not ?´ê?**(07-22: arch?œì¹˜ ìµœì =degenerateÂ·FITS
  ëª??´ê?); ë¦¬ë·°??"mfg ?¤ë²„??" ë°©ì–´??
- **??ì½”ë“œ(ë¯¸ì»¤ë°?ë¡œì»¬, push ?„ìš”)**: `run_grid_ltsf_v2.py`(? ê·œ ??MLP+MSE-aware,
  **kernel?? ê°•ì œ**(kernel=1=DTK ì£½ì—¬ ? í˜•??degenerate ë°°ì œ, FEATHer ?•ì²´??? ì?),
  144config, ê²©ë¦¬ CSV `results/grid_ltsf_v2.csv`, ê¸°ë³¸ 6 ?€ì±„ë„ LTSF) + `build_paper_tables.py`
  (FEATHer ?ŒìŠ¤ canonical?’feather_v2_mlp, params 453??40, **03_accuracy_ltsf.xlsx** ?œíŠ¸
  ì¶”ê?). **????baselines/__init__.py ??ê±´ë“œë¦?= ?¤ìœ• ?ë™?•ì¥ ?¸ë¦¬ê±??†ìŒ = ?ˆì „ ì»¤ë°‹**
  (?œë²„ grid pull ?„í•´ push ?„ìš”).
- **???œë²„ ?íƒœ**: **CycleNetÂ·MixLinear LTSF ë¯¸ì‹¤??*(baseline ?˜ì¤‘ ì¶”ê???mfgë§???. ??  orchestrator `--model`?€ **?¨ì¼ë§?*(ì½¤ë§ˆ X, `--exclude`ë§?ì½¤ë§ˆ) ???°ë¡œ ë°œì‚¬:
  `python run_forecast.py --group ltsf --model CycleNet --exp_tag main --num_seeds 3
  --ngpu 2 --jobs_per_gpu 3`(ê·¸ë‹¤??MixLinear). ê¸°ë³¸ `results/fcst_results.csv` resume.
  ??3ê°?ê°±ì‹ (`analysis/0{1,2,3}_*.xlsx`).
- **???¤ìŒ ?¸ì…˜**: (1) CycleNet/MixLinear LTSF ë°œì‚¬?’í‘œ ?¬ìƒ??(2) grid ?„êµ¬ push?’ì„œë²?pull??  ETTh1 ?„ë¡œë¸?576??1seed)??--summary` ?ì • (3) ê²Œì´???µê³¼ ??6ê°??•ì¥?’best per-dataset??  `_DATASET_OVERRIDES` ê°±ì‹ ??seed ìµœì¢… (4) ë°°í¬??GRU/XGBoost budget-best+?ì—°?¬ê¸°+ë°”ì´?¸ì¶•)
  ?‹ì—…. [[project-jms-deployment-pivot-0804]]

**?¡âš¡?¡âš¡??2026-08-03 (ì²´ê³„??ë¬¸í—Œë¦¬ë·° ???¸ë²¨??ì§„ë‹¨ + ?¥í›„ ë°©í–¥; ?ì„¸ =
`memory/project_lit_review_novelty_0803.md`, vault = `analysis/lit_review/`):**
- **?™ê¸°**: ?¬ìš©??"FEATHer=ì§¬ë½•(ê¸°ì¡´ ?¸ë¦­ ì¡°í•©), ë°©ë²• ?¸ë²¨???†ë‹¤" ?°ë ¤ ??**Obsidian
  vault ì²´ê³„??ë¦¬ë·°**(md+wikilink+graph, gitignored). PDF 1ì°¨ì¶œì²??•ë…?’êµ¬ì¡°í™” ?¸íŠ¸ 1?¸ì”©,
  **10???„ë£Œ**: SparseTSFÂ·FITSÂ·MixLinearÂ·DLinear/NLinearÂ·RevINÂ·OnTheRoleOfRevINÂ·FANÂ·
  FredNormerÂ·SANÂ·TS-FaultÂ·SensorFault-Bench.
- **??ê²°ë¡ (?‰ì •)**: **??FEATHer ë°©ë²• ?¸ë²¨??= 0.** ë¶€???„ë? ? ì (center=NLinearÂ·CI=
  DLinearÂ·?„í„°robust=FITS/MixLinearÂ·FFT?•ê·œ??FAN). **FFT ê²Œì´??ë¶€ì±?*(07-31b ê¸°ì—¬~0 +
  FAN???ë¦¬?ìœ¼ë¡??????¬ë§?ŒìŠ¤???€??. ì¡°ë¦½?€ ?¸ë²¨???„ë‹˜ = ?¬ìš©??ì§ê° ?•í™•.
- **ê°•ê±´ ë²¤ì¹˜ë§ˆí¬ 2?¸ì´ ? ì (ë°©ë²• ?„ë‹ˆ??ì§„ë‹¨?„êµ¬=?°ë¦¬ ë°©ë²•?¸ë²¨???„í˜‘ X, ?¸ìš©?°êµ°)**:
  TS-Fault(clean?”robust ë°˜ë¹„ë¡€ ?=??.544) Â· **SensorFault-Bench(RevIN??clean ?•ê³ 
  degradation ?¤ì? ???°ë¦¬ 07-31b ì§„ë‹¨??benchmarkë¡œÂ·ìš°ë¦??„ë©”??wind SCADA??? ì ).**
- **?•ê·œ???˜ë¦¬ ê³µê°„ ?¬í™”**: FAN(NeurIPS'24)Â·FredNormer(ICLR'25?¬ì‚¬)Â·SAN(NeurIPS'23) ??  **clean ë¹„ì •?ì„±** ?€?? **OnTheRoleOfRevIN**: "scale ?œê±°=predictive context ë²„ë¦¼" ??  **center??N2O/EWELD clean ?…í™”??ë²„ê·¸ ?„ë‹ˆ???Œë ¤ì§??¸ë ˆ?´ë“œ?¤í”„**(center ë§ŒëŠ¥?˜ë¦¬??
  robust-RevIN???¼ë°˜??. **RobustTSF(ICLR'24)**: MAE robustÂ·MSE non-robust ì¦ëª…(07-30
  L1?’MSE ?¬ê³  ?€?? + regime=**training-?¤ì—¼ vs ?°ë¦¬ inference-?ìƒ**=?œì¼ ê¹¨ë—??ì°¨ë³„??
- **??? ì¼?˜ê²Œ ë¹?ë°©ë²• ?ë¦¬**: **?ìƒ ê°€?¥ì„± ?ˆëŠ” ?…ë ¥ ?ˆë„?°ì„œ reversible mean/stdë¥?  within-window robust ì¶”ì •, scale(context) ? ì?, sub-1K.** ê²½ê³„: ì£¼íŒŒ?˜âŒ(FAN/FredNormer/
  SAN)Â·ë¯¸ë˜?ˆì¸¡??SAN)Â·across-dataset??FredNormer). ?¤íŒ¨ê·¼ê±°=median(50%miss breakdown)Â·
  center(scaleë²„ë¦¼)=???ë¦¬ê°€ ?´ë µ???„ì§ ë¹?
- **??ê²°ì • (?¬ìš©??**: **FEATHer = ?´ë?ë¡??¤í—˜ ????JMS ?œì¶œ**(?¬ì„¤ê³????? ?•ì •).
  **?¥í›„(?¤ìŒ ?¼ë¬¸) 2ê°ˆë˜**: ? ëª¨?¸ë§ ?¬ì„¤ê³?"ì§„ë‹¨?’ì•„?¤í…ì²?, ë¶€???¬ë§?ŒìŠ¤?? + **?¡ì†??
  robust ?•ê·œ??*(model-agnostic=?…ë¦½ ë°©ë²•?¼ë¬¸ ?’ì?venue + ? ì˜ ?¸ë²¨??ì²™ì¶”). ê·¸ë¦¼: FEATHer
  (JMS)?’â‘¡(?’ì?venue)?’â‘ (?¡í’ˆ?€ ?¬ì„¤ê³?. ?°ì¶œ=`analysis/lit_review/{FUTURE_WORK.md,
  gaps/SYNTHESIS_solution_approach.md, _MOC.md, papers/*.md}`.
- **??ëª¨ë“œ ?„í™˜**: ë¦¬ë·° ë§ˆë¬´ë¦? ?¤í—˜(êµ??œë²„ FEATHer+MLP full rerun ?? ?€ê¸°í•˜ë©?pending
  ì²˜ë¦¬ ëª¨ë“œë¡? [[project-lit-review-novelty-0803]]

**?¡âš¡?¡âš¡ 2026-07-31b (?•í™•??ì°©ì? + MLP ?¤í—˜ + RevIN robustness ì§„ë‹¨; ?ì„¸ =
`memory/project_mlp_revin_diagnosis_0731.md`):**
- **???•í™•??ì°©ì? (?£ì?-7, mfg ?„ì£¼ 4 = Steel/WindSCADA/BlastFurnace/EWELD, 3seedÃ—4H,
  ?„ë? mse?™ìŠµÂ·mseì±„ì ):** FEATHer = **sub-1K ?•í™•??ë¦¬ë”.** MSE ?‰ê·  **3??7** (DLinear
  0.645[18.6K]Â·CycleNet 0.708[9.5K]ë§??? FEATHer 0.7095[453]) ??**ì§„ì§œ sub-1K(MixLinear
  49Â·SparseTSF 41Â·DiPE 267Â·FITS 1.2K) ?„ë? ?´ê?, MixLinear??16?€ ?„ë? ??** MAE 4??  (MixLinear/SparseTSF???´ê?). **07-30 L1/MSE confound ?´ì†Œ ?•ì •** (val_loss ?¤ì??¼ë¡œ??  mse ?™ìŠµ ê²€ì¦?. ??MetroPT3/N2O ê¼¬ë¦¬(FEATHer/MixLinear/CycleNet) ë¯¸ì™„. DLinear ê°?**?„ë?
  BlastFurnace(D=28)+WindSCADA(D=15) ê³ ì±„??* ??Steel/EWELD??FEATHer 2??
- **???”ì°¨ MLP ?¤í—˜ (? ê·œ, `--use_residual_mlp/--mlp_rank` ê¸°ë³¸ OFF, canonical
  byte-identical):** ê²Œì´???œí˜„ h??**ì±„ë„ì¶??€??¬ ë¹„ì„ ???”ì°¨** `h+=W_up(GELU(W_down(h)))`
  (D?’rank?’D, +2Â·DÂ·r; Steel 453??40@r16, **sub-1K ? ì?**). **clean MSE: Steel ??%Â·
  WindSCADA ??% (3seed), BlastFurnace ~0**(ê³ ì±„?ì? ì±„ë„?¼í•©???´ì§ˆì±„ë„ ?ì–´ ë¬´ìµ). ì¶”ì •
  FEATHer+MLP ~0.68 ??**CycleNet ?œì¹˜ê³?#2**. **robustness ì¤‘ë¦½**(Dw ë³¸ì „, ?ˆë??¤ì°¨??ê°œì„ ).
- **????RevIN robustness ì§„ë‹¨ (ê²°ì •?? Steel 3-arm, ??MLP-on):** canonical FEATHerê°€
  ?£ì? ì¤?**robustness ìµœì•…**(Steel Dw~1.24 vs ?¼ì´ë²?1.03~1.06; ìµœì? severity??ì¦‰ì‹œ ê³„ë‹¨,
  impulse@0.05 Dw 2.17). **ë²”ì¸ ?•ì • = RevIN, FFT ê²Œì´???„ë‹˜.** gate-off(uniform) Dw
  1.202??.200(ë¬´ë??? / **no_revin Dw 1.202??.076(?¼ì´ë²?ê¸‰ë½)**. **ê²Œë‹¤ê°€ no_revin??clean
  MSE??ìµœê³ (Steel 0.477, ??3%) = robustness+clean ?´ì¤‘ ?´ë“.** ë©”ì»¤?ˆì¦˜=?ìƒ ?ˆë„?°ë¡œ
  mean/std ê³„ì‚°?’impulse ?¤íŒŒ?´í¬ê°€ ?µê³„ ?¤ì—¼?’ì •ê·œí™”Â·?? •ê·œí™” ì¦í­. (?¬ìš©??RevIN ê°€???ì¤‘,
  Claude ê²Œì´??ê°€???¤ë‹µ.) **FFT ê²Œì´?¸ëŠ” clean/robust ê¸°ì—¬ ~0 = ê°’ì–´ì¹?ë¯¸ê?ì¦?*(gate ablation
  ?¬ìƒ ìµœì´ˆë¡???.
- **?„ëµ ?„í™˜**: (a) **accuracy-first** ?„ë ˆ?´ë°(robustness-not-accuracy???°ì´?°ë¡œ ?? „?ìœ¼??  RevIN-fixë¡??•ë‹¹ ë¶€??ê°€??. (b) **?¸ë²¨??ì²™ì¶” = "?¼ì„œê³ ì¥ ??instance-norm(RevIN) ì·¨ì•½??  ì§„ë‹¨ + robust-RevIN(median/MAD) fix"** ??JMS(?œì¡°?€?? ML?¸ë²¨??ë°???Œ)???? ?œì¡° ë¬¸ì œ?€
  ?¼ì¹˜. (c) **DLinear ì¶”ì›”?€ BlastFurnace ê³ ì±„???˜ë‚˜???¬ë¦¼** ??ê·¼ë° mfg??**per-dataset
  ?œë‹ ê¸ˆì?(??ëª¨ë¸ default config ?€ì¹?** ???¨ì¼ canonical ?„í‚¤?ì²˜ ë³€ê²½ìœ¼ë¡?ê³ ì±„???´ë¦¬??  ?€ì±„ë„ ??ë§ì³???´ë ¤?€). **?œì?ë¶€??FFN)?¼ë¡œ DLinear ì«“ê¸°=?¸ë²¨??0, ì§€??** (d) FITSê°€
  robust???´ìœ =FFTë¥?**?„í„°(low-pass cut_freq=seq_len//4)ë¡??€**(ë°˜ì‘ ?„ë‹˜)?’ê³ ì£¼íŒŒ ?ìƒ ?ê¸°.
- **???¤ìŒ (RevIN???µì‹¬)**: WindSCADA/EWELDë¡?no_revin ?¬í™•??Steel?€ ?•ìƒ??ì»¤ì„œ ?œê±° ?´ë“??  ???ˆìŒ, ë¹„ì •?ì„± ?°ì´?°ì„  clean ?´ì¹  ?˜ë„??*robust-RevIN???¼ë°˜??*) ??**robust-RevIN(median/
  MAD) ë³€ì¢?êµ¬í˜„=?¼ë¬¸ ê¸°ì—¬ ?µì‹¬.** + MLP A/B EWELD/BlastFurnace ?„ì£¼ + ?•í™•??MetroPT3/N2O ê¼¬ë¦¬.
- **??ì½”ë“œ (personal push): `ffbffde` MLP flag** + robustness MLP ë°°ì„  + **NaN-tolerant
  resume**(?™ì‹œ append torn-row ?½ìŠ¤) + **`--no_revin`/`--gate_uniform` ì§„ë‹¨ ? ê?**(ê¸°ë³¸ OFF).
  ê²©ë¦¬ CSV: `feather_mlp_ab.csv`(?•í™•??Â·`robust_mlp_ab.csv`(robustness). robustness??  **jobs_per_gpu 2 ê¶Œì¥**(torn-row). robustness arm?€ ?™ìŠµê³?**flag ?™ì¼**?´ì•¼(forward ?¼ì¹˜).
  [[project-mlp-revin-diagnosis-0731]]

**?¡âš¡??2026-07-31 ?€ê°œí¸ (?£ì?-tier ?¬êµ¬ì¡°í™”; ?ì„¸ =
`memory/project_edge_tier_restructure_0731.md`):**
- **?„ë ˆ?´ë°**: JMS ? ì?(IoT-J ?´ë™ X) + **IIoT ?„ë ˆ?´ë°ë§?*(?°ì´??êµì²´ X, "?œì¡°"??  "?°ì—… IoT ?£ì?" ?œìˆ ë§?. **ë©”ì¸ ë¹„êµ = ?£ì?-tierë§? ì²œì¥(?€??=Pareto plot ?ìœ¼ë¡œë§Œ**
  (?? œ X, ë¹„êµ??X ??"??SOTA ?†ëƒ" ë°©ì–´ + accuracy-per-param ? ì?). ?£ì? ?•ì˜ =
  "published default?ì„œ ?ì—°???‘ì?" ëª¨ë¸(?µì? ì¶•ì†Œ=strawman ê¸ˆì?).
- **?¼ì¸??14**: ?£ì?-7 (SparseTSF 41Â·**MixLinear 49**Â·DiPE 267Â·**FEATHer 453**Â·FITS
  1.2KÂ·CycleNet-linear 9.5KÂ·DLinear 18.6K) + ì²œì¥-7 (PatchTST~MDMLP 35K~1.6M, **ê¸°ì¡´
  ?°ì´???¬ì‚¬?©Â·ì¬?¤í–‰ X**). **?´ë˜??LSTM/GRU/TCN/XGBoost)=`_archive/` ?„ì¹´?´ë¸Œ**(ë¦¬ë¹„??.
- **??MixLinear(ICLR'26, 49p) = FEATHer ìµœë? ?„í˜‘** ??dual time+freq, sub-1K SOTA ì£¼ì¥,
  FEATHer niche ì§ì ‘ ê²½ìŸ?? ?•í™•?„ë¡œ ?´ê¸°ë©?sub-1K ?•í™•??ë¦¬ë” ì£¼ì¥ ?‘ê³  robustness/ë°°í¬ë¡?
  **ë²¤ì¹˜ë§ˆí¬ ê¸‰í•¨ = Exp1 ì²?ê´€ë¬?** MixLinear/CycleNet ë°°ì„ (ì£¼ê¸°=per-dataset FFT ë¯¸ëŸ¬
  SparseTSF/TQNet, lr=?¼ë¬¸ ?¤í¬ë¦½íŠ¸ default 0.03/0.005; wrapper=debug print ?µì œ+lpf clamp).
- **FEATHer = canonical(d8/k7/B3+FFT period) + loss l1?’mse** (07-30 confound: mse?™ìŠµ??  MSE???´ê?). **??14ëª¨ë¸ train=mse/eval=mse = ê³µì •.** arch/lr ?œì¹˜(grid/OFAT/mfg-search/
  lr-search) ?„ë? `_archive/`. **ê³µì •??ê°ì‚¬ ?„ë£Œ**(??ëª¨ë¸ ?¼ë¬¸ default).
- **robustness = ?£ì?-7 Ã— mfg-6** (LTSF ?µì»¤ ETTh1/Weather ?œë¡­ ???•í™•?„í‘œ ì»¤ë²„ë¦¬ì? ?¼ì¹˜),
  4 faultì¶?report??quant ?œë¡­). **ì²´í¬?¬ì¸??= ëª¨ë¸ë³??´ë”**(`checkpoints/<tag>/<model>/`,
  `migrate_checkpoints.py`ë¡?2151ê°??´ë™ ?„ë£Œ). **dispatch = ê²½ëŸ‰ ë¨¼ì?**(`_MODEL_DISPATCH_ORDER`).
- **ìµœì¢… ?¤í—˜ 9**: ë©”ì¸(1 ?•í™•?„L96Â·2 robustnessÂ·3 lookback ?£ì?7Ã—mfg6Ã—L{48,192,336}Â·
  4 ablation-clean 5ì¶? / ê°€ì¹?5 ?°ì´?°íš¨?¨cold-startÂ·6 ablation-under-corruptionÂ·
  7 ?ë„ˆì§€envelope ê³„ì‚°) / ??8 ParetoÂ·9 ??. **FFT ê²Œì´???´ì„??ë²„ë¦¼.**
- **???œë²„ ?íƒœ**: **Exp1(mfg ?•í™•?? ?‘GPU ngpu2)** + **Robustness(?£ì?7Ã—mfg6, GPU0,
  FORCE_OMP=4)** ë³‘í–‰ RUNNING. Exp1 ê²½ëŸ‰ë¨¼ì???MixLinear?’FEATHer?’CycleNet ??ì²´í¬?¬ì¸??  ?ì„±. robustness=ì²´í¬?¬ì¸???ˆëŠ” ê²ƒë§Œ(ë©±ë“±, ?¬ì‹¤?‰ìœ¼ë¡?ì±„ì?). ??**ë°•ìŠ¤ ê³µìœ  ??FORCE_OMP
  ?„ìˆ˜**(ê°?orchestrator ?ê¸°?Œì»¤?˜ë¡œ ?ë™?€?’ë™??oversubscription?’CPU?¤ë ˆ?±â†’GPU 80%??0%).
- **?¤í—˜ ì½”ë“œ = ?„ë? ì¤€ë¹??„ë£Œ** (board ?¬íŒ…ë§??¨ìŒ): `--train_frac`(?°ì´?°íš¨??`ff139fd`)
  + `pareto_plot.py`(`19c83d4`) + `energy_envelope.py`(`93f0009`) + `run_robustness
  --ablation_axis`(ablation-under-corruption `9df1b40`). lookback=`--seq_len` ë°˜ë³µ,
  ablation-clean=ê¸°ì¡´ ?¸í”„?? ??**?ë„ˆì§€ ë°œê²¬**: "?°ì‚°=?€ê¸?1.5%=ê³µì§œ"??NUCLEO sleep
  (ë¹„ë??? ê¸°ì?; production STOP???°ì‚° 90% ì§€ë°???**ì§„ì§œ ?¼ê±°="FEATHer ì§§ì? latency??  ë°°í„°ë¦??¤ë˜ê°?latency? params)"**. **?¨ì? ì½”ë“œ = board C ?¬íŒ…**(MixLinear ?°ì„ , config
  ?•ì • ??WSL/ë³´ë“œ ë°°ì¹˜; ë¡œì»¬ gcc ?†ìŒ). **ë³‘í–‰ to-do = ?ê³  ë¦¬í”„?ˆì„**(*_jms.md ??  ?„ë ˆ?´ë°?’IIoT/?£ì?tier/14ëª¨ë¸/MixLinear/mse). push `e22236f`~`9df1b40`(personal).
  ì²´í¬ ëª…ë ¹: `results_table.py --data Steel,WindSCADA,EWELD` / `robust_summary.py --csv
  results/robust_results.csv`. [[project-edge-tier-restructure-0731]]

**?¡âš¡ 2026-07-30 ë°œê²¬ (L1/MSE ?ì‹¤ confound ???ì„¸ =
`memory/project_l1_mse_loss_confound_0730.md`):** mfg ?œì¹˜ ì¤‘ê°„?ê? ì¤??¬ìš©?ê?
"FEATHerê°€ clean ?•í™•?„ë¡œ ?•ë„?ìœ¼ë¡??˜ì˜?? ???ì¸ ì§„ë‹¨ ê²°ê³¼ **?ì‹¤/ì§€??ë¶ˆì¼ì¹?*.
- **FEATHerë§?`loss=l1`, ë² ì´?¤ë¼??11ê°œëŠ” ?„ë? `loss=mse`**(ì½”ë“œ `get_method_defaults`
  ?•ì¸)?¸ë° ?´ë¦° ?•í™•???œëŠ” **MSEë¡?ì±„ì **. L1?€ ?´ìƒì¹˜ì— robust ?????”ì°¨ë¥?ë°©ì¹˜ ??  ?¤íŒŒ?´í¬ ë§ì? ê³µì •? í˜¸(Steel ???ì„œ ê·??¤ì°¨ê°€ **MSE?ì„œ ?œê³±????°œ**.
- **ì¦ê±°(?„ì£¼ë¶?Steel/WindSCADA/EWELD, ê°™ì? ?ˆì¸¡Â·ì§€?œë§Œ êµì²´)**: FEATHer\*(?€ë³?  min-config ?¤ë¼?? **MSE ì±„ì  10.3/12(ìµœí•˜??** ??**MAE ì±„ì  4.3/12(sub-1K 1??* ??  FITSÂ·DiPEÂ·SparseTSFÂ·DLinearÂ·PatchTSTÂ·TQNetÂ·MDMLP ?„ë? ?´ê?). ??ëª¨ë¸(TimesNet/LMS)?€
  ?´ëŠ ì§€?œë“  ?¬ì „???? ??"sub-1Kê°€ 500K-param???´ë¦°?¼ë¡œ ëª??´ê¸´????? íš¨, ë°”ë€???  ?ˆëŠ” ê±?"**sub-1K tier ?ˆì—??FEATHer ìµœê°•**"?´ë¼????ê°•í•œ ?•í™•??ì£¼ì¥.
- **?Œê¸‰**: ETTh1 ê·¸ë¦¬??"clean 8/9?? ê²°ë¡ ??MSE ê¸°ì??´ë¼ **ê°™ì? confound** ??07-28
  "FEATHer clean ëª??´ê?" ?¬ê????€??MSE ì§€??ì°©ì‹œ ?±ë¶„).
- **?¬ìš©??ê²°ì • = L1Â·MSE ?????¬í•™??ë¹„êµ.** `search_feather_mfg.py`??**`--loss
  {l1,mse}` ì¶”ê?**(ì»¤ë°‹ `eb91d39`, personal push): `mse`??results/feather_mfg_search_mse.csv`
  ?ë™ ?¼ìš°??+ exp_tag `search_mse_*` ?‘ë‘?????Œë˜ L1 ?œì¹˜(`feather_mfg_search.csv`,
  `search_ds*`)?€ **CSVÂ·resume???„ì „ ê²©ë¦¬, ?™ì‹œ ?¤í–‰ ?ˆì „**(l1 ê²½ë¡œ/?œê·¸ ë¬´ë?ê²?.
- **??07-30 ë°œì‚¬**: êµ??œë²„ GPU0 `FORCE_OMP=5 CUDA_VISIBLE_DEVICES=0 python
  search_feather_mfg.py --loss mse --ngpu 1 --jobs_per_gpu 2`(324??. ?ì • =
  `--loss mse --summary`. ???œì¹˜ ??MSEÂ·MAE ê¸°ë¡ ??**4ë¶„ë©´(L1?™ìŠµ/MSE?™ìŠµ Ã—
  MSE??MAE??**?¼ë¡œ ?¤ë“œ?¼ì¸ ì§€?œÂ·ì†??ìµœì¢… ê²°ì •, canonical?€ 3seed ìµœì¢…??
  ??**MSE-?™ìŠµ ?œì¹˜ ê²°ê³¼ ?„ì§ ?†ìŒ**(ë°©ê¸ˆ ë°œì‚¬) ????10?? ?? „?€ L1-?™ìŠµë¶?MAE
  ?¬ì±„?ìœ¼ë¡œë§Œ ?•ì¸. [[project-l1-mse-loss-confound-0730]]
- **???œë²„ ?íƒœ (07-30):** êµ??œë²„ GPU1 = mfg ë² ì´?¤ë¼???¤ìœ• RUNNING(505/792 ??4%;
  Steel/BF/WindSCADA ?„ì£¼, EWELD ì§„í–‰ ì¤?heavy 3ëª¨ë¸ ê¼¬ë¦¬[iTransformer 0Â·TimesNet 4Â·
  TimeMixer 9], N2O/MetroPT3 ë¯¸ì°©??. êµ??œë²„ GPU0 = FEATHer mfg ?œì¹˜ L1(135/324) +
  **MSE ?œì¹˜ ? ê·œ(0/324)**. LTSF `fcst_results.csv` ê±°ì˜ ?„ì£¼(?œë“œ ?¼ì¬). ê°?ê±´ê°•
  (NaNÂ·zeroÂ·dup 0).

**??2026-07-28 ê²°ì • (?œì¡°-ê°€ì¹??¤í—˜ ?¤ì½”???•ì •; ?ì„¸ =
`memory/project_mfg_value_experiments_0728.md`):** ETTh1 ê·¸ë¦¬???„ì£¼(360/360) ???¤ë¼?????•ì¸ ??**FEATHer clean ?•í™•?„ë¡œ ëª??´ê? ?¬í™•??*(test-peek ?¤ë¼??ì¤˜ë„
ETTh1 16ì¹?ì¤?1ì¹¸ë§Œ 1?? ê³µì • val-? íƒ 9/13??. ?€ë§?= robustness + ë°°í¬. ê·?ë°°í¬ ê¸°ë‘¥??"3D"ë¡??ˆë‹¤?’ì˜¤???ˆë‹¤?’ì¬?œë‹ ?†ì´ ?ˆë‹¤) ê°•í™”?˜ë ¤ ?¡â‘¢ ê²€????ê²°ë¡ :
- **??one-config-fits-all-sensors = ?ê¸°.** ?œì¡°??`_DATASET_OVERRIDES`??**?„ì›
  ?¼ë¬¸ default config + period ëª¨ë¸ë§?FFT ?ë™ì£¼ê¸°** = ëª¨ë‘ "??config" ??FEATHerë§?  ê°•ì  ?„ë‹˜, ë¹„êµ ?°ìœ„ ?†ìŒ. (??full=canonical vs per-dataset-ìµœì  FEATHer ê°??
  ?œì¡° ê·¸ë¦¬??8.6k?°ì´??ë¦¬ë·°???”êµ¬ ?œì—ë§? ì§€ê¸ˆì? ë¬¸ì¥ ?˜ë‚˜ë¡?ê°•ë“±.)
- **???ë„ˆì§€/ë°°í„°ë¦?envelope = ì±„íƒ** (?œì¡°-ê°€ì¹?1ë²?. ?¤ì¸¡(35mJ/ì¶”ë¡ Â·latencyÂ·
  ?œì„±?„ë¥˜ 30mA) ê³„ì‚°ë§? **???™ìŠµ/ë³´ë“œ ?¤í—˜ ë¶ˆìš”.** ??NUCLEO sleep ?¬ì¸¡??ê¸ˆì?
  (11.4mA=ST-LINK ì§€ë°?ë¹„ë??? ??production bare-MCU **?°ì´?°ì‹œ??STOP(~14-24ÂµA)**
  ê¸°ì?. ?¼ë¬¸??**?„ì(ë°°í„°ë¦??˜ëª… projection "ì½”ì¸?€ N??, sleep=?°ì´?°ì‹œ??
  projection ëª…ì‹œ) + ?„ì(?œìˆ˜ ?¤ì¸¡ "35mJ/ì¶”ë¡ , duty???°ì‚° 1.5%") ????** ?µì‹¬=
  latency? param?œâ†’FEATHerê°€ FITS/DiPEë³´ë‹¤ ë°°í„°ë¦??¤ë˜ê°â†’?ë„ˆì§€Ã—robustness Pareto.
- **??PdM ì¡°ê¸°ê²½ë³´ ë¦¬ë“œ?€??= NO-GO.** êµ¬ì¡°??anomaly detection(residual?’ê²½ë³??´ë¼
  forecasting ?•ì²´???ë¦¬ê³?AD-baseline flank ?´ë¦¼ + small N(MetroPT ê³ ì¥ 3~4ê±? +
  ?¼ë²¨ ?Œì´?„ë¼?¸ì— ?†ìŒ(???¼ë¬¸?ë§Œ). **?™ê¸°/future-work ??ë¬¸ì¥?¼ë¡œë§?** ?œì¡°-ê°€ì¹?
  ???ë„ˆì§€ + ê¸°ì¡´ ?•í™•??robustness/ë°°í¬ë¡?ì¶©ë¶„.
- **ë¶€???•ì¸**: seq_len=96Â·H={96,192,336,720}; ?œì¡° ?€ì¹??„ë¡œ? ì½œ(?„ì› default+FFT
  period) ê³µì •???´ìŠˆ ?†ìŒ = settled. **Results ?ë¦„ ?•ì •**: Â§6.1 ?•í™•??mfg+Pareto)
  ?’Â?.2 robustness?’Â?.3 ë°°í¬?’Â?.4 ?ë„ˆì§€(???’Â?.5 LTSF(ê°•ë“±)?’Â?.6 ablation.
- **?°ì¶œë¬?*: `ltsf_baseline_results.xlsx` FEATHer* grid-ceiling ?‰ì„ ?„ì£¼ ETTh1(360)
  +prelim h2/m1/m2 ?¤ë¼?´ë¡œ ê°±ì‹ (min-MSE config MSE/MAE/CORR, ?˜ì´?¼ì´???¬ê³„??.
- **???œë²„ ?íƒœ (07-28, 3 ?¤ìœ• ?™ì‹œ):** **êµ??œë²„ GPU1 = mfg ë² ì´?¤ë¼???¤ìœ•
  RUNNING** (792 / **245 done** ??1%, FEATHer ?œì™¸ 11ëª¨ë¸; Steel ë¨¼ì?Â·N2O/MetroPT3
  ê¼¬ë¦¬; Dropbox 0?‰ì? ?±í¬ì§€?? LTSF ??32/32 ?„ì£¼). **êµ??œë²„ GPU0 = FEATHer mfg
  arch ?œì¹˜ RUNNING** (`FORCE_OMP=5 CUDA_VISIBLE_DEVICES=0 python
  search_feather_mfg.py --ngpu 1 --jobs_per_gpu 2`, 324??27configÃ—3dsÃ—4HÃ—1seed,
  ê²©ë¦¬ CSV `feather_mfg_search.csv`). **???œë²„ = FEATHer LTSF ê·¸ë¦¬??RUNNING** on
  ETTh1/h2/m1/m2+Exch/Weather. **CSV 3ê°??¬ë¼ ?™ì‹œ ?ˆì „.**
- **??FEATHer mfg HP = per-dataset ?œë‹ ê¸ˆì?(?€ì¹?Â·?œì¹˜?’ë‹¨??canonical ? íƒ.** ?œì¹˜
  1?Œë¡œ (a) per-dataset best[ì§„ë‹¨+one-config-gap ë¶€ë¡? + (b) mean-rank canonical vs
  base d8/k7/B3 ??ë·? arch 27ê°?d_state{4,8,16}Ã—kernel{3,5,7}Ã—num_bands{2,3,4},
  k=1 ?œì™¸), period=FFTÂ·lr/Î» ê³ ì •. ?„êµ¬ `search_feather_mfg.py`(personal `b9c9ed7`).
  ?ë‚˜ë©?`--summary`ë¡??ì •?’í™•?¤íˆ ?˜ìœ¼ë©?canonical êµì²´, ?„ë‹ˆë©?? ì?. robustness
  Plan B/sub-1K tier?œëŠ” ?°ì´???˜ì˜¨ ?? [[project-mfg-value-experiments-0728]]

**?¡âš¡ 2026-07-27b ê²°ì • (mfg lr ?œì¹˜ ?ê¸° ??default lr; ?ì„¸ =
`memory/project_mfg_lr_default_decision.md`):**
- **?œì¡° lr ?œì¹˜(?£a baseline + ?£b FEATHer)ë¥??„ë©´ ?ê¸°?˜ê³  ê°?ëª¨ë¸??  `_METHOD_DEFAULTS` lr???€.** ê²°ì •??ì¦ê±° = `lr_search.csv` 44ê°?mfg ?€?ì„œ
  default lr vs 5-lr ?œì¹˜ best-lr??val_loss ê°œì„ ??**ì¤‘ì•™ê°?0.1%**(?‰ê·  0.4%,
  ìµœë? 6.4%=MDMLPÂ·Steel ?˜ë‚˜; >3% 1/44Â·>10% 0/44). best-lr??default?€ 10ë°?  ?¬ë¼??val_loss??0.5%ë§??€ì§ì„ ??mfg??**loss ?œë©´??lr???€???‰í‰**, ?œì¹˜
  best???¸ì´ì¦?
- **??BASELINE?ë§Œ ?ìš©, FEATHer???œì™¸.** ë² ì´?¤ë¼??default??ê°??¼ë¬¸ ?€??  ê³µí‘œê°’ì´???•ë‹¹?˜ì?ë§? FEATHer???°ë¦¬ ëª¨ë¸?´ë¼ **ê³µí‘œ default ?†ìŒ**(1e-3?€ ?°ë¦¬ê°€
  ?„ì˜ë¡?ì°ì? ê°? ??"default ?¤í‚µ" ?¼ë¦¬ ?±ë¦½ ???? FEATHer??ì§„í–‰ ì¤‘ì¸ **ê·¸ë¦¬?œì„œì¹?  (ETTh1/h2/m1/m2+?•ì¥) ê²°ê³¼ë¡?config ?•ì • ???¬ìš©?ì? ?¨ê»˜ ë³´ê³  ë³„ë„ ë°œì‚¬**(arch=
  canonical mean-rank, period=FFT, lr=ê·¸ë¦¬?œê¸°ë°?. FEATHer??sub-1K??ë³‘ëª© ?„ë‹˜.
- **?¨ê³¼: ë² ì´?¤ë¼??mfg ë³??¤ìœ•(11ëª¨ë¸, `--exclude FEATHer`) ì¦‰ì‹œ ë°œì‚¬ ê°€??**
  `--num_seeds 3`(?œì¶œ?? ?˜ì¤‘??`--num_seeds 5` resume=ì´?5?œë“œ ?¼ë¬¸ê°? LTSF?€ ?™ì¼
  ?•ì±…) + `--save_model`. ë² ì´?¤ë¼??mfg lr?€ `_DATASET_OVERRIDES`??lr ?†ìŒ?’default
  ?´ë°±(period_len/cycleë§?override), **ì½”ë“œ ?˜ì • ë¶ˆìš”**. heavy(iTransformer 0?‰Â·OOM
  ?¸ê¾¹ì§? lr ?€ê¸??Œë©¸.
- **LTSF??ë¬´ì˜??* ??baseline?€ ê°??¼ë¬¸ per-dataset ê³µí‘œ lr ? ì?, FEATHer LTSF
  ?™ê²° OFAT ??? ì?. ??ê²°ì •?€ **mfg ì¶•ì—ë§?* ?ìš©, [[project-open-lr-policy]]ë¥?  mfg?ì„œ ?€ì²?
- `lr_search.csv`???ê¸° ?„ë‹ˆ??**??ê²°ì •??ì¦ê±°ë¬?*ë¡?ë³´ì¡´(?¬ìš©??ê²°ì •: ?? œ X).
  LR_SEEDS 2?? TEMP ??`5398d89`)?€ ?´ì œ ë¬´ì˜ë¯?mfg lr ???? correctness ë¬´í•´,
  ì²?†Œ ??[2025,2026] ?ë³µ). ??SparseTSF default(0.02)???œì¹˜ ê²©ì ë°–ì´??ê³µí‘œê°’ì´??? ì?.
- **???œë²„ ?íƒœ (07-27b ë°œì‚¬ ?????œë²„ ê°€??:**
  - **êµ??œë²„ = mfg ë² ì´?¤ë¼???¤ìœ• RUNNING** ??`--exp_tag main --save_model
    --exclude FEATHer --num_seeds 3 --ngpu 1 --jobs_per_gpu 3` (11ëª¨ë¸Ã—6mfgÃ—4HÃ—3seed
    =792????`fcst_results.csv` mfg ??. seed 3=?œì¶œ?? ?˜ì¤‘ `--num_seeds 5` resume=
    ì´?5?œë“œ. `--save_model`=robustness ì²´í¬?¬ì¸??
  - **???œë²„ = ê·¸ë¦¬?œì„œì¹?RUNNING** ??ETT(ETTh1 ~?„ì£¼/h2Â·m1Â·m2 ì§„í–‰) + **Exchange??    Weather ?•ì¥(ë³‘ë ¬, ETT?€ byte-identical 360ê²©ì; Electricity/Traffic?€ ?ë¦¼+
    non-sub-1K???¤í‚µ)**. **lr ?œì¹˜ ì¤‘ì?**(mfg lr ?ê¸°ë¡?ë¬´ì˜ë¯?.
  - **?°ì¶œë¬?*: `results/ltsf_baseline_results.xlsx` ê°±ì‹  = 11 ë² ì´?¤ë¼?????œë“œ?¼ì¬
    seed?‰ê· ) + **FEATHer\* grid-ceiling ??*(sub-1K ì¹¸ë³„ min-MSE=test-peek, ETTh1ë§?    ? ë¢°/h2m1m2 preliminary/ë¹?ETT ê³µë°±; 1?±ì´ˆë¡Â??±ë…¸?? FEATHer??ETTh1@336ë§?1??    =clean ?ˆë??•í™•???¹ë? ?„ë‹˜ ?¬í™•??. ??LTSF ë² ì´?¤ë¼??11ëª¨ë¸ 32/32 ?„ì£¼(?œë“œ?¼ì¬:
    4ëª¨ë¸ 5?œë“œÂ·2ëª¨ë¸ 24Ã—5+8Ã—3Â·5ëª¨ë¸ 3?œë“œ).
  - **?¤ìŒ ê²Œì´??*: mfg 792???„ì£¼?’robustness ?¤ìœ• / ê·¸ë¦¬???„ì£¼?’FEATHer config ?ì •
    (kernel=1 arch-degenerate ì§€ë°°ë¼ canonical êµì²´ X ?ˆìƒ)?’FEATHer mfg+LTSF ë°œì‚¬.
- **??dispatch ?œì„œ ?µì¼ (ì»¤ë°‹ `67d69ea`??3c4d27f`, personal push):** `run_forecast.py`
  `_DATA_DISPATCH_ORDER` ??mfg+ltsf **?µì¼**, ?‘ì? ?°ì´?°ì…‹ ë¨¼ì? ??ê²??˜ì¤‘(tier?ˆí–‰Ã—ì±„ë„)
  ??ë¹ ë¥¸ ê²°ê³¼ ë¨¼ì?Â·?ë¦° ê¼¬ë¦¬ê°€ ??ë§‰ìŒ. mfg: Steel?’BlastFurnace?’WindSCADA?’EWELD??  **MetroPT3?’N2O**(N2O 210kÂ·MetroPT3 306k?‰ì´ ìµœì¥); ltsf: Exchange?’ETTh1/h2/m1/m2??  Weather??*Electricity?’Traffic**(ì±„ë„ 321Â·862ê°€ ìµœì¥). ?´ì œ **dataset-major**(???°ì´?°ì…‹
  11ëª¨ë¸ ???Œê³  ?¤ìŒ). **êµ??œë²„ ?ìš©=?Œë˜ mfg Ctrl+C(`37e4758`)??git pull`?’ê°™?€ ëª…ë ¹
  ?¬ì‹¤??resume+???œì„œ)**. `run_forecast.py`ë§?push(?´ë˜???ˆë? ?œì™¸).

**??2026-07-27 ?…ë°?´íŠ¸ (MCU ë°°í¬ ?¤ì¸¡ ?„ë£Œ ???¸ë™ ?¨ì¼ ?ŒìŠ¤ =
`deployment/EDGE_EXPERIMENT.md`, ?ˆì°¨??= `deployment/board/BOARD_RUNBOOK.md`):**
- **ë³´ë“œ(NUCLEO-F103RB) ?„ì°© ??5ëª¨ë¸ ?¤ë³´???¤ì¸¡ ?„ë? ?„ë£Œ** (ê¸°ì—¬ 4ë²?"Cortex-M3
  ë°°í¬ ì¦ê±°"ê°€ ì¶”ì •?’ì‹¤ì¸?. CubeIDE ?„ë¡œ?íŠ¸ = `C:\Users\lee\STM32Cube_projects\
  blink_test`(repo ë°?. golden 5ëª¨ë¸ **bit-exact PASS**. **FEATHer @F103RB
  (p12/453, -O2 soft-float): 349ms / ì´?flash 17.6KB(learnable 1,812B+twiddle
  768B=Â§8 ?¤ì¦) / ~35mJ/ì¶”ë¡  / ?œì„± 30mAÂ·?€ê¸?11.4mA**. ?ˆì´?´ì‹œ: pers 0.08 /
  STSF 47 / FEATHer 349 / FITS 796 / DiPE 1312 ms (**?Œë¼ë¯¸í„°?œâ‰ ?ˆì´?´ì‹œ??*=?¼ê±°;
  ?œì„±?„ë¥˜??5ëª¨ë¸ ?„ìˆ˜ 29-31mA=ëª¨ë¸ë¬´ê? ?¤ì¦). ??ë°°í¬ ?ì„¸??**CLAUDE.md??  ???€**(ê·œì¹™) ???„ë? `EDGE_EXPERIMENT.md` 07-27 ë¡œê·¸. **?¨ìŒ = FEATHer p24(489p)
  ê³¨ë“  ?¬ìƒ?????¬ì¸¡???œë²„ config ?•ì • ?? + ?¼ë¬¸ Â§7/Â§8 ?œÂ·ë¬¸??**
  [[project-deployment-layer3-plan]]

**??2026-07-26/27 ?…ë°?´íŠ¸ (ì¤‘ê°„?ê? + lr ?ë„?? ?ì„¸ = `memory/project_grid_search_pivot_0722.md`):**
- **?œë²„ ?íƒœ(?„ì¬)**: êµ??œë²„ = ??LTSF(`fcst_results.csv`) ê³„ì†. ???œë²„ =
  ETTh1 ê·¸ë¦¬??`grid_search.csv`) + ?£a ?œì¡° lr(`lr_search.csv`) ?™ì‹œ. CSV??  `Dropbox\WDSLab\05. ?¤í—˜\FEATHer\results\`ë¡??±í¬??ë¡œì»¬???´ëŒ ê°€??
- **??LTSF ì»¤ë²„ë¦¬ì?**: 11ëª¨ë¸ ê±°ì˜ ?„ì£¼(iTransformerë§?Traffic/Weather ë¯¸ì™„,
  TimesNet 2?€). ???œë“œ ?¼ì¬ ??6ëª¨ë¸ 5?œë“œ(??Â·5ëª¨ë¸ 3?œë“œ(07-16 ?´í›„). ìµœì¢…??  ???œë“œ ?•ì±… ?•ì • ?„ìš”.
- **ê·¸ë¦¬??ì»¤ë²„ë¦¬ì?**: **ETTh1 95.1%(1,370/1,440)**, ?¨ì? 70 = `grid_ds8_k9_*`
  ê¼¬ë¦¬(?•ë ¬??ë§ˆì?ë§‰ì´???•ìƒ). ETTh2/m1/m2 = ë°©ê¸ˆ ?œì‘(~33 config). ?¬ìš©??  **?œì°¨ ?¬ì‹¤??*(ETTh1?’h2?’m1?’m2, `foreach($d){run_grid_search.py --data $d ??`).
- **ETTh1 ë¶€ë¶„íŒ??95%Â·1?œë“œ)**: ?•ì‹??"grid BEATS base(val)"ì§€ë§?**?¹ìê°€
  arch-degenerate** ??base d8/k7/p12/B3 = sub-1K 272ì¤?val **160??ì¤‘ìœ„)**, best
  sub-1K `grid_ds16_k1_p4_B3`(981p) ?val+0.0165Â·MSE??.6%, **top8??kernel=1
  ì§€ë°?*(DTK=temporal kernel ì£½ì„). clean?•í™•?„ëŠ” ?œê·¸?ˆì²˜ arch ë­‰ê°¤ ?Œë§Œ????  **robustness-not-accuracy ì¦ê±°ì§€ canonical êµì²´ ê·¼ê±° ?„ë‹˜, ?¬ë™ê²?ê¸ˆì?**. ê³µì •
  ??val? íƒ) FEATHer = ETTh1 **9/13??*Â·ETTh2 **10/12??*(?ˆSparseTSF), ?¤ë¼??  (test-peek)=ETTh1 1?„ì?ë§?ë°˜ì¹™. ?€ë§?= accuracy-per-param + robustness.
- **?£a ?œì¡° lr = 72%(237/330 mfg?€)**: ê°€ë²¼ìš´ 7ëª¨ë¸ ?„ë£Œ, **ë¬´ê±°??4ê°?  (iTransformer 0Â·TimeMixer/TimesNet Steelë§ŒÂ·TQNet ë¶€ë¶?** ?¨ìŒ(=OOM ?¸ê¾¹ì§?  ëª¨ë¸??. ?ë¦¼?ì¸ = heavy Ã— big-ROW(N2O 210kÂ·MetroPT3 306k ?? ì±„ë„?€ ?‘ìŒ
  6??8) Ã— 50ep Ã— 5lr Ã— 2seed.
- **??lr ?ë„??(ì»¤ë°‹ `5398d89`, personal push)**: `LR_SEEDS 2??` (**TEMP ??  lr ?ë‚˜ë©?`[2025, 2026]` ?ë³µ ?„ìˆ˜**) + `run_lr_search.py --model
  iTransformer,TimeMixer,TimesNet,TQNet --num_epochs 20 --jobs_per_gpu 1` =
  **4~6ë°?*. **?ˆì „ê·¼ê±° = lr?€ ? íƒ?„êµ¬**(ì¶œë ¥=best-lrë§Œâ†’`_DATASET_OVERRIDES`,
  ìµœì¢…?«ì???¤ì„œ 50ep/3seed ?¬ì‚°ì¶???ep/seed ì»·ì´ ë³´ê³ ê°?ë¬´ì˜?? per-cell
  ?…ë¦½?´ë¼ ë¶ˆê³µ??X). 1seed??pullë¡?ë°›ê²Œ ì»¤ë°‹ ?„ìš”(ë¯¸ì»¤ë°‹ì? pull ????.
  **??lr `--summary` ??LR_SEEDS ?ë³µ + heavy best-lr??ê²©ì??0.0001/0.01)
  ?„ë‹Œì§€ ?•ì¸.** ??ë¯¸ì»¤ë°?ë¡œì»¬ ?¬ì „ = ?´ë˜??ë² ì´?¤ë¼??+ ?Œì»¤ ?„ë¡œë¸??Œë˜ê·?

**?¡âš¡ 2026-07-22 ?…ë°?´íŠ¸ (?´ë²ˆ ?¸ì…˜ ??HP ë°©í–¥?„í™˜; ?ì„¸ = `memory/project_grid_search_pivot_0722.md`):**
- **?¬ëŒ ë£¨í”„ 1 ?¤í–‰??*: ??OFAT `--validate`??--summary` ??**LTSF 8???¬ë™ê²?*
  (ì»¤ë°‹ `0768e9b` + push ijaehun; validateê°€ raw combo 4/8 ê¸°ê° ??base+1ì¶??´ë°±).
  **Î» flat ?¬í™•????confound CLOSED**; **canonical mfg d8/k7/p12/B3 ? ì?**.
  (CLAUDE.md 07-21 ?ìŠ¤?¸ë„ ??ì»¤ë°‹???™ë´‰??)
- **TPAMI ê°?ë°œê²¬ (?•ì§)**: ?¬ë™ê²?FEATHerê°€ ETTh1/h2 **clean 8/9??*
  (SparseTSF 69pÂ·DiPE 511p?œí…Œ ì§?, ??TPAMI FEATHerë³´ë‹¤ **~5-7% ?˜ì¨**,
  **SPK@720 ?¥ê¸°?ˆì¸¡ ì£¼ì¥ ?¬í˜„ ????*(ê°™ì? L=96). thesis(robustness-not-
  accuracy)?€ ëª¨ìˆœ?€ ?„ë‹ˆ??"ETT clean ?¹ë¦¬/??720 ê°•ì„¸" ì£¼ì¥?€ **?€??*.
- **?„ë¡œë¸Œë¡œ ?ì¸ ê·œëª… (ë¡œì»¬ ETTh1, 3?œë“œ)**: lr(0.01 ë¶ˆì•ˆ??Â·loss(MSEê°€ L1ë³´ë‹¤
  ?˜ì¨)Â·**AdamW+wd+lr0.01(TPAMI ?•í™•?¬í˜„)=0.410 ???˜ì¨** ??**?´ë–¤ training
  recipe??TPAMI 0.373 ?Œë³µ ëª???* ????TPAMI ETT = **legacy ?„í‹°?©íŠ¸**(??  ?™ê²°????ì¶•ë“¤??near-best). ?Œì»¤??`--optimizer/--weight_decay` ?„ë¡œë¸?  ?Œë˜ê·?ì¶”ê?(backward-compat, **ë¡œì»¬ ë¯¸ì»¤ë°?*).
- **ê²°ì •: OFAT ??ì¡°ì¸??ê·¸ë¦¬?œì„œì¹?* (? ì¼ ?¨ì? ?ˆë²„ = arch ?í˜¸?‘ìš©).
  **grid > Optuna**(?‘ì? ?´ì‚°ê³µê°„+ì´ˆì†Œ??ëª¨ë¸+ë°©ì–´??ë³‘ë ¬). **ë¶ˆê?ì¹?= val ? íƒ**
  (test-peek ê¸ˆì?). **ê³µì •??= ëª¨ë“  ëª¨ë¸ per-dataset ?œë‹?´ë¼ ?€ì¹?*(OFAT????  ë²„ë¦¼ ??ë¦¬ë·°??R2#23 ë¯¼ê°?„ë¶„??+ num_bands/Î» flat ì¦ê±°ë¡??¬ì „???°ì„).
  **2ì¸?*: LTSF=per-dataset ê·¸ë¦¬??/ MFG=**mean-rank canonical 1ê°?*(?£ì?
  "one sub-1K configê°€ ëª¨ë“  ?¼ì„œ?? ?¤í† ë¦?ê°•í™”; period=per-dataset FFT, lr=lr?œì¹˜).
- **`run_grid_search.py` ?œì‘Â·push** (`0f6dd63`+`0c2576a`): arch 4ì¶??„ìˆ˜,
  Î»/lr/loss ê³ ì •, `--seeds`(1?œë“œ ? íƒ), resumable/`--check`/`--summary`.
  `results/grid_search.csv`. **0722b ê·¸ë¦¬???•ì¥(`fbe077d`, personal)**: ê·¹ë‹¨-?Œí˜•
  ì½”ë„ˆë¡??“í˜ ??d_state{**2**,4,8,16}Â·kernel{**1**,3,5,7,9}Â·period{**2,4**,6,8,12,24}
  Â·num_bands{2,3,4}=**360ì¡°í•©**, GRID_PREDS=**[96,192,336,720]**(period ?„ë?
  gcd=48 ?˜ëˆ  ?œë¡­ 0). ê·¹ë‹¨ì½”ë„ˆ(d_state2/k1/p2/H720) ?¤ëª¨??? í•œ(kernel=1?€ DTKë§?
  FFTê²Œì´??kernel=5 ?˜ë“œì½”ë”©; period=1?€ SPK degenerate???œì™¸). ì»¤ë°‹=
  run_grid_search.py **?¨ë…**(?´ë˜???„ë¡œë¸?ë¯¸ì»¤ë°?? ì?).
- **RUNNING = go/no-go (???œë²„ ETTh1 ê·¸ë¦¬??**: **??FEATHer ì¤‘ì???* ??ETTh1
  ê·¸ë¦¬?œê? **?£a ë² ì´?¤ë¼??mfg lr(`lr_search.csv`)ê³??™ì‹œ** ?¤í–‰ ì¤?(CSV ?¬ë¼ ?ˆì „,
  ?¨ì¼ GPU???ë¦¼Â·RAM ì£¼ì‹œ). **0722bë¡??Œì¼??1,440??*(360Ã—4HÃ—1?œë“œ). ??˜ ETTh1/h2
  ?‰ì? `fcst_feather_ltsf.csv`??**?„ë™ê²??ˆí¼?°ìŠ¤ë¡?ë³´ì¡´**(ë¹„êµ?? ?? œ ê¸ˆì?).
  êµ??œë²„ = ??LTSF(`fcst_results.csv`) ê³„ì†. ë¡œì»¬ = ?°ë¦¬ ?¤í—˜ ?†ìŒ. **?œë²„ ?¬ë°œ??*=
  ??ê·¸ë¦¬??Ctrl+C??git pull personal main`??run_grid_search.py --data ETTh1
  --pred_len 96,192,336,720 --seeds 2025 --jobs_per_gpu 1`; **resumeê°€ ??H96 144??  ? ì?**(??44?‚ìƒˆ360)Â·?˜ë¨¸ì§€ë§?ì±„ì?(purge ë¶ˆìš”). `--summary`??ê°™ì? `--pred_len/
  --seeds`+`PYTHONUTF8=1`. **?ì •**: **grid?«base(val)** ??LTSF-8+mfg-6 ?•ì¥Â·?¬ë™ê²°Â?  ?¢â‘¤ ?¬ì‹¤??loop-1 ?¬ì˜¤??. **base ties/wins** ??OFAT ì¶©ë¶„, ???™ê²° ? ì?Â·robustness.
- ??**???œë²„ ?£a lr ?¸ê¾¹ì§??¬ìš©??ë³´ë¥˜)**: MDMLP_EIA/MetroPT3/H96ê°€ CUDA
  `illegal memory access`ë¡?ì£½ìŒ(async??`.item()`/`empty_cache`????ê²ƒë¿; ê·?  ?„ë¡œ?¸ìŠ¤ ì»¨í…?¤íŠ¸ë§??¤ì—¼, ê·¸ë¦¬?œëŠ” ë³??„ë¡œ?¸ìŠ¤??ë¬´ê?). ? ë ¥=?¨ì¼32GB GPUë¥?  ê·¸ë¦¬???£a ìµœì¤‘??MDMLP 1.6M)???˜ëˆ  ?¨ì„œ ë©”ëª¨ë¦??•ë°•. ê°€ì§œí–‰ ?†ìŒ(exit1+
  MASK-NAN-GRAD ê°€?œâ†’missing ? ì?, resume ?¬ì‹œ??. **?˜ì¤‘??ê·??€ë§?ê²©ë¦¬+
  `CUDA_LAUNCH_BLOCKING=1`ë¡??¬í˜„**(?¨ë…?µê³¼=ë©”ëª¨ë¦¬â†’?£a jobs ì¤„ì´ê¸?/ ?¨ë…?¤íŒ¨=
  ëª¨ë¸Â·?°ì´??ë²„ê·¸).
- ??**ë¯¸ì»¤ë°?ë¡œì»¬**: ?´ë˜??ë² ì´?¤ë¼??list_models 15) + ?Œì»¤ ?„ë¡œë¸??Œë˜ê·?  (--optimizer/--weight_decay). ?´ë˜?ì? ?¬ì „??push ê¸ˆì?(?¤ìœ• ?ë™?•ì¥ ?¸ë¦¬ê±?.
  (07-23: throwaway `results/probe_*.csv` + repro??CSV ?? œ, LTSF ?¬í˜„ê²°ê³¼??  `results/ltsf_baseline_results.xlsx`ë¡??•ë¦¬ ??MSE/MAE/COR ?œì ë³?+ Coverage ?œíŠ¸.)

**??2026-07-16 ?…ë°?´íŠ¸ (?”ì•½ ???ì„¸???„ë˜ ë¸”ë¡/Decision history):**
- **ë² ì´?¤ë¼??5?? ?œë“œ (êµìˆ˜ ê²°ì •)**: `--exclude FEATHer` ?¤ìœ•??`--num_seeds 3`
  ?¼ë¡œ ??¶° ?œì¶œ ?°ì„ , **ë¦¬ë¹„?„ì—??`--num_seeds 5`ë¡?ë³´ê°•**(3?œë“œ=[2025,2026,
  2027]???œë“œ??resume???¨ì? 2ê°œë§Œ ì±„ì?; base_seed/exp_tag ë¶ˆë? ?„ìˆ˜). **??LTSF??  3?œë“œë¡??¬ë°œ?¬ë¨.** ???œì¡° ë³??¤ìœ•?€ FEATHer ?¬í•¨ ?µì§¸ë¡?`--num_seeds 3`(ë©”ì¸
  ?Œì´ë¸?12ëª¨ë¸ ?œë“œ ?¼ê???. ?£a lr?€ ?ë˜ 2?œë“œ ê³ ì •?´ë¼ ë¬´ê?. **ì½”ë“œ ?˜ì •
  ?†ìŒ**(CLI ?Œë˜ê·? ?¤ìš´?¤íŠ¸ë¦??„êµ¬ ?„ë? `nunique` ?™ì  ì§‘ê³„, wilcoxon?€ ?€ ?˜ì–´
  ?€ë§ì´??3?œë“œ??? ì˜??ì¶©ë¶„). ??OFAT ? íƒ?€ 3?œë“œë¡??˜ì? ë§?ê²??ˆì •??.
- **?ê³  ?ìŠ¤???ŒíŠ¸ ?¬ì‘??(?œë²„ ë¬´ê?)**: `manuscript/drafts/*_jms.md` 5ê°?  (abstract/intro/related/datasets/methodology+theory)ë¥?mfg-6 + A+C ê³ ì†-"?¤ë¦¬"
  ?„ë ˆ?´ë°(?™ê¸°ë§? ?°ë¦¬??slow-process ?‰ê?) + **robustness ë³´ìˆ˜ ?œìˆ ** + SWIFT
  ì°¨ë³„??+ half-RevIN + ?´ë¡  ?¸ë¦¼(R8 placeholder Theorem 3/4/5 ?œê±°)?¼ë¡œ ê°±ì‹ .
  ??`0X_*_draft.md`??IoT-J ?„ì¹´?´ë¸Œ. ?ì„¸: `memory/project_manuscript_jms_drafts.md`.
  ?¨ì? ?ìŠ¤?? Conclusion ë¦¬í”„?ˆì„ + Appendix ì¦ëª… 2ê°?+ Algorithm 1 ê²€??
- **ê³ ì† ?„ë ˆ?´ë° ?ì •**: HR-SHM(ê³ ì† ?¼ì‹±?’ë‹¤?¤í… ?ˆì¸¡, decimation???´ë²¤?¸ë?
  aliasing) = ?¤ì¬ ë¬¸í—Œ?´ë‚˜ ?°ë¦¬ ?°ì´?°ëŠ” slow-process ??**?™ê¸° ?¸ìš©ë§?*, ?°ë¦¬
  ?‰ê? regime?´ë¼ ì£¼ì¥ ê¸ˆì?("horizon=step count" ?¤ë¦¬ë¡?ë´‰í•©). EWELD = ê¸ˆì†?°ì—…
  ?„ë ¥**ë¶€??*(??ê·¹í•œê¸°ìƒ ë¶€??) + ê·¹í•œê¸°ìƒ ?´ë²¤??ì°?

**??ì§€ê¸??œë²„ ?íƒœ (07-18b 2-?œë²„ ?¤ìœ•-ë¶„í•  ??ì²´ì¸???¨ì¼ ?ŒìŠ¤??`ROADMAP.md`):**
**?„ìš© GPU 1ê°œì§œë¦????œë²„ ?•ë³´ ???¤ìœ•ë³?CSV ë¶„ë¦¬ ?´ìš©??ë³‘ë ¬ ë¶„ë°°**(?¬ìš©??ê²°ì •). ? â‘¡?£a??CSVê°€ ?¬ë¼(`fcst_results`/`hp_search`/`lr_search`) ?™ì‹œ ?ˆì „.
- **êµ??œë²„ (Linux, ê³µìœ  128GB, GPU1)** = **??LTSF** (`FORCE_OMP=5
  CUDA_VISIBLE_DEVICES=1 python run_forecast.py --group ltsf ??--ngpu 1
  --jobs_per_gpu 2 --exclude FEATHer --num_seeds 3`). `fcst_results.csv`.
- **???œë²„ (Windows, ?„ìš© 32GB, GPU 1ê°? env `lee`, `C:\Users\user\Desktop\
  FEATHer`)** = **??FEATHer OFAT + ?£a ?œì¡° lr ????* (CSV ?¤ë¦„ ???™ì‹œ ?ˆì „).
  êµ??œë²„??ë©ˆì¶°?ˆë˜ hpë¥?`results/hp_search.csv` ë³µì‚¬???´ì‚¬(resume). ?¨ì¼ GPU
  ?????µì…˜ ìµœì†Œ: **??OFAT = `python run_hp_search.py`(jobs 1 ? ì? ??elec/traffic
  ?Œì»¤ê°€ RAM ë¬´ê±°?€)**, **?£a lr = `python run_lr_search.py --exclude FEATHer
  --jobs_per_gpu 3`(mfg??RAM ê°€ë²¼ì›Œ ë³‘ë ¬ 3ë°? GPU 10%Â·RAM ~8GB ?¬ìœ ???´ê²Œ ?ë„
  ?ˆë²„)**. RAM?€ ??OFATê°€ Traffic(862ch) ?¡ì„ ?Œë§Œ ì¡°ì‹¬(`free`). ?£a??mfg
  data/ + `lr_search.csv` ?„ì†¡ ?„ìš”(ë±…í¬ Steel/EWELD/BF resume).
- **?????œë²„ `bash`??bash 4+ ?„ë‹˜** ??`setup_baselines.sh`??`declare -A`ê°€
  ì¡°ìš©???¤íŒ¨("division by 0" ??clone 0ê°œì¸??"All present" ê°€ì§??±ê³µ). **11ê°?  upstream repo??cmd?ì„œ ?˜ë™ `git clone --depth 1 <url> baselines/<N>/<N>-main`**
  (URL = `setup_baselines.sh`??REPOS). reformer_pytorch???´ë? ?¤ì¹˜?? ???˜ë©´
  ë² ì´?¤ë¼?¸ì´ upstream import?ì„œ ì£½ìŒ(?? DiPE_Linear `No module 'timeprophet'`).
- **??ê·œì¹™: ê°™ì? CSV????ë°•ìŠ¤?ì„œë§?** `lr_search.csv`=???œë²„ë§?êµ??œë²„ lr ê¸ˆì?),
  `hp_search.csv`=???œë²„ë§? `fcst_results.csv`=êµ??œë²„ë§? ìµœì¢…??ê°?CSVë¥???  results/ë¡?ëª¨ìœ¼ê³?`check_duplicates.py` ê²€ì¦? `FORCE_OMP` env = ì»¤ë°‹ `880c83e`
  (worker??OMP/MKL ?¤ë ˆ??ê³ ì • ??ë°•ìŠ¤ ê³µìœ  ?œë§Œ ?„ìš”).
- **??07-19 `d56aa05`(MASK-NAN-GRAD) ?¸ì‹œ???????œë²„ pull ?„ìˆ˜, ?¹íˆ ??  ?œë²„??pull ???£a ?¤í–‰ ê¸ˆì?**(ë§ˆìŠ¤??3ì¢…ì´ ê°€ì§?0.0?¼ë¡œ ??ê¸°ë¡??.
  pull ???´ë¦° lr_search.csv(844?? ë°°ì¹˜ ??`run_lr_search.py --exclude FEATHer`
  ?¬ë°œ????resume??ë§ˆìŠ¤??3ì¢?ë¯¸ì°©??ëª¨ë¸ ?¬ì¶©??
**?¤ìŒ ?¬ëŒ ë£¨í”„(ë£¨í”„ 1)**: ?¡Â·â‘£a ?„ë£Œ ??`--validate`(32) ??`PYTHONUTF8=1
--summary` ì¶œë ¥ ?„ë‹¬ë°›ì•„ LTSF 8???¬ë™ê²?+ 07-06 diff ê¸°ë¡ + canonical mfg
?„í‚¤?ì²˜(d8/k7/p12/B3) ?¬í™•??+ Î» ìµœì¢… ?ì • ??ì»¤ë°‹ ??pull ????FEATHer LTSF
160 + ?£b FEATHer lr ë°œì‚¬.

**Positioning (settled):** Fresh **JMS** submission (TPAMI reject ??IoT-J ??JMS manufacturing pivot). Main table = **6 manufacturing datasets** (Steel Â·
WindSCADA Â· BlastFurnace Â· N2O Â· MetroPT3 Â· EWELD), LTSF-8 demoted to a
generalization section, short-horizon PdM (CMAPSS/CMAPSS3/PMSM) **NOT in the
paper** (lr-banked shelf, `--group cmapss`). FEATHer's 453p > SparseTSF 41p is
justified by **ROBUSTNESS, not accuracy** (OFAT num_bands FLAT on clean).
Framing guard: NOT "more params = more robust" but "41p linear lacks headroom;
FEATHer's inductive bias buys graceful degradation at sub-1K". R3's
"long-term-on-edge is contrived" attack resolved via **A+C framing**: (A)
industrial edge NEEDS on-device long-horizon (air-gap/OT isolation, per-sensor
scale, outage autonomy ??our datasets are the evidence), (C) edge = param
budget, horizon = task, orthogonal axes; "why 1K" ??accuracy-per-param Pareto
frontier. 12-model baseline lineup unchanged.

**??CODE AUDIT (2026-07-14c) ??full-repo review done, findings in
`manuscript/notes/code_audit_2026-07-14.md`. ??THE CRITICAL BATCH IS FIXED
(same session, all validated end-to-end):**
- ??**P0-1 robustness override plumbing**: `run_robustness.py` dispatch now
  mirrors the training-time `get_method_defaults`+`get_dataset_overrides`
  merge ??`--model_overrides`; the worker gained the flag +
  `_apply_overrides()`; `gate_logging.py` applies the same merge itself.
  Verified live: trained ETTh1 d16/k3/p8/B4 ckpt loads + clean MSE
  reproduces training-time test MSE exactly.
- ??**P0-2 corruption seeding**: salted `hash((ft,sev))` ??  `utils/noise.corruption_seed()` (deterministic index-based). Verified:
  re-run reproduces corrupt-MSE to 6 decimals across processes.
- ??**P0-3 robustness resume**: `load_robust_set` counts DISTINCT
  (fault,severity) pairs; worker skips conditions already in the CSV
  (per-condition resume ??no duplicate rows on retry). Verified via
  truncate-and-refill test.
- ??**P1-1 spectral loss band order** fixed (`bands[S-1-s]` ??POINT owns the
  highest slice, LOW the lowest). Synthetic check: correct placement ??loss
  0.0, swapped ??2.0. Frozen LTSF OFAT rows kept (Î» axis was flat).
- ??**P1-2 Weather split** ??standard 0.7/0.1/0.2 (legacy hard-coded branch
  + dead PM/PEMS_BAY/METR branches deleted). ALL pre-fix Weather rows are
  invalid ??purge + re-run (see server procedure below).
- ??**P1-3 RMSE** = sqrt(global MSE) in both workers (+ NaN-aware CORR/R2
  accumulation). Old rows: derive sqrt(MSE) at table time.
- ??**P2-1 worker exit code**: training worker now exits 1 if any seed run
  failed (kills the reformer-style silent-skip class). **P2-2 CSV header
  race**: atomic `open(mode="x")` create-with-header in both workers.
- ??**P2-3..8 + P3 batch (2026-07-14d) ??AUDIT FULLY CLOSED** (23-check
  smoke suite passed + all 4 orchestrators `--check` exit 0; details in the
  audit doc UPDATE 2 block):
  - P2-3 check_duplicates schema-detects the robust CSV (key extends with
    train_exp_tag/fault_type/severity ??`--fix` now safe on it). P2-4
    explicit FEATHer CLI arch flags beat `_DATASET_OVERRIDES` (flags default
    None; sweep behavior byte-identical). P2-5 best_state=None raises (no
    row/ckpt; exits via P2-1 path). P2-6 mCE drops no-ref + model-incomplete
    cells for ALL models (warns; emits n_cells). P2-7 ablation_table = one
    table per dataset (no raw-MSE pooling). P2-8 lr/hp `--summary` flag seed
    under-coverage; hp combo "complete" requires full seeds.
  - P3: prep `_fill_gaps` gap policy (linear interp ??~1 h, hold-last
    beyond ??no lookahead/linear bridging of outages; audited max-gap
    asserts); BlastFurnace winsorize bounds from the TRAIN region (also
    finally clips the late Tp4 ~3311 sentinel: hi 3313??55); EWELD D=14
    assert; metrics.py np.divide out= init; severity tables +
    corruption_seed ??torch-free `utils/severity.py` (noise.py re-exports,
    robust_summary shares the object, utils/__init__ now lazy PEP 562);
    FEATHer half-RevIN (means detached, stdev not) KEPT + documented as
    deliberate (changing it would invalidate all frozen OFAT rows); stale
    docstrings/comments swept.
  - ??**DATA REGENERATED** under the new gap policy ??cells changed:
    MetroPT3 16.6% / N2O 2.7% / WindSCADA 1.1% / BlastFurnace 0.9% /
    EWELDÂ·Steel 0%. Pre-fix copies kept at `data/<name>/data.csv.preP3bak`.
    ??transfer the 4 changed data.csv to the server and purge/refill their
    lr_search rows (added to SERVER PROCEDURE below).

**??SERVER PROCEDURE after pulling the fix commit** (order matters):
```bash
git pull   # (at a natural pause; running workers finish on old code harmlessly)
# invalidated rows ??resume refills them:
python tools/audit/purge_rows.py --csv results/fcst_results.csv --data Weather --apply
python tools/audit/purge_rows.py --csv results/hp_search.csv    --data Weather --apply
python tools/audit/purge_rows.py --csv results/fcst_results.csv --model FEATHer --apply
python tools/audit/purge_rows.py --csv results/lr_search.csv    --model FEATHer --apply
# (07-14e ê²°ì • ??Î» ?¬ë¼?´ìŠ¤ ë¶€ë¶„í¼ì§€ë¥??€ì²? FEATHer OFAT ?„ë©´ ?¬ì‹¤??
# hp_searchë¥??µì§¸ë¡?ë¹„ìš°ê³?run_hp_search.pyê°€ 512???„ë? ?¬íƒ??# (êµì •??spectral loss + ?œì? Weather split ?„ë˜?ì„œ):
python tools/audit/purge_rows.py --csv results/hp_search.csv --model FEATHer --apply
# (P3 data regen, 2026-07-14d) transfer the 4 regenerated data.csv files
# from local (WindSCADA / BlastFurnace / N2O / MetroPT3 ??Steel/EWELD
# unchanged), THEN purge their lr rows so lr is re-selected on the new data
# (~960 of 2,160 lr runs; MetroPT3 changed 16.6% ??too big to caveat away):
python tools/audit/purge_rows.py --csv results/lr_search.csv --data WindSCADA,BlastFurnace,N2O,MetroPT3 --apply
# then just re-run the same sweep commands ??resume fills the gaps.
# Weather + lambda OFAT slices (hp_search) refill via run_hp_search.py;
# re-freeze the Weather row in _DATASET_OVERRIDES from --summary afterwards.
#
# (MASK-FIX, 2026-07-15) observed-only scoring ??transfer + re-refill:
#   1. transfer the 3 regenerated data.csv (WindSCADA / N2O / MetroPT3 ??now
#      hold-last, no interp) AND the 4 companion mask.csv (WindSCADA / N2O /
#      MetroPT3 / BlastFurnace) to data/<name>/.  (EWELD unchanged: 0 imputed.)
#   2. purge + refill lr for the 3 whose data changed (BF's 1-cell mask is
#      below noise -> its rows stay valid; EWELD unchanged):
python tools/audit/purge_rows.py --csv results/lr_search.csv --data WindSCADA,N2O,MetroPT3 --apply
#   (this SUPERSEDES the 07-14d BlastFurnace lr purge line above ??BF no longer
#    needs a refill under MASK-FIX; the other 3 do, for both the P3 regen and
#    the interp->hold+mask change.)
```

**Evidence state (verified from hp_search.csv)**: only **num_bands is
confirmed FLAT on clean** LTSF-8 (B2/B3/B4 mean MSE 0.4505/0.4418/0.4457).
Gate on/off clean effect UNTESTED (needs ablation sweep); ALL robustness
UNTESTED (sweep unrun). The robustness thesis = 1 confirmed fact + an unproven
bet ??the robustness sweep is the #1 risk.
??**Î»-flat confound ???´ì†Œ ë°©ì‹ ê²©ìƒ (user ê²°ì • 2026-07-14e)**: Î» ?¬ë¼?´ìŠ¤ë§??¬ëŠ” ?€??**FEATHer OFAT 512???„ë©´ ?¬ì‹¤??* ??ëª¨ë“  FEATHer HPê°€ êµì •??loss + ?œì? Weather split ?„ë˜?ì„œ ?ìƒ‰?˜ì–´ provenance caveat ?ì²´ê°€ ?Œë©¸.
07-06 ?™ê²°ë¶„ê³¼??diffë¥?ê¸°ë¡??ê²?(?¬ê²Œ ?¤ë¥´ë©?ê·??ì²´ê°€ ë²„ê·¸ ?í–¥??ì¦ê±°).
canonical mfg ?„í‚¤?ì²˜(d8/k7/p12/B3)???¬ì§‘ê³„ë¡œ ?¬í™•??(sub-1K ?œì•½??ì§€ë°°ì ?´ë¼ ? ì? ?ˆìƒ). Memory: project_lambda_flat_confound.md.
All audit-fix sites are tagged in-code ??`grep -rn "AUDIT-FIX"` lists every
fixed location with its finding ID.

**Server state**: 07-18b **2-?œë²„ ?¤ìœ•-ë¶„í• ** ??êµ??œë²„(Linux, ê³µìœ , GPU1)=
**??LTSF**(`fcst_results.csv`), ???œë²„(Windows, ?„ìš© GPU1, env `lee`)=
**??FEATHer OFAT**(`hp_search.csv`) + **?£a ?œì¡° lr**(`lr_search.csv`) ????(CSV ?¬ë¼ ?™ì‹œ ?ˆì „; ????ë¸”ë¡ + ROADMAP.md ë³´ë“œ).
**??07-20 ???œë²„ GPU ?¬ê³ +?˜ë¦¬**: env `lee`??torchê°€ `2.8.0+cpu`(CPU ?„ìš©
ë¹Œë“œ)ë¡?ê¹”ë ¤ ?ˆì–´ `cuda.is_available()=False` ???¡Â·â‘£aê°€ **?„ë? CPUë¡?* ?Œì•„
GPU 0%. `pip install torch==2.5.1 --index-url .../cu121`ë¡?êµì²´(`2.5.1+cu121
True 1` ?•ì¸). ë¡œì»¬ `feather`?€ ?™ì¼ ë²„ì „(ë³‘í•© ?•í•©??. CPUë¡???OFAT/lr ?‰ì?
**? íƒ???¤ìœ•?´ë¼ ?ê¸° ë¶ˆìš”**(ìµœì¢… ?«ì??????GPU ?¬ì‹¤??. ë°œì‚¬ ?•ì¸ = ?Œì»¤
`Using device: cuda:0` + `nvidia-smi`??python ?„ë¡œ?¸ìŠ¤(util ??? ê±?ì´ˆì†Œ??ëª¨ë¸?´ë¼ ?•ìƒ). ?¸ë? = `memory/reference_new_server_env_lee.md`.
**07-20 Ctrl+C ?¨ì¹˜ `37e4758`**(personal push): `utils/dispatch_queue.py` ???¨ì¼ Ctrl+Cê°€ ?¤ìœ• ?¸ë¦¬ ?„ì²´ ì¢…ë£Œ(stop ?´ë²¤??+ in-flight ?ì‹ ?¸ë“¤ ì¶”ì  ??`subprocess.run`??Popen`, ?Œì»¤ê°€ ??????ì§‘ê³  ?Œë˜ ?ì‹ terminate, ê³ ì•„ ë°©ì?).
**???œë²„ pull ?„ìš”.** (ê¸°ì¡´??Ctrl+Cê°€ ì£½ì¸ ?ì‹??"?¤íŒ¨ ???¼ë¡œ ë³´ê³  ?¤ìŒ ?¡ì„
?„ì›Œ ë§??°ë§ˆ??Ctrl+C ?´ì•¼ ?ˆìŒ.)

**Local git**: main fully PUSHED to personal/ijaehun **through `f906d1a`**
(07-23). 07-20 ?´í›„ ?¸ì‹œë¶? ê·¸ë¦¬???„êµ¬(`fbe077d` ê·¸ë¦¬?œí™•?¥Â?f569e04` summary
pred_len ?„í„°Â·`ed4549e` ?„ì„±-only+`--sort_by`Â·`e7924ee` `--per_horizon`) + **numeric
ë²„ê·¸ 3?½ìŠ¤**(`fb09aeb` dispatch Popenê°€?œÂ?15eba56` mCE skipnaÂ·`f906d1a` RMSE=?šì§‘ê³„MSE).
**???œë²„ pull ?„ìš”**(?„ëŠ” ?¤ìœ• ë¬´ì˜?? ?¤ìŒ ë°œì‚¬/?œìƒ?±ë???. Push policy:
**ijaehun (personal) ONLY**, not WDSLab, unless asked. ë¯¸ì»¤ë°?ë¡œì»¬ = ?´ë˜??ë² ì´?¤ë¼??LSTM/GRU/TCN + run_classical.py + `baselines/__init__.py` M) + ?Œì»¤ ?„ë¡œë¸??Œë˜ê·?--optimizer/--weight_decay) + ê²€? ìˆ˜??E ????ì²´ì¸ ?ë‚œ ??ë³„ë„ ì»¤ë°‹(?ìœ¼ë©?list_models=15ë¡??¤ìœ• ?ë™ ?•ì¥).

**??NEXT**
1. ??~~Apply audit fixes~~ ??ENTIRE audit closed 2026-07-14d (P0/P1 in
   cfd90c5, P2-3..8+P3 in the close-out commit). Nothing left in the audit
   doc.
2. **Server (07-18 ?œì°¨ ?¨ì¼-GPU ëª¨ë“œ ??ROADMAP.mdê°€ ?¨ì¼ ?ŒìŠ¤)**: 3-?™ì‹œ
   ??GPU 1 ?œì°¨ë¡??„í™˜. ì§€ê¸?**??LTSF ?¤í–‰ ì¤?* ???ë‚˜ë©?**??OFAT ???£a lr**
   ?˜ë‚˜?? ?¨ì? ì²´ì¸: **?¡Â·â‘£a ?„ë£Œ** ??`--validate` ??`--summary` ?„ë‹¬ ??   (Claude) 8???¬ë™ê²?+ 07-06 diff + canonical ?¬í™•??+ Î» ?ì • ??ì»¤ë°‹ ??   pull ????FEATHer LTSF 160 + ?£b FEATHer lr ??lr `--summary` MERGE ??mfg
   ë³??¤ìœ• 1,440 (`--save_model`) ??robustness ??`robust_summary.py` +
   `gate_logging.py`.
3. **A+C Intro/Motivation draft** (`manuscript/drafts/`) ??needs air-gap /
   OT-isolation citations; scope to genuinely-local settings.
4. **Still-missing analysis code**: component ablation UNDER corruption
   (gate-off/band-off variants through the robustness sweep) + Pareto plot
   generator (MSE-vs-params frontier).
4b. **JMS ?´ë˜??ë² ì´?¤ë¼??(êµ¬í˜„?? ë¯¸ì»¤ë°?ë¯¸í™œ?±í™”)**: LSTM/GRU/TCN
   (`baselines/`, list_models 15) + XGBoost (`run_classical.py`). ì»¤ë°‹ ??   **??ì²´ì¸ ?ë‚œ ??* ?œë²„ ?œì„±??15ëª¨ë¸ ?¤ìœ• + `run_classical.py --group
   mfg`). ARIMA/LightGBMÂ·ë°°í¬?¤ì¸¡(#1êµ¬ë©)Â·SWIFTë¬¸ë‹¨?€ ?„ì†. ?ì„¸ =
   ROADMAP.md "JMS ?¬í¬ì§€?”ë‹" + `memory/project_jms_baseline_repositioning.md`.
5. ??~~Push to ijaehun~~ ??done 2026-07-14d (through 59c85da).
6. Raw-data cleanup under `data/`: DECIDED skip (raw = regen safety net; drop
   PMSM raw first if disk pressure).

### Decision history (compact ??full blocks in `manuscript/notes/CLAUDE_STATUS_ARCHIVE_2026-07-14.md`)

- **2026-07-23 (TSLib ?•ë? ?€ì¡????Œì´?„ë¼??clean + LMS EAAI ë°œê²¬ + xlsx ?•ë¦¬)**:
  (1) **TSLib(thuml/Time-Series-Library) shallow cloneë¡?eval/train/metric/data
  4ì¶??•ë? ?€ì¡?*(?œë¸Œ?ì´?„íŠ¸) ??**ë²„ê·¸ 0, ?•í™•???µì‹¬ ?„ë? ?¼ì¹˜**. ?¹íˆ **metric
  scale = ?•ê·œ??ê³µê°„(????--inverse ê¸°ë³¸ off) ???°ë¦¬ ?«ì ë¬¸í—Œë¹„êµ ê°€???•ì •**;
  best-val ? íƒÂ·split(ETT 12/4/4Â·Custom .7/.1/.2)Â·train-only ScalerÂ·ê³µìœ 5ëª¨ë¸ HP ?¼ì¹˜.
  ?€?„í—˜ ?´íƒˆ 3ê°?val-loss ?˜í”Œê°€ì¤‘í‰ê·?FEATHer val=L1/train drop_last=True=?›TSLib)
  ?„ë? ?˜ë„?Â·ë°©?´ê??¥Â·ì¡°ì¹˜ë¶ˆ?? DLinear/PatchTST ?ë³¸ ?ì„±?ë§Œ ë¡œì»¬ ë¯¸í™•?¸ì´??  **?œë²„ ?¤í–‰?±ê³µ+ë¬¸í—Œë²”ìœ„ ?«ìë¡?ê²½í—˜???•ì¸**. clone=`C:/Users/lee/tslib_ref`(repoë°?5.4M).
  (2) **LMS-AutoTSF = EAAI(Elsevier Q1) 2026 ê²Œì¬ ?•ì¸**("?¦confidence-aware??,
  arXiv 2412.06866?€ ?„ë¦¬?„ë¦°?? ??R4 ë°©ì–´ peer-reviewed 3?? **?¬í˜„=ê³µì‹ ê¸°ë³¸ base
  `LMSAutoTSF`(V2 confidence-aware???´ë–¤ forecasting ?¤í¬ë¦½íŠ¸??ë¯¸ë°°?? ???•ë‹¹, ?¬ì‹¤??  ë¶ˆìš”**. ?ˆì??¤íŠ¸ë¦?venue ê°±ì‹ . [[project-baseline-lineup]]. (3) LTSF ?¬í˜„ê²°ê³¼ë¥?  **`results/ltsf_baseline_results.xlsx`**(ë³‘í•©?¤ë”+bestê°•ì¡°+Coverage 2?œíŠ¸)ë¡??•ë¦¬,
  throwaway probe CSV ?? œ(??_reference/`??TSLib clone ë³´ê?). (4) **?™ìŠµ?Œì»¤+?°ì´??  ë¡œë”©+utils ???Œì¼ ë²„ê·¸?ŒíŠ¸(?œë¸Œ?ì´?„íŠ¸ 2??**: ?«ì ì¡°ìš©??ë§ì¹  ë²„ê·¸ **0**
  (ë§ˆìŠ¤?¹Â·metric?„ì Â·windowingÂ·scalingÂ·CMAPSSÂ·seedÂ·lossesë°´ë“œ?œì„œÂ·noise?ìƒ?˜í•™Â·
  severity?œë“œ ??clean). **?¤ë²„ê·?1ê°??˜ì •Â·?¸ì‹œ `fb09aeb`**: `dispatch_queue.gpu_worker`ê°€
  `Popen` ?ˆì™¸ ??job??ì¡°ìš©???œë¡­(failed ë¯¸ê¸°ë¡??’launch+wait try/except ê°€?œë¡œ
  failed ê¸°ë¡+ê³„ì†. ???œë²„ pull ê¶Œì¥(?„ëŠ” ?¤ìœ• ë¬´ì˜?? ?¤ìŒ ë°œì‚¬ë¶€??. ?”ì—¬=?¬ëŒ?•ì¸??  minor(CORR/R2 ë§ˆìŠ¤??ë°°ì¹˜ê°€ì¤‘Â·Scalerê°€ outage hold-last??ë¬¼ë“¦Â·Electricity timeF ê³„ì‚°??¹„).
  **(ì¶”ê? ?¼ìš´??** config/ëª¨ë¸ ?´ëŸ¬?¤í„°(`baselines/__init__.py` override?¼ìš°??82?¤Â?
  wrapperÂ·FEATHer SPK/DTK/denorm)=**clean ë²„ê·¸0**; ?¤ìš´?¤íŠ¸ë¦¼ì„œ **2ê°????˜ì •**:
  `15eba56` robust_summary **mCE skipna ë²„ê·¸**(faultì¶??µì§¸ ?„ë½ ëª¨ë¸??3ì¶•í‰ê· ìœ¼ë¡???‚¹
  ì¡°ì‘?’skipna=False+ê²½ê³ , robustness ?¤ë“œ?¼ì¸?? + `f906d1a` main_table **RMSE=??ì§‘ê³„MSE)**
  (mean?šMSE?’âˆšmeanMSE, Jensen ê³¼ì†Œë³´ê³  ?´ì†ŒÂ·TSLib per-run ?¼ì¹˜Â·RMSE std??point?´ë¼ ?ëµ,
  MSE/MAE?ë§Œ Â±std). **?´ë²ˆ ê²€ì¦ì„œ numeric ë²„ê·¸ 3ê°??¡ìŒ(dispatchÂ·mCEÂ·RMSE), robustness
  ?¤ìœ• ?„ì§?´ë¼ ?¤í—˜ ë¬´ì˜?? paper ??ë½‘ê¸° ??? ì œ ?˜ì •.** lr-searchê°€ period-tied ëª¨ë¸??  default periodë¡???ë¬¸ì„œ?”ëœ caveat(?ê³  ??ë¬¸ì¥).
- **2026-07-21 (?°ì´?°ì…‹ 6ì¢??¬í™•??+ ë² ì´?¤ë¼???œì™¸?¬ìœ  ?µì¼ + CycleNet ?ì •
  ???„ë? ?ìŠ¤??ê²°ì •, ì½”ë“œÂ·?°ì´?°Â·ì„œë²?ë³€ê²?0)**: (1) **mfg-6 ? ì?, MetroPT3
  KEEP** ???œë¡­ ?„ë³´ë¡?4???™ê³ ?ˆìœ¼???€ì²??„ë³´ ?„ë? ??PMSM=2Hz persistence-
  trivial / TEP=?œë? / flotation=ê³µì •ì¤‘ë³µÂ·?´ë? 07-06 ?¨ìŠ¤ / ETT=generalization+
  robustness ?¼ì¤‘?¬ìš© / Korean10=ê°ì‚¬ë¦¬ìŠ¤???„ë ¥?´ëŸ¬?¤í„°ì¤‘ë³µ). MetroPT3=ì½”ë“œ
  no-op(?œë¡­?????†ìŒ)+?Œì „ê¸°ê³„ ?ì‚°?´ë˜?¤ê? ì±„ìš°??ì¶? **? ì •ê¸°ì? ?„ë ˆ?´ë°=
  "ë¹„íŠ¸ë¦¬ë¹„??persistence-ê°ì‚¬)+?¤ì–‘???¤ì¸¡, ê²°ì¸¡=?„ë¡œ? ì½œ-handled"**(??ë¬´ê²°??  thresholdÂ·?„ë©”?¸ìˆœ??ê¸ˆì? ??MetroPT3/WindSCADAê°€ ?„ì›ƒ?¼ì´?´ë¼ ?ê¸°ëª¨ìˆœ).
  [[project-mfg6-metropt3-decision-0721]] + ?°ì´?°ì…‹ **ê¸°ê° ë¡œê·¸**
  [[reference-dataset-reject-log]](?°ì´?°ì…‹ ?¼ì˜ ?¬ê°œ ??ë¨¼ì? ?˜ì§ˆ ê²?. (2)
  **ë² ì´?¤ë¼???œì™¸?¬ìœ  self-consistency ?½ìŠ¤**(CLAUDE.md "Excluded from
  benchmark" ë°˜ì˜): S-Mamba "no MCU port"??*CUDAì»¤ë„ ?¬í˜„ë¶ˆê?**, TimeCMA
  "5 orders"??*?¬ì „?™ìŠµ=?¤ë¥¸ ?ì›?´ë˜??*, SWIFT "18k"??*official code ë¯¸ì„±??
  FITS tie**(DLinear 18kê°€ ?œì— ?ˆì–´ sub-1K ?¼ë¦¬ ì¦‰ì‚¬), ë°©ì–´ë¬¸ë‹¨ ?¬ì‘?? seq_len=
  96 "LTSF convention"??*ë°°í¬?„ë¡œ? ì½œ+MCU ?…ë ¥ë²„í¼ ë°”ì´?¸ì˜ˆ??*(lookbackë§??µì¼
  ëª…ì‹œ). **size???ˆë? ?œì™¸ê¸°ì? ?„ë‹˜ ??41p??.6M ?¤íŒ¬??ëª©ì **. (3) **CycleNet/
  RLinear=cite-not-baseline**(ì¤‘ê°„-tierÂ·ë¯¸ìš”ì²?·ê²Œ?´íŒ…; "period ëª¨ë¸?´ë¼"???ê¸°
  ??SparseTSF??period; ?ì¹™???¬í•¨ê°€?¥ì´???”êµ¬/1ì°¨ìŠ¤?•ê°•?˜ë©´ ì¦‰ì‹œ escalation).
  (4) **robustness ë°©ì–´?¤í—˜ ?¬ì„¤ê³?*: ablation-under-corruption=**1ì°????©ëŸ‰
  confound-closer**(gate/band-off ??53p) / 2Ã—2 ?©ëŸ‰?¤ì™‘=ë³´ê°•(SparseTSF period-
  ?¸ë¸Œ?‘Â·FEATHer num_bands?? clean?•í™•??? ì? ë³´ê³ ë¡?detuneë°˜ë¡  ë´‰í•©) / ?´ì¤‘?„ë¡œ? ì½œ
  (ë©”ì¸ 96 + ë¶€ë¡?literature lookback, **SparseTSF@720 ?„ìˆ˜**) / int8=limitations
  ??ë¬¸ì¥(ë¦¬ë¹„?? / **?„ë? 1ì°??¤ìœ• ?¨ê³¼?•ì¸??ê²Œì´??*. [[project-baseline-lineup]]
  [[project-robustness-justification]] 07-21 ë¸”ë¡. **?¼ì¸??12Â·? â‘¡???¤ìœ• ë¬´ì˜??*;
  ì²´ì¸ ???´ë˜??4ì¢?LSTM/GRU/TCN+XGBoost)ë§??¤ì œ ì¶”ê?. CLAUDE.md ?ìŠ¤?¸ë¶„ ë¯¸ì»¤ë°?  (?¤ìŒ ?ê³ /ë¬¸ì„œ ì»¤ë°‹???™ë´‰).
- **2026-07-20c (??LTSF ?¤ìœ• sanity-audit ???„ë? clean)**: ?Œë˜ LTSF CSV ?‘ìŒ.
  TPAMI Table 3(ETTh1/ETTh2) ?¬í˜„ MSEÂ·MAE ?€ë¶€ë¶?Â±0.01(FITS/TQNet ?™ì¼ê°?
  PatchTST ?„ì¬ê°€ ??ì¢‹ìŒ=ê°œì„ , DLinear/ETTh2=?œë“œë¶ˆì•ˆ???¹ì„±), Weather ? íš¨
  (07-14+ ?¬ì‹¤??, RMSE 07-15+ë§??•ìƒ(????sqrt(MSE) ? ë„), HP?Œì‹± ?œì?
  (7 per-dataset ê³µì‹ / 4 ?¨ì¼ default = ?´ì©” ???†ê³  ?•ë‹µ), split ?œì?
  (ETT 12/4/4, ?˜ë¨¸ì§€ 0.7/0.1/0.2), ê³µì •??OK. **?¤í—˜ ?¤ê³„ ë¬¸ì œ?†ìŒ, ?¨ì? ê±?  ????ê²ƒë¿.** ?¸ë? = `memory/project_ltsf_baseline_verification_0720.md`.
  **decimation-vs-native ?¤í—˜ = ????*(?˜ì¤‘ì£¼ì¥ ?„ë‹˜+slow-process????’
  ë¦¬ìŠ¤????ë¦¬ë¹„??ì¹´ë“œ; ?¸íŠ¸ë¡?anti-decimation?€ ?¸ìš©+?¤ë‹¤?´ìœ¼ë¡?ë´‰í•©).
- **2026-07-20b (???œë²„ GPU ?˜ë¦¬ + Ctrl+C ?¨ì¹˜)**: ???œë²„ env `lee` torchê°€
  `2.8.0+cpu`???¡Â·â‘£aê°€ CPUë¡œë§Œ ?Œë˜ ê²?ë°œê²¬ ??`torch==2.5.1+cu121` ?¬ì„¤ì¹˜ë¡œ
  ?˜ë¦¬(`2.5.1+cu121 True 1`). CPU ?‰ì? ? íƒ?©ì´??? ì?. `dispatch_queue.py`
  ?¨ì¼-Ctrl+C ì¢…ë£Œ ?¨ì¹˜(`37e4758`, personal push, ???œë²„ pull). ???œë²„ ?¬ì‹¤??  = ?°ë??A `run_hp_search.py --jobs_per_gpu 3` / ?°ë??B `run_lr_search.py
  --exclude FEATHer --jobs_per_gpu 4`(RAM ?¬ìœ ë¡?3/4 ?Œì»¤; ??Traffic ê²¹ì¹  ??  RAM ì£¼ì‹œ). ?¸ë? = `memory/reference_new_server_env_lee.md`.
- **2026-07-20 (ì½”ë“œ/?„ë¡œ? ì½œ ê²€???˜ì • A~E ?„ë£Œ)**: ?œë¸Œ?ì´?„íŠ¸ ê²€??A/B
  (`manuscript/notes/review_findings_2026-07-19.md`) ë°œê²¬ 10ê±??„ë? ?˜ì • ??  **A `1e078d7`**(#1 test/robustness ?œë¡œì¹´ìš´?¸â†’?¤íŒ¨Â·#2 pred-NaN ê°€?œÂ?9
  robustness resume ?Œì†ê°€?œÂ?10 override ?„ë¡œ? ì½œì¶?ë¸”ë¡ë¦¬ìŠ¤?? + **B
  `c9ddd94`**(#3 ?„ì—­ ê´€ì¸??€ MSE/MAE ì¶”ì •???µì¼: ë°°ì¹˜-bsê°€ì¤‘â†’nansum/finite-
  count, persistenceÂ·classicalê³??¼ì¹˜, ë¬´ë§ˆ?¤í¬ byte-identical) + **CÂ·D
  `bbe6067`**(#4 main_table RMSE=?šMSE ?‰ë³„ ? ë„Â·#5 `--seeds` ?„í„°+?œë“œ??ë¶ˆì¼ì¹?  ê²½ê³ Â·#6 robust_summary dw/Î¼_rel/AUC ê³µí†µ-?€ ê°€?? ?œë²„ ?¸ì‹œ. **E??ë¡œì»¬
  ë¯¸ì»¤ë°?*(#7 run_classical ?¤íŒ¨??exit1Â·#8 XGBoost early-stop??ì§„ì§œ val
  split?¼ë¡œ ???´ë˜??ë¬¶ìŒê³??¨ê»˜ ?€ê¸?. ê²€ì¦? Steel 0.4884/N2O 0.7333
  byte-identical ??**ê¸°ì¡´ ???¬ì‹¤??ë¶ˆí•„?? ? Â·â‘¡ ë¬´ì˜??*. ?¨ì? ê²??¼ë¬¸ ?œìˆ 
  ë´‰í•© 3ê±?lr-config ëª…ì‹œ/FFT-period ?¤í¬ë¦½íŠ¸/wilcoxon ?ë???.
- **2026-07-19 (MASK-NAN-GRAD, `d56aa05` ?¸ì‹œ??**: MASK-FIX ?„ì† ì¹˜ëª… ë²„ê·¸
  ë°œê²¬Â·?˜ì • ??`_masked_loss`ê°€ full-tensorë¡?`(output-target)` ë¨¼ì? ê³„ì‚° ??  NaN ?€ê¹??€??autograd ê·¸ë˜??ì§„ì… ??backward `0*NaN=NaN` gradient ??  **ë§ˆìŠ¤??3ì¢?N2O/WindSCADA/MetroPT3)?ì„œ ì²??¤í… ê°€ì¤‘ì¹˜ ?„ì²´ NaN ë°œì‚°** ??  train/val/test ì§‘ê³„ `sum/max(1,cnt)=0`??**ê°€ì§?0.0 ê¸°ë¡**(P2-5 ê°€?œë„
  0.0=? í•œ?´ë¼ ?µê³¼). ?˜ì • (1) ?¸ë±??? í–‰ `output[mask]-target[mask]`,
  (2) val_cnt==0?’NaN(P2-5ê°€ ?œë„?½ê²Œ ?¤íŒ¨). 3ì¤??¤ëª¨??ê²€ì¦?N2O 0.0??.7333 /
  MetroPT3 0.6914 / Steel 0.4884 ë¬´íšŒê·€). **???œë²„ pull ???£a ?¬ê°œ ?„ìˆ˜**.
  ?„ì†¡??lr_search.csv??ë§ˆìŠ¤??ì¢?GasTurbine 286??purge ?„ë£Œ(1130??44).
  ê°™ì? ?? ??ì¤‘ê°„?ê?(886???´ë¦°, ê°?ë¬¸í—Œë²”ìœ„, DLinear/ETTh2 ?œë“œë¶ˆì•ˆ???¹ì„±)
  + ??TPAMI FEATHer ?˜ì¹˜ vs ??ë² ì´?¤ë¼??ë¹„êµ(ETT 1?„ê¶Œ ? ì?, DiPE ?°ì„¸,
  Electricityë§??½ì ) ??`manuscript/notes/ltsf_midsweep_check_2026-07-19.md`.
- **2026-07-16**: (1) **ë² ì´?¤ë¼??5?? ?œë“œ**(êµìˆ˜ ê²°ì •: ?œì¶œ ?°ì„  + ë¦¬ë¹„??  ë³´ê°•) ??`--num_seeds 3`, ì½”ë“œ ?˜ì • ?†ìŒ, ??LTSF 3?œë“œ ?¬ë°œ?? ?¤ëŠ” ?µì§¸ë¡?3?œë“œ,
  lr?€ 2?œë“œ ? ì?; ë¦¬ë¹„??top-up = `--num_seeds 5`(resume??2028/2029ë§?ì±„ì?).
  (2) **?ê³  ?ìŠ¤???¬ì‘??* ??`*_jms.md` 5ê°?abstract/intro/related/datasets/
  methodology) mfg-6 + A+C ê³ ì†-?¤ë¦¬(?™ê¸°ë§? + robustness ë³´ìˆ˜ + SWIFT + half-RevIN
  + ?´ë¡  ?¸ë¦¼. (3) **ê³ ì† ?„ë ˆ?´ë°**: HR-SHM?€ ?¤ì¬ ë¬¸í—Œ?´ë‚˜ ?°ë¦¬ ?°ì´??slow-
  process ???™ê¸° ?¸ìš©ë§? EWELD=ê¸ˆì†?°ì—… ?„ë ¥ë¶€??+ê·¹í•œê¸°ìƒ ì°?. Memory:
  project_manuscript_jms_drafts.md.
- **2026-07-15 (MASK-FIX)**: ê²°ì¸¡ ì²˜ë¦¬ ?•ë¡ ????ëª¨ë“  ê°?**hold-last**(ë³´ê°„
  ?œê±°, ?¸ê³¼?? + companion `mask.csv`ë¡?imputed ?€ ?œì‹œ ??ë¡œë”ê°€ ?•ë‹µë§?  NaN ??**loss/metric ê´€ì¸?only ì±„ì **(traffic/?˜ë£Œ/?¼ì„œ ?œì?; 12ëª¨ë¸ ?™ì¼??  ê³µì •). ì½”ë“œ 5?Œì¼(prep/data_loader/data_factory/run_forecast/metrics;
  robustness??ê³µìœ  nan-aware metric ?ë™ ?ì†). ê²€ì¦? ë°°ì„  ?¤ëª¨??ALL PASS
  + ?™ìŠµ Steel 0.5648/MetroPT3(17.7%ë§ˆìŠ¤?? 0.7533. ?¤ì¸¡ ê²°ì¸¡ë¥? MetroPT3
  17.67%/N2O 5.79%/WindSCADA 1.20%/BlastFurnace 1ì¹?EWELD 0. **?¬ì‹¤??3ê°?*
  (MetroPT3/N2O/WindSCADA ??dataê°€ interp?’hold). BF 1ì¹¸Â·EWELD 0 = ë¶ˆí•„??
  ? í˜•ë³´ê°„ lookahead ê³µê²©ë©??Œë©¸. ?¸ë?: ROADMAP.md "MASK-FIX" ë¸”ë¡.
- **2026-07-15 (JMS ?¬í¬ì§€?”ë‹ + ?´ë˜??ë² ì´?¤ë¼??êµ¬í˜„)**: JMS ë¦¬ë·°???œì¡°
  ?°êµ¬?????¤ë¬´ ? ë¢° ?µì»¤ë¡?**?´ë˜??ë² ì´?¤ë¼??* ì¶”ê? + **"?¤ë¬´ì¸??•í™•Â·ë¹„ë°°??
  vs ?£ì?ì¸?sub-1KÂ·ë°°í¬)" 2ì¸??„ë ˆ?´ë°** + **ë°°í¬ ?¤í—˜=#1 êµ¬ë©**(QEMU Layer-2
  skeleton ???¤ì¸¡) + ETT/Weather ê°•ë“±(?´ë? ê²°ì •). **SWIFT(2501.16178)=?¸ìš©ë§?*
  (18.1k=sub-1K ?„ë‹˜Â·FITS ?€?´Â·robustness 0Â·ICLR2025 withdrawn). **êµ¬í˜„ ?„ë£Œ
  (ë¯¸ì»¤ë°?ë¡œì»¬)**: LSTM/GRU/TCN(nn.Module, `baselines/{LSTM,GRU,TCN}/`,
  list_models 15, ê¸°ì¡´ workerë¡??™ìŠµ, arch ê³ ì •+lr?œì¹˜) + XGBoost(ë³„ë„
  `run_classical.py`, ì±„ë„?…ë¦½ horizon-feature ì§ì ‘?¤ìŠ¤?? ë§ˆìŠ¤??aware,
  GPU, ê³ ì •config+early-stop, lr?ìƒ‰X). ?¤ëª¨???„ë? ? í•œ. **???œë²„ ?œì„±??  ë³´ë¥˜**(push?˜ë©´ ?¤ìœ• 3ëª¨ë¸?? ??ì²´ì¸ ?ë‚œ ???˜ë„?ìœ¼ë¡?. ARIMA/LightGBM
  ë¯¸ê²°. Memory: project_jms_baseline_repositioning.md.
- **2026-07-14d/e**: ê°ì‚¬ P2-3..8+P3 ?„ë? ?˜ì •(23-check ê²€ì¦? `59c85da`) ??  ê°ì‚¬ ì¢…ê²°; prep ê°??•ì±…?¼ë¡œ ?°ì´??4ì¢??¬ìƒ??MetroPT3 16.6% ?? +
  BlastFurnace winsorize train-?ì—­; **07-14e ?¬ìš©??ê²°ì •: Î»-?¬ë¼?´ìŠ¤
  ?¬ì¸¡???€??FEATHer OFAT 512 ?„ë©´ ?¬ì‹¤??*(07-06 ?™ê²° 8???€ì²??ˆì •,
  diff ê¸°ë¡); LTSF/lr?€ `--exclude FEATHer`ë¡??¬ë°œ?? FEATHer ëª??¢â‘£b)?€
  ?¬ë™ê²??? TSLib ?€ì¡°ë¡œ eval ?„ë¡œ? ì½œ ?•ì„ ?•ì¸; push through 30f037d.
- **2026-07-14b**: `robust_summary.py` +4 metrics (Î¼_rel / AUC / mCE-vs-
  SparseTSF / degradation curve) + `gate_logging.py`, both synthetic-validated;
  R3 resolved (A+C, no short-term in paper); ROBUSTNESS_DATASETS
  GasTurbine?’EWELD = [Steel, WindSCADA, EWELD, ETTh1, Weather].
- **2026-07-14**: mfg lineup LOCKED at 6 (EWELD in; GasTurbine fully removed ??  broken, no revival); freq-tied period HPs data-driven in
  `_DATASET_OVERRIDES`; SCINet added-then-reverted; TEP/ziya07/AI4I/GasSensor
  vetted+rejected; PyTSF-MfG = cite for mfg-scarcity gap.
- **2026-07-13c**: short-horizon PdM DROPPED from paper (kept as shelf);
  mfg 3?? (+MetroPT3 +N2O +BlastFurnace-stress); Tier-2 loader prune
  (data_dict = 18 keys); push policy ??ijaehun only.
- **2026-07-13/b**: strategy pivot ??robustness (not accuracy) justifies
  params, protocol+citations in
  `manuscript/notes/robustness_justification_evidence.md`; server bugs fixed
  (reformer_pytorch import-time dep; --ngpu `torch.cuda.set_device`) ??  4 models were silently missing from CSVs, resume fills;
  `check_duplicates.py` added (fcst 769 rows, 0 dups).
- **2026-07-06/b**: OFAT 512 + combo 32 done ??FEATHer LTSF 8 rows FROZEN into
  `_DATASET_OVERRIDES` (raw-verified); dataset precedent audit ??all 6
  confirmed, "first to cast as edge LTSF" holds; wording guards in
  `manuscript/notes/dataset_precedent_audit.md`.
- **2026-07-03**: OFAT verdict ??canonical mfg arch stays d8/k7/p12/B3
  (d16/p6 win val but break sub-1K); `--ngpu` dynamic queue + `--jobs_per_gpu`
  added; ??FEATHer params scale with pred_len ??sub-1K claim is H=96-scoped.
- **2026-07-02/b**: PMSM dropped from main (2Hz trivial persistence), rebuilt
  for PdM; PHM-2018 vetted+rejected; protocol sequencing locked (arch from
  LTSF OFAT, lr-only on mfg, symmetric).
- **2026-06-25/26**: JMS manufacturing pivot (professor's call, from IoT-J);
  TEP dropped, C-MAPSS added as separate short-horizon section (later
  shelved); Word submission format.
- **2026-06-11..18**: eval protocol rewritten (best-val-epoch, test-once);
  per-dataset FEATHer tuning replaces single-config narrative; baseline LTSF
  sweep launched; tex symbol/citation fixes.

## Environment

- **Conda env**: `feather` at `C:\Users\lee\.conda\envs\feather\` ??Activate
  with `conda activate feather` before any commands. Has `torch 2.5.1+cu121`,
  `darts 0.37.1`, `reformer_pytorch 1.4.4` (for iTransformer's
  SelfAttention_Family).
- `mamba-ssm` is intentionally **not** installed (CUDA-toolchain build,
  out-of-scope for our edge-deployable benchmark).

## Repository structure

Layout rule (2026-08-24): **`utils/` = data + training machinery**, **`tools/` =
things a human runs to produce or inspect outputs** (paper artifacts, audits,
HP searches). Nothing under `tools/` is imported by the training path.

```
FEATHer/
?œâ??€ run_forecast.py                 THE orchestrator ??main sweeps + --ablation_arm (--check, resume, dispatch)
?œâ??€ setup_baselines.sh              git-clone baseline upstream repos (idempotent) + pip deps
?œâ??€ scripts/benchmarks/run_forecast.py   worker (one model Ã— dataset Ã— pred_len Ã— N seeds)
?œâ??€ baselines/                      top-level model directory
??  ?œâ??€ _import_helper.py           sys.modules isolation for upstream imports
??  ?œâ??€ __init__.py                 registry (14) + _METHOD_DEFAULTS + _DATASET_OVERRIDES
??  ?œâ??€ FEATHer/FEATHer.py          our model (2 variants via channel_mode; see _FEATHER_MFG_N / _FEATHER_LTSF_CI)
??  ?”â??€ <13 baseline dirs>/wrapper.py + <Name>-main/ upstream clones (gitignored)
?œâ??€ utils/
??  ?œâ??€ prep_manufacturing.py       raw ??data/<name>/data.csv + mask.csv (9 recipes; data/ gitignored)
??  ?œâ??€ data_factory.py             data_provider + data_select (lazy darts import; data_dict = 18 keys)
??  ?œâ??€ data_loader.py              Dataset classes (ETT_hour/minute, Custom, CMAPSS)
??  ?œâ??€ dispatch_queue.py           --ngpu dynamic job queue (one worker subprocess per GPU)
??  ?œâ??€ losses.py                   spectral_separation_loss_scales (FEATHer-specific)
??  ?œâ??€ metrics.py                  metric() (MSE/MAE/RMSE/CORR/R2)
??  ?œâ??€ seed.py                     set_seed() + parse_seed_list()
??  ?”â??€ timefeatures.py
?œâ??€ tools/
??  ?œâ??€ audit/                      check_progress Â· check_duplicates Â· purge_rows Â· results_table Â· migrate_checkpoints
??  ?œâ??€ search/                     search_feather_{N_mfg,CIN_ltsf,L_ltsf}.py ??how the frozen configs were selected
??  ?”â??€ paper/                      table / figure generators
??      ?œâ??€ merge_raw.py            per-sweep CSVs ??mfg_raw.csv / ltsf_raw.csv (applies the LTSF anchor rule)
??      ?œâ??€ build_paper_tables.py   ??analysis/01,02,03_*.xlsx
??      ?œâ??€ build_complexity_xlsx.py + measure_complexity.py   FFT-aware MACs/params ??analysis/04
??      ?œâ??€ ablation_summary.py     --ablation_arm deltas vs the main-table row (??% = signal)
??      ?œâ??€ compute_r2.py           post-hoc R2 = 1 ??MSE/Var(y_test) (stored R2 column is unusable)
??      ?œâ??€ main_table.py           meanÂ±std table + persistence column (md / latex / csv)
??      ?œâ??€ wilcoxon.py             pairwise signed-rank vs FEATHer
??      ?œâ??€ pareto_plot.py Â· energy_envelope.py               accuracy-per-resource figures
??      ?”â??€ persistence_baseline.py copy-last baseline under the exact worker protocol
?œâ??€ deployment/                     board/ (C port + golden) Â· cortex_m3/ (estimator) Â· EDGE_EXPERIMENT.md
?œâ??€ results/                        mfg_raw.csv Â· ltsf_raw.csv Â· complexity.csv Â· _archive/ (provenance) Â· checkpoints/
?œâ??€ analysis/                       01??4 xlsx tables Â· RESULTS_STATUS.md Â· lit_review/ (gitignored)
?œâ??€ _archive/                       retired-but-retrievable: robustness/ Â· ablation_v1/ Â· grids/ Â· classic_baselines/
?œâ??€ data/                           raw + processed data (gitignored ??transfer to server manually)
?”â??€ manuscript/                     reviews, drafts, notes (gitignored; tex_workspace/ + drafts/*.md selectively tracked)
```

**Retired 2026-08-24** (all under `_archive/`, each with a README): the
robustness track (orchestrator + worker + `utils/noise.py` + `utils/severity.py`
+ `robust_summary.py` + `gate_logging.py` ??FEATHer degraded worst of the edge
tier, claim dropped); the 30 `FEATHer_<axis>_<variant>` ablation builders and
`ablation_table.py` (predate the residual MLP and channel mode, so they built
the pre-08 architecture); `run_grid_ltsf_v2.py` (superseded by `tools/search/`).
The OFAT/lr search orchestrators went earlier.

## Common commands

```bash
# Always activate the env first
conda activate feather

# === One-time: clone the 11 baseline upstream repos (+ reformer_pytorch dep) ===
bash setup_baselines.sh           # clone all
bash setup_baselines.sh DLinear   # clone one

# === Manufacturing main table (default group; 6 datasets) ===
python run_forecast.py --check --exp_tag main             # 1,440 mfg runs (12Ã—6Ã—4Ã—5)
python run_forecast.py --exp_tag main --save_model        # group=mfg by default; --save_model makes robustness ckpts
python run_forecast.py --group ltsf --exp_tag main --save_model   # LTSF-8 generalization (1,920)
python run_forecast.py --group cmapss --exp_tag main --save_model # PdM shelf (540; NOT in paper)
python run_forecast.py --group all  --exp_tag main --save_model   # mfg + ltsf = 3,360
python run_forecast.py --model FEATHer --save_model                # one model across group
python run_forecast.py --data ETTh1 --pred_len 96 --save_model     # one (data, horizon)
python run_forecast.py --exclude TimesNet,MDMLP_EIA                # skip heavy models
python run_forecast.py --num_seeds 1 --num_epochs 2 --exp_tag smoke

# Multi-GPU: all four orchestrators take --ngpu N (dynamic queue, one worker
# subprocess per GPU; utils/dispatch_queue.py; no DDP) and --jobs_per_gpu M
# (ngpuÃ—M workers, auto OMP/MKL pinning). ngpu=1 default = sequential.
python run_forecast.py --exp_tag main --save_model --ngpu 2 --jobs_per_gpu 3

# === lr search for the mfg main table (Option B; runs BEFORE the main sweep) ===
# 12 models Ã— 5 lr Ã— 9 datasets (6 mfg + 3 PdM shelf) Ã— 2 horizons Ã— 2 seeds = 2,160.
# Bypasses _DATASET_OVERRIDES (default arch) ??period overrides only affect the main sweep.
python run_lr_search.py --check
python run_lr_search.py --ngpu 2            # resumable
python run_lr_search.py --summary           # best lr per (method,dataset) ??MERGE into _DATASET_OVERRIDES
                                            # (merge into EXISTING period entries ??no duplicate keys!)

# === HP search (DONE ??configs frozen in _DATASET_OVERRIDES) ===
# Kept as reproducibility evidence for how the shipping configs were picked.
python tools/search/search_feather_N_mfg.py --summary     # mfg  ??ds16_k7_B4
python tools/search/search_feather_CIN_ltsf.py --summary  # LTSF ??per-dataset
# Windows gotcha: run --summary with PYTHONUTF8=1 PYTHONIOENCODING=utf-8 (em-dash vs cp949)

# === Component ablation (mfg; one component off per arm, on the SHIPPING model) ===
python run_forecast.py --ablation_arm all --group mfg --num_seeds 1   # 6 arms Ã— 24 = 144
python run_forecast.py --ablation_arm gateuni,band2 --group mfg       # subset
python tools/paper/ablation_summary.py --per_dataset                  # verdict (??% = signal)
# arms: noresid / ci / norevin / gateuni / band2 / d4.  Rows are tagged
# abl_<arm> and routed to results/ablation.csv; the main sweep is the baseline
# and is never re-run.

# === Progress audit ===
python tools/audit/check_progress.py --exp_tag main
python tools/audit/check_duplicates.py

# === Paper artifact generators ===
python tools/paper/merge_raw.py                              # ??mfg_raw.csv / ltsf_raw.csv
python tools/paper/build_paper_tables.py                     # ??analysis/01,02,03_*.xlsx
python tools/paper/compute_r2.py --csv results/mfg_raw.csv   # post-hoc global R2
python tools/paper/main_table.py --metric MSE --format latex > main_mse.tex
python tools/paper/wilcoxon.py --reference FEATHer

# === Data regen (data/ is gitignored) ===
python utils/prep_manufacturing.py            # all 6 mfg recipes
python utils/prep_manufacturing.py EWELD      # one (BlastFurnace needs openpyxl)

# === Cortex-M3 edge cost estimator / QEMU (Layer 1 / Layer 2, WSL) ===
python -m deployment.cortex_m3.run --check
bash deployment/cortex_m3/qemu/run_qemu.sh FEATHer ETTh1 96 64

# === Manuscript build (IoT-J tex now legacy; JMS = Word via drafts/*.md) ===
cd manuscript/tex_workspace
pdflatex feather_iotj && bibtex feather_iotj && pdflatex feather_iotj && pdflatex feather_iotj
```

## Model registry (12 entries)

```python
from baselines import get_model, list_models, get_method_defaults, get_dataset_overrides
print(list_models())
# ['DLinear', 'DiPE_Linear', 'FEATHer', 'FITS', 'LMS_AutoTSF', 'MDMLP_EIA',
#  'PatchTST', 'SparseTSF', 'TQNet', 'TimeMixer', 'TimesNet', 'iTransformer']
```

| Model | Venue | Year | Params (ETTh1 paper config, D=7) |
|---|---|---|---|
| DiPE_Linear | DASFAA | 2026 | 267 |
| SparseTSF | ICML Oral | 2024 | 41 |
| FEATHer (ours) | (JMS) | 2026 | 453 |
| FITS | ICLR Spotlight | 2024 | 1,200 |
| DLinear | AAAI | 2023 | 18,624 |
| PatchTST | ICLR | 2023 | 35,171 |
| TimeMixer | ICLR | 2024 | 75,497 |
| TQNet | ICML | 2025 | 86,192 |
| LMS_AutoTSF | EAAI (Elsevier) | 2026 | 181,318 |
| TimesNet | ICLR | 2023 | 605,479 |
| iTransformer | ICLR | 2024 | 841,568 |
| MDMLP_EIA | AAAI | 2026 | 1,632,832 |

(SCINet was briefly added 2026-07-13 then REMOVED ??reviewer only named it as
prior multi-scale work, not a baseline request; Related-Work differentiation
instead.) ??5 orders of magnitude (267 ??1.6M). Three recent peer-reviewed
SOTA (DiPE-Linear DASFAA'26, MDMLP-EIA AAAI'26, LMS-AutoTSF EAAI'26 ??the
last confirmed 2026-07-23 as journal-published, not arXiv-only) address R4's
"outdated baselines" critique.

Each baseline lives at `baselines/<Name>/`. `wrapper.py` exposes
`build(args, n_features, seq_len, pred_len)` that:
1. Uses `_import_helper.isolated_import(repo_root, "models.Foo")` to load
   the upstream `Model` class with full `sys.modules` isolation.
2. Constructs an `argparse.Namespace` with paper-default hyperparameters.
3. Wraps non-standard forward signatures into uniform `forward(x) -> (B, pred_len, D)`.

### Excluded from benchmark (cite in Related Work only)

- **S-Mamba** ??selective-scan **custom CUDA kernels** ??not fairly
  reproducible under our unified from-scratch protocol (AMP off, identical
  training loop); `mamba-ssm` intentionally not installed. *(NOT "no MCU
  port" ??the ceiling models, e.g. iTransformer 841k / MDMLP 1.6M, don't fit
  an MCU either; MCU-portability can't be the exclusion criterion.)*
- **TimeCMA** (AAAI 2025) ??frozen **pretrained GPT-2 features = a different
  resource class** (not trainable from scratch). *(NOT a size cutoff ??we KEEP
  MDMLP 1.6M as a ceiling; size is never the exclusion criterion, the 41p??.6M
  span IS the point.)*
- **SWIFT** (arXiv 2501.16178) ??**no mature official code** ??not fairly
  reproducible under our protocol; performance **ties FITS** (already in the
  table ??FITS represents that point). ICLR2025-withdrawn = secondary note
  only. *(NOT "18k not sub-1K" ??DLinear 18,624p is a benchmarked baseline.)*

Reviewer-defense paragraph (paper draft):
> "Model size is not our exclusion criterion ??our benchmark deliberately
> spans 41 to 1.6M parameters to map the capacity?“accuracy frontier. We
> exclude only architectures we cannot fairly reproduce under a unified
> from-scratch protocol ??Mamba-family models (S-Mamba) requiring custom
> CUDA kernels, and methods without mature official code (SWIFT) ??or that
> belong to a different resource class, namely LLM-augmented / pretrained
> forecasters (e.g., TimeCMA) depending on frozen external features rather
> than being trained from scratch. These are cited in Related Work but lie
> outside our reproducibility-controlled benchmark."

## Training protocol ??TFB-style per-(method, dataset) paper HP

Resolved 2026-06-02 via TFB-style per-(method, dataset) overrides. See
`memory/project_open_lr_policy.md` for the full rationale.

### ??Unified across all (method, dataset) cells

| Axis | Value | Reason |
|---|---|---|
| `seq_len` | 96 | Deployment protocol: unified across all models; justified by MCU input-buffer byte budget (720Ã—DÃ—4B vs 96Ã—DÃ—4B). Other HP from each paper's recipe ??lookback is the one deliberate override (state this to avoid the "published recipe vs 96 override" self-contradiction; SparseTSF@720 goes in an appendix rank-stability check). |
| `batch_size` | 32 | Memory protocol axis |
| `num_epochs` | 50 | Epoch-budget protocol axis |
| `patience` | 10 | Early stop on val loss; selection = best-val epoch, test evaluated once on that model |
| Optimizer | Adam | Standard |
| Scheduler | CosineAnnealingLR | Standard |
| AMP | **OFF** | Fairness + FITS complex grads + TimesNet cuFFT half-precision |
| Seeds | 2025??027 (3 seeds; â¤? in revision) | 07-16 triage: `--num_seeds 3` submit-first, top-up to 5 (`--num_seeds 5`) in revision ??3??, resume fills the 2; base_seed/exp_tag fixed. lr search stays `LR_SEEDS=[2025,2026]`. |
| `deterministic` | `cudnn.deterministic=True` | Reproducibility |
| Data split | identical | All baselines see the same train/val/test |

### ? ï¸ Per-(method, dataset) overrides

| Axis | Source |
|---|---|
| `lr` | baselines: each paper's per-dataset script; mfg: `run_lr_search.py` (merge pending) |
| `loss` | each paper's default (MSE for baselines, L1 for FEATHer) |
| Architecture HPs | paper Table or official script; FEATHer LTSF-8 rows from OFAT (frozen) |
| Freq-tied (`cycle`, `period_len`, FEATHer `period`) | data-driven per mfg dataset (FFT-detected), same rule for all 3 period-models |

Routing: `_METHOD_DEFAULTS` + `_DATASET_OVERRIDES` in `baselines/__init__.py`;
orchestrator merges defaults + overrides + CLI, forwards `lr`/`loss` as worker
flags and the rest as `--model_overrides "k=v;k2=v2"`; worker's
`_apply_overrides()` sets each key on `args` before model construction.
FEATHer is tuned per-dataset like every baseline (symmetric fair-comparison
framing); canonical mfg arch = base d8/k7/p12/B3.

## Architecture (FEATHer model)

`baselines/FEATHer/FEATHer.py`, four components:

1. **Multi-band Decomposition** ??depthwise 1D convs separate input into
   frequency bands (POINT, HIGH?, MID, LOW). `num_bands ??{2,3,4}`.
2. **DenseTemporalKernel (DTK)** ??project to latent ??depthwise causal conv
   ??project back. Shared across all bands.
3. **FFTFrequencyGate** ??FFT magnitude ??Conv1d ??softmax weights, one per
   band, adaptive per sample. Casts to float32 (cuFFT half-precision).
4. **SparsePeriodKernel (SPK)** ??period-aware head: reshape by period phases,
   shared linear projection, reconstruct.

Loss = main (L1/MSE) + `lambda_spec` Ã— spectral-separation loss (FEATHer-only;
??band-order inversion bug ??audit P1-1 ??fix pending).

Key FEATHer args: `--d_state 8` (DTK latent), `--kernel_size 7`, `--period 12`
(must divide seq_len AND pred_len), `--num_bands 3`, `--lambda_spec 0.01`.

## Results CSV schema

Single source of truth: `results/fcst_results.csv`. Append-only.

Columns (resume key in bold):
**`exp_tag`**, **`model`**, **`data`**, **`pred_len`**, **`seq_len`**,
**`seed`**, `MSE`, `MAE`, `RMSE`, `CORR`, `R2`, `val_loss`, `best_epoch`,
`num_params`, `timestamp`

Test metrics computed once on the best-val-epoch model. `exp_tag` partitions
experiments (`main`, `robust`, `ablation_dtk`, `smoke`, `lr_*`, `hp_*`).
Use `--results_csv PATH` to redirect verification runs.

## Datasets

Loaded via `utils/data_factory.py` (lazy darts import; `data_dict` = 18 keys
after the 2026-07-13c Tier-2 prune).

| Dataset | Source | Horizon set |
|---|---|---|
| **Steel Â· WindSCADA Â· BlastFurnace Â· N2O Â· MetroPT3 Â· EWELD** (MFG **main table**; BlastFurnace D=28 = stress tier) | local CSV (`utils/prep_manufacturing.py`) | [96, 192, 336, 720] |
| CMAPSS / CMAPSS3 / PMSM (PdM **shelf**, not in paper; unit-aware `Dataset_CMAPSS`) | local CSV | [24, 48, 96] |
| ETTh1/ETTh2/ETTm1/ETTm2/Weather/Exchange/Electricity/Traffic (generalization) | darts | [96, 192, 336, 720] |
| nrel (kept per user) | local CSV | [96, 192, 336, 720] |

Robustness set: `run_robustness.py:ROBUSTNESS_DATASETS` =
**[Steel, WindSCADA, EWELD, ETTh1, Weather]** (EWELD's 5,741 extreme-weather
event windows = the only non-synthetic distribution-shift test).
EWELD gotcha: strong daily cycle ??MUST report a seasonal-naive baseline
(copy-last-day ~0.55). Each `data/<name>/` holds raw + cleaned `data.csv`;
regenerate with `python utils/prep_manufacturing.py [<name> ...]`
(BlastFurnace needs `openpyxl`). `data/` is gitignored ??transfer manually.

## Known issues / gotchas

- **??AUDIT (2026-07-14) ??FULLY CLOSED** (P0/P1/P2-1,2 in cfd90c5;
  P2-3..8 + P3 in the 07-14d close-out). Residual DATA facts to remember:
  pre-fix Weather rows and stored-RMSE values in old CSVs are stale (purge /
  derive sqrt(MSE) respectively); lr rows for WindSCADA/BlastFurnace/N2O/
  MetroPT3 predate the gap-policy data regen (purge+refill per SERVER
  PROCEDURE); `data/<name>/data.csv.preP3bak` = pre-fix data copies (delete
  once the server transition is confirmed).
- **`--ngpu N` + upstream hardcoded `.cuda()` (fixed 2026-07-13)**: workers
  call `torch.cuda.set_device(args.gpu)` at startup. Do NOT sed the upstream
  `baselines/*/*-main/` files (gitignored, re-cloned).
- **`reformer_pytorch` is an import-time dep of 4 baselines** (iTransformer,
  LMS_AutoTSF, TimeMixer, TimesNet). Missing ??those 4 die at import, write
  no CSV row, and look "not yet run" (failed workers currently exit 0 ??  audit P2-1). `setup_baselines.sh` installs `reformer_pytorch==1.4.4`.
- **AMP is forcibly disabled** in the worker. Re-enabling breaks FITS
  (complex grads) and TimesNet (half-precision cuFFT).
- **`_import_helper.isolated_import`** must be used for every upstream
  baseline import (sys.modules collisions otherwise).
- **TQNet** receives `cycle_index = torch.zeros(B)` (our DataLoader has no
  per-sample cycle position) ??consistent across all batches.
- **R2 metric** can explode to extreme negatives on near-constant channels
  (SST??); MSE/MAE/RMSE unaffected; R2/CORR supplementary only.
- **FEATHer params scale as DÂ²** (`in_proj`): sub-1K holds for D??4 only.
  Weather 1.4K / Electricity 115K / Traffic 776K / BlastFurnace(D=28) 1,986
  are framed as stress tests.
- **FEATHer params also scale with pred_len** (SPK backbone): quoted counts
  are H=96 values; at H=720 D??1 exceeds 1K. Sub-1K claim must be scoped to
  H=96 in the manuscript (flagged 2026-07-03, unresolved).

## Workflow phases

- **Phase 1??** ??infra (multi-seed, 12-model registry, 4-axis noise).
- **Phase 4** (running) main sweeps on server: mfg 1,440 + LTSF 1,920.
- **Phase 4b** (blocked on P0 fixes + mfg checkpoints) robustness sweep:
  12 models Ã— 5 datasets Ã— 4 horizons Ã— 5 seeds Ã— 20 conditions = 24,000 rows,
  inference-only from `--save_model` checkpoints.
- **Phase 5** paper artifacts under `tools/paper/` (+ still to write:
  ablation-under-corruption aggregation, Pareto plot).

## Paper context

- Target venue: **Journal of Manufacturing Systems (JMS)**, Elsevier/SME, Q1.
  Story: edge PdM/process-monitoring forecasting on machine-side MCUs;
  FEATHer = sub-1K Cortex-M3-deployable forecaster, robust to shop-floor
  sensor faults, validated across 6 industrial domains. Contributions:
  (1) sub-1K multiscale-frequency forecaster, (2) first edge/sub-1K mfg LTSF
  benchmark (6 domains), (3) sensor-fault robustness, (4) Cortex-M3
  deployment evidence.
- **Submission format: Word** (Elsevier "Your Paper Your Way"; no rigid
  template; Highlights 3?? bullets ??5 chars; Vancouver refs; CRediT +
  declarations). Draft in markdown under `manuscript/drafts/` ??paste into
  Word. Order: Intro (A+C framing) ??Datasets ??Related ??Method polish ??  results when sweeps land.
- Reuse map from `feather_iotj.tex`: Methodology + Theoretical Analysis
  ~as-is (trim theory); Intro/Related/Datasets need the manufacturing
  reframe; Results/Ablation/Robustness wait for sweep numbers.
- Wording guards: CMAPSS = "first to cast as a fixed-horizon multivariate
  sensor forecasting BENCHMARK" (not "first to forecast"); PMSM cite AIP
  Advances 16(4):045101; Steel all-channels departure stated explicitly;
  mfg-LTSF precedent = "first to cast as edge PdM LTSF"
  (`manuscript/notes/dataset_precedent_audit.md`).
- Reviewer-by-reviewer notes in memory; reject letter at
  `manuscript/notes/reject_mail.md`; robustness protocol citations at
  `manuscript/notes/robustness_justification_evidence.md`.
