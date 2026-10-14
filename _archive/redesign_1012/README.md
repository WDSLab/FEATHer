# 10-12 redesign of FEATHer-mix (finished 10-14, nothing adopted)

Candidates (revin_mode mean / learn_level, mix_stats, no RevIN) were chosen on validation only; the stage-1 winner
was "no RevIN" (`use_norm=False`, still an ablation arm).  Stage 2 (its lr x0.3 / x3) was paused and closed on 10-14.
Results stay in results/raw_mfg.csv (role "redesign").  The options themselves are in
_archive/FEATHer_full_20261014/FEATHer.py; `redesign.yaml` and `redesign_verdict.py` here are the definitions and
the verdict tool as they were.
