# -*- coding: utf-8 -*-
"""제조에서 FEATHer_mix 가 경량 모델을 이기는 이유를 가른다.

    python analysis/diagnostics/why_we_win.py

질문 (2026-09-10)
  우리 우위가 (가) 선형으로는 못 잡는 비선형 구조 때문인가,
  (나) 센서끼리의 관계 때문인가.

재료
  results/analysis/linear_ceiling_mix.csv   데이터만으로 구한 두 선형 천장
      ci   센서 따로 곱해서 더하기의 최선 (val 로 규제 선택, test 1회)
      mix  센서 섞어서 곱해서 더하기의 최선 (CI 쪽으로 당기는 규제)
  results/_archive/pre_raw_20261007/mfg_raw.csv              모든 모델의 test MSE (3시드 평균)

지표 (칸마다, 전부 우리 모델과 무관하게 정의되는 것과 우리 성적을 분리)
  lin_mix_gain   (ci - mix) / ci          센서를 섞어서 선형으로 얻는 이득
  nl_head        (best_lin - heavy) / best_lin
                 선형 최선(ci, mix 중 낮은 쪽) 너머로 무거운 모델이 간 거리
                 = 비선형 여유.  무거운 모델 7개만 쓰고 FEATHer 는 뺀다 (순환 방지)
  ours_mix_gain  (FEATHer_CI - FEATHer_mix) / FEATHer_CI   우리가 섞어서 얻은 이득
  ours_vs_light  (경량 최선 - FEATHer_mix) / 경량 최선      우리의 우위

해석 원칙
  - 결론 단위는 칸(24)이 아니라 데이터셋(6).  같은 데이터의 호라이즌 4칸은 서로
    독립이 아니다.  상관계수는 참고용이며 데이터셋별 표를 먼저 본다.
  - lin_mix_gain 은 test 에서 음수가 될 수 있다: val 로 고른 섞기가 test 기간에
    깨지는 경우(분포 변화).  그건 도구 오류가 아니라 "센서간 관계가 불안정하다"는
    측정이다.  mix_is_ci=True 는 val 이 "섞어도 이득 없음"을 고른 칸.
"""
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_ROOT)

LIGHT = ["SparseTSF", "MixLinear", "FITS", "DiPE_Linear", "DLinear", "CycleNet",
         "SparseTSF_MLP"]
HEAVY = ["TimesNet", "PatchTST", "LMS_AutoTSF", "iTransformer", "TimeMixer",
         "MDMLP_EIA", "TQNet"]


def main():
    cpath = "results/analysis/linear_ceiling_mix.csv"
    if not os.path.exists(cpath):
        sys.exit(f"{cpath} 없음 - linear_ceiling_mix.py 먼저")
    c = pd.read_csv(cpath, on_bad_lines="skip")
    acc = (pd.read_csv("results/_archive/pre_raw_20261007/mfg_raw.csv")
           .groupby(["model", "data", "pred_len"]).MSE.mean().unstack(0))

    rows = []
    for r in c.itertuples():
        key = (r.data, r.pred_len)
        if key not in acc.index:
            continue
        a = acc.loc[key]
        best_lin = min(r.ci, r.mix)
        heavy = a[HEAVY].min()
        light = a[LIGHT].min()
        rows.append(dict(
            data=r.data, H=r.pred_len, D=r.D,
            ceil_ci=r.ci, ceil_mix=r.mix, fe_mix=a.FEATHer_mix, fe_ci=a.FEATHer_CI,
            heavy=heavy, light=light, light_who=a[LIGHT].idxmin(),
            lin_mix_gain=(r.ci - r.mix) / r.ci * 100,
            nl_head=(best_lin - heavy) / best_lin * 100,
            ours_mix_gain=(a.FEATHer_CI - a.FEATHer_mix) / a.FEATHer_CI * 100,
            ours_vs_light=(light - a.FEATHer_mix) / light * 100,
            fe_beats_ci_ceil=a.FEATHer_mix < r.ci,
            fe_beats_mix_ceil=a.FEATHer_mix < r.mix,
            mix_is_ci=bool(getattr(r, "mix_is_ci", False)),
            lam_edge=bool(getattr(r, "lam_edge_ci", False) or getattr(r, "lam_edge_mix", False)),
        ))
    t = pd.DataFrame(rows)
    if t.empty:
        sys.exit("분석할 칸 없음")
    pd.set_option("display.width", 220)

    print(f"칸 {len(t)} / 24,  데이터셋 {t.data.nunique()} / 6")
    if t.lam_edge.any():
        print("⚠ 규제 격자 끝값 칸:", t[t.lam_edge][["data", "H"]].values.tolist())
    print()
    cols = ["data", "H", "D", "ceil_ci", "ceil_mix", "heavy", "fe_mix", "fe_ci",
            "lin_mix_gain", "nl_head", "ours_mix_gain", "ours_vs_light", "mix_is_ci"]
    print(t[cols].round(4).to_string(index=False))

    print("\n=== 데이터셋별 평균 (%) — 결론은 이 표로 ===")
    g = t.groupby("data")[["lin_mix_gain", "nl_head", "ours_mix_gain",
                            "ours_vs_light"]].mean().round(2)
    g["우리가 섞기천장 이김"] = t.groupby("data").fe_beats_mix_ceil.sum().astype(int)
    g["우리가 CI천장 이김"] = t.groupby("data").fe_beats_ci_ceil.sum().astype(int)
    print(g.to_string())

    if t.data.nunique() >= 3:
        print("\n=== 순위상관 (참고용, 데이터셋 6개 수준) ===")
        gd = t.groupby("data")[["lin_mix_gain", "nl_head", "ours_mix_gain",
                                 "ours_vs_light"]].mean()
        rho = lambda x, y: gd[x].rank().corr(gd[y].rank())   # noqa: E731
        print("  우리 섞기이득     ~ 선형 섞기이득   rho=%+.2f" % rho("ours_mix_gain", "lin_mix_gain"))
        print("  경량대비 우리우위 ~ 선형 섞기이득   rho=%+.2f" % rho("ours_vs_light", "lin_mix_gain"))
        print("  경량대비 우리우위 ~ 비선형 여유     rho=%+.2f" % rho("ours_vs_light", "nl_head"))

    out = "analysis/diagnostics/why_we_win.csv"
    t.to_csv(out, index=False)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
