"""흩어진 결과 CSV를 벤치마크별 raw 파일로 병합.

    python tools/paper/merge_raw.py

**2026-08-31 구조 변경 — 베이스라인과 FEATHer를 분리한다.**
베이스라인은 확정(재실험 없음)이고 FEATHer는 모델이 아직 바뀌는 중이라,
FEATHer가 갱신될 때마다 베이스라인까지 다시 병합할 이유가 없다.

출력 (전부 seq_len=96 본 실험, 3시드):

    ★ 얼린 것 — FEATHer가 바뀌어도 건드릴 일 없음
    results/_archive/pre_raw_20261007/mfg_baselines.csv    베이스라인 14 × 6데이터  × 4H × 3seed
    results/_archive/pre_raw_20261007/ltsf_baselines.csv   베이스라인 14 × 8데이터  × 4H × 3seed

    ☆ 계속 바뀌는 것 — 모델 확정 후 재생성됨
    results/_archive/pre_raw_20261007/feather_mfg.csv      FEATHer_mix + FEATHer_CI × 6데이터 × 4H × 3seed
    results/_archive/pre_raw_20261007/feather_ltsf.csv     FEATHer_CI + FEATHer_mix × 8데이터 × 4H × 3seed

    (호환) results/_archive/pre_raw_20261007/mfg_raw.csv / ltsf_raw.csv = 위 둘을 그냥 이어붙인 것.
    build_paper_tables.py 등 기존 소비자가 이걸 읽으므로 유지한다.
    --no_combined 를 주면 만들지 않는다.

FEATHer 행 선택 규칙 (두 변종을 모두 싣는다; 2026-09-23 재동결 = mlp_rank 4):
  mfg   FEATHer_mix = results/_archive/pre_raw_20261007/feather_mfg_candR4.csv      "candR4"    Steel H96 296p
        FEATHer_CI  = results/_archive/pre_raw_20261007/feather_mfg_candR4_ci.csv   "candR4ci"  Steel H96  76p
  LTSF  FEATHer_CI  = results/_archive/pre_raw_20261007/feather_ltsf_candR4.csv     "ltsfR4"    ETTh1 H96 298p
        FEATHer_mix = results/_archive/pre_raw_20261007/feather_ltsf_candR4_mix.csv "ltsfR4mix" ETTh1 H96 454p
  로더가 num_params 로 각 변종을 검증한다.
  (09-04 의 452p 판 입력은 results/_archive/feather/ 에 그대로 있다:
   feather_mfg_shipping 452 · feather_mfg_ci_contrast 112 ·
   feather_ltsf_shipping 334 · feather_ltsf_mix 634.)

`source` 열에 어느 파일에서 왔는지 남긴다 (추적용).
"""
import argparse
import os
import pandas as pd

R = "results"
SRC = "results/_archive"      # 병합 입력 원본 (provenance 보관)
MFG = ["Steel", "WindSCADA", "BlastFurnace", "MetroPT3", "N2O", "EWELD"]
LTSF = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity", "Traffic", "Exchange"]


def tag(df, src):
    df = df.copy()
    df["source"] = src
    return df


# ---------------------------------------------------------------- baselines
def load_baselines(datasets):
    """확정된 베이스라인 전부. SparseTSF_MLP 는 별도 실행분이라 따로 읽는다."""
    out = []
    b = pd.read_csv(f"{SRC}/baselines/fcst_results_baselines.csv")
    out.append(tag(b[(b.exp_tag == "main") & (b.data.isin(datasets))],
                   "fcst_results_baselines.csv"))

    # SparseTSF-MLP: 보드 실측 라인업에 있고 배포집합에 드는데 메인 CSV 에
    # 빠져 있었다 (2026-08-31 발견).  mfg 는 08-31, LTSF 는 09-09 에 별도 실행.
    for fn in ("mfg_sparsetsf_mlp.csv", "ltsf_sparsetsf_mlp.csv"):
        p = f"{SRC}/baselines/{fn}"
        if os.path.exists(p):
            s = pd.read_csv(p)
            s = s[(s.exp_tag == "main") & (s.data.isin(datasets))]
            if len(s):
                out.append(tag(s, fn))
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------ new baselines (10-06)
# CMoS (ICML'25) · TimeBase (ICML'25) · PhaseFormer (ICLR'26) · SAMformer (ICML'24), added
# 2026-10-06 (memory project_cmos_baseline_1006).  Their runs live in their own CSVs (local
# seed 2025, server seeds 2026-27 and LTSF); they are merged into *_baselines.csv / *_raw.csv
# with the frozen 14 so every table reads one file.  Seeds may be incomplete while the runs
# are in progress - the report below says so, and the manuscript builder marks such models.
NEW_MODELS = ["CMoS", "TimeBase", "PhaseFormer", "SAMformer"]
NEW_FILES = {
    "mfg": [f"{R}/_archive/pre_raw_20261007/cmos_mfg.csv", f"{R}/_archive/pre_raw_20261007/new3_mfg.csv", f"{R}/new4_mfg_s2627.csv",
            f"{R}/_archive/pre_raw_20261007/timebase_lr_mfg.csv"],
    "ltsf": [f"{R}/new4_ltsf.csv"],
}
# TimeBase mfg lr: the official lr 0.1 (exp_tag "main") is re-selected on the VALIDATION loss
# among {0.1, 0.01, 0.001} (tools/dev/timebase_lr_verdict.py; the DLinear-ETTh2 precedent).
# Set this to "tb_lr1e-2" / "tb_lr1e-3" once the verdict is in; seeds 2026-27 must then be run
# with the same lr and exp_tag.
TIMEBASE_MFG_LR_TAG = "tb_lr1e-3"      # 10-07 verdict: val loss 0.7085 (lr 0.001) < 0.7087 (0.01) < 0.7109 (0.1), best in 14/24 cells
# parameter counts that identify the configuration (Steel D = 6 / ETTh1 D = 7, H = 96);
# a wrong wrapper default would show up here, not in the MSE
NEW_PARAMS = {("mfg", "CMoS"): ("Steel", 96, 930), ("mfg", "TimeBase"): ("Steel", 96, 58),
              ("mfg", "PhaseFormer"): ("Steel", 96, 1500), ("mfg", "SAMformer"): ("Steel", 96, 21740)}


def load_new_baselines(group, datasets):
    """every finished run of the four new baselines (missing files are skipped)."""
    out = []
    for p in NEW_FILES[group]:
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p, on_bad_lines="skip")
        d = d[d.model.isin(NEW_MODELS) & d.data.isin(datasets)]
        want_tag = d.model.map(lambda m: TIMEBASE_MFG_LR_TAG if (m == "TimeBase" and group == "mfg")
                               else "main")
        d = d[d.exp_tag == want_tag]
        if len(d):
            out.append(tag(d, os.path.basename(p)))
    if not out:
        return pd.DataFrame(columns=["exp_tag", "model", "data", "pred_len", "seq_len", "seed",
                                     "MSE", "MAE", "num_params", "source"])
    d = pd.concat(out, ignore_index=True)
    d = d[d.seq_len == 96]
    # a cell re-run later replaces the earlier row (same model/data/H/seed)
    if "timestamp" in d.columns:
        d = d.sort_values("timestamp")
    d = d.drop_duplicates(["model", "data", "pred_len", "seed"], keep="last")
    for (g, m), (data, H, want) in NEW_PARAMS.items():
        if g != group:
            continue
        got = d[(d.model == m) & (d.data == data) & (d.pred_len == H)].num_params.unique()
        assert len(got) == 0 or list(got) == [want], (
            f"{m} {data} H={H} num_params={got}, expected {want} (wrapper default changed?)")
    return d


# ------------------------------------------------------------------ FEATHer
# 확정 모델(2026-09-23 재동결, mlp_rank 4, Steel 296p)의 전수 실행분.  두 벤치마크 모두 한 파일에
# 전 셀이 들어 있어 앵커·별도실행분 병합이 필요 없다 — 옛 822p 시절의
# ltsf_final2 + ltsf_highD + highD 프로브 3중 병합은 그래서 사라졌다.
# 두 변종을 이름으로 구분한다 (2026-09-10).  예전에는 "FEATHer" 한 이름이
# mfg 에서는 mix, LTSF 에서는 ci_full 을 뜻했고, 그 모호함이 표/설정/측정
# 도구가 서로 다른 모델을 가리키는 사고의 원인이었다.  또 도메인별로
# 정확도가 나은 쪽만 싣는 것은 선택 편향이라, 이제 두 변종을 모두 싣는다.
#   FEATHer_mix  channel_mode=mix     (채널 혼합, 재계산 포트로 칩당 43센서)
#   FEATHer_CI   channel_mode=ci_full (채널 독립, 전 D 통과)
# 각 파일은 num_params 로 설정을 검증한다 - 옛/다른 변종 CSV 를 가리켜도
# MSE 는 비슷해서 안 걸리고 파라미터로만 잡힌다.


def _load_feather(path, exp_tag, name, expect):
    f = pd.read_csv(path, on_bad_lines="skip")
    f = f[f.exp_tag == exp_tag].copy()
    data, H, want = expect
    got = f[(f.data == data) & (f.pred_len == H)].num_params.unique()
    assert list(got) == [want], (
        f"{path}: {data} H={H} num_params={got}, {name} 이면 {want} 여야 한다")
    f["model"] = name
    return tag(f, os.path.basename(path))


def load_feather_mfg():
    return pd.concat([
        _load_feather(f"{R}/_archive/pre_raw_20261007/feather_mfg_candR4.csv", "candR4",
                      "FEATHer_mix", ("Steel", 96, 296)),
        _load_feather(f"{R}/_archive/pre_raw_20261007/feather_mfg_candR4_ci.csv", "candR4ci",
                      "FEATHer_CI", ("Steel", 96, 76)),
    ], ignore_index=True)


def load_feather_ltsf():
    return pd.concat([
        _load_feather(f"{R}/_archive/pre_raw_20261007/feather_ltsf_candR4.csv", "ltsfR4",
                      "FEATHer_CI", ("ETTh1", 96, 298)),
        _load_feather(f"{R}/_archive/pre_raw_20261007/feather_ltsf_candR4_mix.csv", "ltsfR4mix",
                      "FEATHer_mix", ("ETTh1", 96, 454)),
    ], ignore_index=True)


# -------------------------------------------------------------------- build
def _finish(df, datasets):
    df = df.copy()
    df["exp_tag"] = "main"                      # 병합 후 태그 통일
    df = df[df.data.isin(datasets)]
    return df.sort_values(["model", "data", "pred_len", "seed"]).reset_index(drop=True)


def _report(label, df, datasets, out):
    cells = df.groupby(["model", "data", "pred_len"]).ngroups
    n_models = df.model.nunique()
    print(f"  → {out}  {len(df)}행 / {n_models}모델 / {cells}셀 "
          f"(완전체 {n_models * len(datasets) * 4})")
    miss = []
    for m in sorted(df.model.unique()):
        n = df[df.model == m].groupby(["data", "pred_len"]).ngroups
        if n != len(datasets) * 4:
            miss.append(f"{m} {n}/{len(datasets) * 4}")
    print(f"  {label} 결측:", ", ".join(miss) if miss else "없음")


def build(name, datasets, feather_loader, combined=True):
    print(f"[{name}]")

    new = load_new_baselines(name, datasets)
    base = _finish(pd.concat([load_baselines(datasets), new], ignore_index=True), datasets)
    bout = f"{R}/_archive/pre_raw_20261007/{name}_baselines.csv"
    base.to_csv(bout, index=False)
    _report("베이스라인", base, datasets, bout)
    for m in NEW_MODELS:                        # seeds still missing for the new baselines
        n = new[new.model == m].groupby(["data", "pred_len"]).seed.nunique() if len(new) else pd.Series(dtype=int)
        full = len(datasets) * 4
        n3 = int((n == 3).sum())
        print(f"  {m:12s} 3시드 완료 {n3}/{full}칸, 시드 1~2개 {int(((n > 0) & (n < 3)).sum())}칸, "
              f"없음 {full - len(n)}칸")

    feat = _finish(feather_loader(), datasets)
    fout = f"{R}/_archive/pre_raw_20261007/feather_{name}.csv"
    feat.to_csv(fout, index=False)
    _report("FEATHer", feat, datasets, fout)

    if combined:
        allrows = pd.concat([base, feat], ignore_index=True)
        allrows = allrows.sort_values(
            ["model", "data", "pred_len", "seed"]).reset_index(drop=True)
        cout = f"{R}/_archive/pre_raw_20261007/{name}_raw.csv"
        allrows.to_csv(cout, index=False)
        print(f"  → {cout}  {len(allrows)}행 / {allrows.model.nunique()}모델 (호환용 합본)")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_combined", action="store_true",
                    help="호환용 *_raw.csv 를 만들지 않는다")
    a = ap.parse_args()
    build("mfg", MFG, load_feather_mfg, combined=not a.no_combined)
    build("ltsf", LTSF, load_feather_ltsf, combined=not a.no_combined)
