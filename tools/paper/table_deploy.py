# -*- coding: utf-8 -*-
"""배포 결정표 — 실측 flash/SRAM 과 정확도를 조인한다 (논문 Table: deployability).

    python tools/paper/table_deploy.py

이 표가 답하는 질문은 "어느 모델이 제일 정확한가" 가 아니라
**"내 라인의 센서 D개를 H스텝 앞까지 예측하려면, 128KB/20KB 부품에 무엇을
올릴 수 있고 그 중 뭐가 제일 정확한가"** 이다.  정확도 표만으로는 답이 안 나오고
(안 들어가는 모델이 1등이다), 발자국 표만으로도 안 나온다.

입력
    results/board/board_footprint.csv   ARM 크로스컴파일 실측 (모델 x D x H)
    results/analysis/complexity.csv        모델 x 데이터 x H 의 파라미터 (미포팅 모델 판정용)
    results/_archive/pre_raw_20261007/mfg_raw.csv           정확도 (14 베이스라인 + FEATHer_mix + FEATHer_CI)
    results/_archive/pre_raw_20261007/ltsf_raw.csv          정확도 (14 베이스라인 + FEATHer_CI (+ FEATHer_mix))

★ 변종: 결과 CSV 는 두 변종을 이름으로 구분한다 (FEATHer_mix / FEATHer_CI).
  발자국 CSV 는 mix 를 "FEATHer" 로 기록하므로 FP_KEY 로 옮겨 찾는다.
    mix -> 측정 격자 {1,4,6,7,8,14,16,21,28,32} 에서 D<=21 전 H 통과, D=28 부터 SRAM 초과
           (2026-09-14 규칙 ① 적용 후 재측정, 스택 528B 포함; 이전 값 D<=14 는 포팅 탓)
    CI  -> 전 D 통과
  어느 변종을 쓸지는 이 표가 칸마다 정한다 - 들어가는 것 중 정확한 쪽.

칸의 상태 4가지 — 실측과 산술을 절대 섞지 않는다:
    FITS      포팅됨 + 실측 flash/SRAM 둘 다 예산 안
    NO        포팅됨 + 실측이 예산 초과 (어느 축인지 표시)
    EXCLUDED  가중치(params x 4B)만으로 flash 초과 -> 포팅 불필요, 산술적 배제
    UNPORTED  가중치는 들어가는데 C 포트가 없음 -> SRAM 미검증, 주장 불가
  UNPORTED 를 "안 들어간다" 로 세지 않는다.  그건 우리가 아직 안 한 일이지
  그 모델의 성질이 아니다.
"""
import argparse
import os

import pandas as pd

R = "results"
OUT_DIR = "analysis/deployment"

FLASH_BYTES = 128 * 1024
SRAM_BYTES = 20 * 1024
FIRMWARE_FLOOR = int(10.4 * 1024)      # 실측 (BOARD_RESULTS_20260824.md)
SEQ_LEN = 96
STACK_BYTES = 528                      # 실측 펌웨어 peak stack (08-24), measure_footprint.py 와 동일

MFG = ["Steel", "MetroPT3", "WindSCADA", "N2O", "EWELD", "BlastFurnace"]
LTSF = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Exchange",
        "Electricity", "Traffic"]
HS = [96, 192, 336, 720]

# 발자국 CSV 의 모델 키 <- 정확도 CSV 의 모델 이름
PORTED = {"FEATHer_mix", "FEATHer_CI", "DLinear", "CycleNet", "SparseTSF", "SparseTSF_MLP",
          "MixLinear", "DiPE_Linear", "FITS"}

# 결과 CSV 이름 -> 발자국 CSV 이름 (measure_footprint.py 는 mix 를 "FEATHer" 로 씀)
FP_KEY = {"FEATHer_mix": "FEATHer"}
_FP_INV = {v: k for k, v in FP_KEY.items()}


# ------------------------------------------------------------------- 입력
# 2026-09-23 재동결(mlp_rank 4) 모델의 실측.  board_footprint.csv 의 FEATHer /
# FEATHer_CI 행은 09-04 452p 모델(mlp_rank 16) 것이라, 같은 키를 이 파일로
# 덮어쓴다.  mix 는 재계산 포트(feather2r_infer.c, 정규화+밴드를 센서마다 다시
# 계산) 측정값이다 - 규칙 ④ 예외, 원고에 명시.
FROZEN_FP = f"{R}/board/board_footprint_r4_frozen.csv"
FROZEN_KEY = {"FEATHer_recompute_r4": "FEATHer", "FEATHer_CI_r4": "FEATHer_CI"}


def load_footprint():
    fp = pd.read_csv(f"{R}/board/board_footprint.csv")
    fz = pd.read_csv(FROZEN_FP)
    fz = fz[fz.model.isin(FROZEN_KEY)].assign(model=lambda d: d.model.map(FROZEN_KEY))
    fp = pd.concat([fp[~fp.model.isin(FROZEN_KEY.values())], fz], ignore_index=True)
    fp = fp.dropna(subset=["total_flash"])
    return {(r.model, int(r.D), int(r.H)): r for r in fp.itertuples()}


def load_params():
    """(model, data, H) -> 파라미터 수, 그리고 데이터셋 -> D.

    1순위는 결과 CSV 의 num_params 다 — 실제로 학습된 모델이 가졌던 수이므로
    표의 정확도와 같은 실행분에서 나온다.  complexity.csv 는 보조 (거기 없는
    조합이 있고, LMS/MDMLP 는 상류가 .cuda() 하드코딩이라 CPU 스윕에서 빠진다).
    """
    par = {}
    cx_path = f"{R}/analysis/complexity.csv"
    dim = {}
    if os.path.exists(cx_path):
        cx = pd.read_csv(cx_path)
        par.update({(r.model, r.data, int(r.pred_len)): int(r.params)
                    for r in cx.itertuples()})
        dim.update({r.data: int(r.D) for r in cx.itertuples()})

    for path in (f"{R}/_archive/pre_raw_20261007/mfg_raw.csv", f"{R}/_archive/pre_raw_20261007/ltsf_raw.csv"):
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path)
        for r in d.itertuples():
            par[(r.model, r.data, int(r.pred_len))] = int(r.num_params)
    return par, dim


def load_accuracy(group):
    """(model, data, H) -> 시드평균 MSE."""
    raw = pd.read_csv(f"{R}/_archive/pre_raw_20261007/{group}_raw.csv")
    acc = raw.groupby(["model", "data", "pred_len"]).MSE.mean().to_dict()
    return {(m, d, int(h)): v for (m, d, h), v in acc.items()}


# ------------------------------------------------------------------- 판정
def verdict(model, data, H, D, fp, par):
    """한 칸의 상태와 자원.  (state, note, flash_kb, sram_kb) 를 돌려준다."""
    key = (FP_KEY.get(model, model), D, H)
    if key in fp:
        r = fp[key]
        f_kb, s_kb = r.total_flash / 1024, r.total_sram / 1024
        if r.fits == "YES":
            return "FITS", "", f_kb, s_kb
        why = []
        if r.total_flash > FLASH_BYTES:
            why.append("flash")
        if r.total_sram > SRAM_BYTES:
            why.append("SRAM")
        return "NO", "+".join(why), f_kb, s_kb

    # 입력 버퍼(L*D*4B)는 모델과 무관하다.  이것만으로 SRAM 을 넘으면 어떤
    # 모델도 못 올린다 - Electricity(D=321) 123KB, Traffic(D=862) 331KB.
    # 측정하지 않은 게 아니라 잴 필요가 없는 칸이므로 "?" 가 아니라 EXCLUDED.
    buf = SEQ_LEN * D * 4 + STACK_BYTES
    if buf > SRAM_BYTES:
        return "EXCLUDED", f"입력버퍼만 {buf / 1024:.0f}KB", float("nan"), buf / 1024

    p = par.get((model, data, H))
    if p is None:
        return "?", "파라미터 미상", float("nan"), float("nan")
    w_kb = p * 4 / 1024
    if p * 4 + FIRMWARE_FLOOR > FLASH_BYTES:
        return "EXCLUDED", f"가중치만 {w_kb:.0f}KB", w_kb, float("nan")
    if model in PORTED:
        # 포팅은 됐는데 이 (D,H) 를 안 쟀다 = 측정 구멍
        return "?", "미측정", float("nan"), float("nan")
    return "UNPORTED", f"가중치 {w_kb:.0f}KB", w_kb, float("nan")


def build_rows(datasets, group, fp, par, dim, acc):
    rows = []
    models = sorted({m for (m, _, _) in acc})
    for data in datasets:
        D = dim[data]
        for H in HS:
            for m in models:
                st, note, f_kb, s_kb = verdict(m, data, H, D, fp, par)
                rows.append(dict(group=group, data=data, D=D, H=H, model=m,
                                 state=st, note=note, flash_kb=f_kb,
                                 sram_kb=s_kb,
                                 MSE=acc.get((m, data, H), float("nan")),
                                 params=par.get((m, data, H))))
    return pd.DataFrame(rows)


# ------------------------------------------------------- 칩당 센서 수
def streams_per_chip(fp, stack=528):
    """칩 하나가 감당하는 센서 스트림 수 (메모리 기준).

    모델은 하나만 올려두고 센서마다 돌려 쓴다.  그러면 SRAM 은
        고정분 = 정적 아레나(D=0 으로 외삽) + 스택
        가변분 = 스트림당 입력버퍼 L*4B (+ 재계산 포트는 센서당 아레나 8B)
    로 갈리고, 남는 자리를 스트림당 비용으로 나눈 값이 답이다.

    ⚠ 이 수는 우리에게 유리하지 않다 (2026-09-29 재측정).  섞는 FEATHer-mix
    는 전 호라이즌 43 으로, 안 섞는 경량 모델들(H=96 48~50, H=720 DiPE 42 ·
    FITS 47 · 나머지 48~49)보다 대부분 적다 - 섞기의 대가다.  "호라이즌이
    길수록 남들이 못 버틴다"는 옛 서술(FITS 34 / DiPE 31)은 포팅 탓이었고
    09-14 에 폐기됐다.  그대로 싣는다: 우리가 지는 칸이 있어야 표를 믿는다.

    ⚠ 메모리 기준일 뿐이다.  재예측 주기가 짧으면 연산이 먼저 막히고,
    거기서는 우리가 불리하다(08-24 보드: SparseTSF 47ms vs FEATHer 655ms).
    확정 모델의 보드 latency 는 아직 측정 전이므로 연산 한계는 비워 둔다.
    """
    # 스트림당 비용 = 입력버퍼 L*4B + 센서당 아레나 증가분.  센서를 하나씩
    # 처리하는 포트는 아레나가 D 와 무관해 증가분 0 (09-14 이전 계산과 동일),
    # 재계산 포트(재동결 mix)는 센서당 8B 가 붙는다 - D=1 과 D=4 빌드의 기울기.
    rows = []
    for (m, D, H), r in fp.items():
        if D != 1:
            continue
        r4 = fp.get((m, 4, H))
        per_arena = (r4.sram - r.sram) / 3 if r4 is not None else 0.0
        per = SEQ_LEN * 4 + per_arena
        fixed = r.sram - per_arena + stack      # D=0 로 외삽한 고정분
        free = SRAM_BYTES - fixed
        rows.append(dict(model=_FP_INV.get(m, m), H=H, fixed_B=int(fixed),
                         free_B=int(free), per_stream_B=int(per),
                         streams=int(free // per) if free > 0 else 0))
    return pd.DataFrame(rows).sort_values(["model", "H"])


# ------------------------------------------------------------------ 출력
def winners(df):
    """(data, H) 별로 배포가능한 것 중 최정확 + 전체 최정확."""
    out = []
    for (data, H), g in df.groupby(["data", "H"]):
        dep = g[(g.state == "FITS") & g.MSE.notna()]
        allm = g[g.MSE.notna()]
        best_dep = dep.loc[dep.MSE.idxmin()] if len(dep) else None
        best_all = allm.loc[allm.MSE.idxmin()] if len(allm) else None
        out.append(dict(
            data=data, H=H, n_fits=len(dep),
            best_deployable=None if best_dep is None else best_dep.model,
            MSE_deployable=float("nan") if best_dep is None else best_dep.MSE,
            best_overall=None if best_all is None else best_all.model,
            MSE_overall=float("nan") if best_all is None else best_all.MSE))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="both", choices=["mfg", "ltsf", "both"])
    a = ap.parse_args()

    fp = load_footprint()
    par, dim = load_params()

    frames = []
    for group, datasets in (("mfg", MFG), ("ltsf", LTSF)):
        if a.group not in (group, "both"):
            continue
        acc = load_accuracy(group)
        df = build_rows(datasets, group, fp, par, dim, acc)
        frames.append(df)

        print(f"\n=== {group} — 칸 상태 집계 ===")
        print(df.state.value_counts().to_string())
        holes = df[df.state == "?"]
        if len(holes):
            print("\n측정/데이터 구멍:")
            print(holes.groupby(["model", "note"]).size().to_string())

        print(f"\n=== {group} — (데이터셋, H) 별 배포가능 최정확 ===")
        w = winners(df)
        print(w.to_string(index=False,
                          float_format=lambda v: f"{v:.4f}"))

    sp = streams_per_chip(fp)
    print("")
    print("=== 칩 하나가 감당하는 센서 수 (메모리 기준, 연산 미반영) ===")
    print(sp.pivot(index="model", columns="H", values="streams").to_string())

    os.makedirs(OUT_DIR, exist_ok=True)
    sp.to_csv(f"{OUT_DIR}/streams_per_chip.csv", index=False)
    out = f"{OUT_DIR}/deploy_cells.csv"
    pd.concat(frames, ignore_index=True).to_csv(out, index=False)
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
