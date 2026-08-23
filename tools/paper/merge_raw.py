"""흩어진 결과 CSV를 벤치마크별 단일 raw 파일로 병합.

    python tools/paper/merge_raw.py

출력 (둘 다 seq_len=96 본 실험, 3시드):
    results/mfg_raw.csv    베이스라인 13 + FEATHer-N     × 6데이터 × 4H × 3seed
    results/ltsf_raw.csv   베이스라인 13 + FEATHer-CIN   × 8데이터 × 4H × 3seed

FEATHer 행은 실험이 CSV마다 흩어져 있어 여기서 규칙대로 골라 넣는다:
  - mfg  = feather_final.csv, exp_tag "final" (N 셀, ds16_k7_B4)
  - LTSF = ltsf_final2.csv, 레벨 앵커는 train-only 규칙(2026-08-15)
           → Exchange만 last_std, 나머지 5개는 window-mean(standard)
  - LTSF 고채널(Electricity/Traffic)은 별도 프로브 CSV가 생기면 자동 편입

`source` 열에 어느 파일에서 왔는지 남긴다 (추적용).
"""
import os
import pandas as pd

R = "results"
SRC = "results/_archive"      # 병합 입력 원본 (provenance 보관)
MFG = ["Steel", "WindSCADA", "BlastFurnace", "MetroPT3", "N2O", "EWELD"]
LTSF = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity", "Traffic", "Exchange"]

# LTSF 레벨 앵커: 데이터셋 → ltsf_final2.csv 안의 exp_tag
ANCHOR = {"Exchange": "final_last_std"}
ANCHOR_DEFAULT = "final_standard"

# 고채널 CIN 프로브 (있으면 편입) — exp_tag → 채택 여부
HIGH_D_PROBE = ("probes/probe_ltsf_highD.csv", "cin_probe")


def tag(df, src):
    df = df.copy()
    df["source"] = src
    return df


def load_baselines(datasets):
    b = pd.read_csv(f"{SRC}/baselines/fcst_results_baselines.csv")
    b = b[(b.exp_tag == "main") & (b.data.isin(datasets))]
    return tag(b, "fcst_results_baselines.csv")


def load_feather_mfg():
    f = pd.read_csv(f"{SRC}/feather/feather_final.csv")
    return tag(f[f.exp_tag == "final"], "feather_final.csv")


def load_feather_ltsf():
    lt = pd.read_csv(f"{SRC}/feather/ltsf_final2.csv")
    keep = [t == ANCHOR.get(d, ANCHOR_DEFAULT) for d, t in zip(lt.data, lt.exp_tag)]
    out = [tag(lt[keep], "ltsf_final2.csv")]

    path, want = HIGH_D_PROBE
    if os.path.exists(f"{R}/{path}"):
        hd = pd.read_csv(f"{R}/{path}")
        hd = hd[hd.exp_tag == want]
        if len(hd):
            out.append(tag(hd, os.path.basename(path)))
            print(f"  + 고채널 프로브 편입: {len(hd)}행 "
                  f"({', '.join(sorted(hd.data.unique()))})")
    return pd.concat(out, ignore_index=True)


def build(name, datasets, feather_loader):
    print(f"[{name}]")
    df = pd.concat([load_baselines(datasets), feather_loader()], ignore_index=True)
    df["exp_tag"] = "main"                      # 병합 후 태그 통일
    df = df[df.data.isin(datasets)]
    df = df.sort_values(["model", "data", "pred_len", "seed"]).reset_index(drop=True)

    out = f"{R}/{name}_raw.csv"
    df.to_csv(out, index=False)

    cells = df.groupby(["model", "data", "pred_len"]).ngroups
    full = len(df.model.unique()) * len(datasets) * 4
    print(f"  → {out}  {len(df)}행 / {len(df.model.unique())}모델 / "
          f"{cells}셀 (완전체 {full})")
    miss = []
    for m in sorted(df.model.unique()):
        n = df[df.model == m].groupby(["data", "pred_len"]).ngroups
        if n != len(datasets) * 4:
            miss.append(f"{m} {n}/{len(datasets)*4}")
    print("  결측:", ", ".join(miss) if miss else "없음")
    return df


if __name__ == "__main__":
    build("mfg", MFG, load_feather_mfg)
    build("ltsf", LTSF, load_feather_ltsf)
