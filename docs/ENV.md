# 실행 환경 (2026-10-07)

논문 실험을 다시 돌리는 데 필요한 버전. 설치: `requirements.txt` → `bash setup_baselines.sh`(공식 저장소를 아래 커밋으로 받음).

## 파이썬

| 장소 | Python | PyTorch | 기타 |
|---|---|---|---|
| 로컬 (Windows, conda env `feather`) | 3.9.25 | 2.5.1+cu121 | numpy 1.26.4 · pandas 2.3.3 · scipy 1.13.1 |
| 서버 (Linux, 2×RTX PRO 6000, env `lee`) | 메모리 `reference_new_server_env_lee` 참고 | 2.8.0+cu128 | |

- 두 환경에서 같은 시드라도 GPU 연산 차이로 마지막 자리 값이 다를 수 있다. 정식 결과는 칸마다 한 곳에서 돌린 값 그대로 쓴다.
- AMP 는 모든 모델에서 끈다 (FITS 복소 기울기, TimesNet half cuFFT).

## 베이스라인 공식 저장소 (setup_baselines.sh 가 이 커밋으로 받음)

| 모델 | 커밋 |
|---|---|
| CMoS | `b696a0c33b5a` |
| CycleNet | `d807e51fc2dc` |
| DLinear | `0c113668a3b8` |
| DiPE_Linear | `b26e25015fba` |
| FITS | `d040bb015b62` |
| LMS_AutoTSF | `412884a19e16` |
| MDMLP_EIA | `0d68442c014a` |
| MixLinear | `71b5db62e38b` |
| PatchTST | `204c21efe0b3` |
| PhaseFormer | `ed1db61c6abf` |
| SAMformer | `71f10eaa696f` |
| SparseTSF | `b8c2740eecc8` |
| TQNet | `15e19cb23483` |
| TimeBase | `369b330f3d77` |
| TimeMixer | `e24610583b36` |
| TimesNet | `4e938a176710` |
| iTransformer | `c2426e68ca13` |

## 실행

```bash
python run.py --stage <ltsf|ltsf_select|select|main|lookback|ablation|control> --check
python run.py --stage main --ngpu 1 --jobs_per_gpu 3
python tools/results/merge_incoming.py
```
규칙: `docs/EXPERIMENT_PROTOCOL.md` · 고른 설정: `docs/chosen_{mfg,ltsf}.json` · 결과: `results/README.md`.
