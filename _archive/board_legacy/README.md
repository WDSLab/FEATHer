# Retired board export scripts (moved here 10-14)

- `export_golden.py` — old 938-parameter FEATHer port (`src/feather_infer.c`, `make test` target test_golden).
- `export_batch.py` — batched golden export of the old port and three baselines.
- `export_golden_feather2in.py` — rejected `feather2in` port (input-side mixer, per-sensor gate).

They build FEATHer from `_archive/FEATHer_full_20261014/FEATHer.py` (every research option, as on 10-14) and still
write to `deployment/board/golden/`.  The shared C-array writer they used lives in `deployment/board/golden_io.py`.
