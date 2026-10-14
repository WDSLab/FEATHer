# Retired tools (moved 10-14)

Finished analyses and one-off scripts, kept as the record of how earlier numbers were produced.  Paths inside them
still assume their old location under tools/ (and some use FEATHer options removed on 10-14 - see
_archive/FEATHer_full_20261014/), so they are not expected to run as they are.

- search/   08 FEATHer configuration searches (frozen config choice)
- audit/migrate_checkpoints.py   07-31 checkpoint folder migration
- paper/coupling_vs_mixing.py    10-06 coupling vs mixing gain (pre-declared rule failed)
- paper/stability_vs_mixing.py   10-09 pre-registered one-shot test (docs/prereg_stability_1009.md)
- paper/memory_matched.py        10-07 memory-matched comparison (input-length claim dropped from the main text)
- dev/control_verdict.py         10-06 control-arm verdict
- dev/sensor_lag.py, dev/sensor_mixing_probe.py   09 dev probes
- dev/gate_swap.py               gate-swap test; its load_worker() now lives in tools/dev/worker.py
