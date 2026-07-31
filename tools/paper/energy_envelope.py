# -*- coding: utf-8 -*-
"""Energy / battery-life envelope from the Cortex-M3 board measurements.

Turns the measured per-inference latency + active current (NUCLEO-F103RB,
`deployment/EDGE_EXPERIMENT.md` 2026-07-27 log) into two paper numbers:

  (1) Pure measured:   energy per inference (mJ) and, at a given duty cycle,
      the compute share of standby ("35 mJ/inference; at 60 s duty the compute
      is ~1.5% of standby = essentially free").
  (2) Battery-life projection: with a coin cell + duty-cycled inference and a
      PRODUCTION bare-MCU sleep current (datasheet STOP ~14-24 uA — NOT the
      NUCLEO 11.4 mA, which is ST-LINK-dominated and unrepresentative), how many
      years the node runs. Stated as a projection.

The point (deployment selling): latency does NOT track parameter count, so
FEATHer (453p, 349 ms) outlasts FITS/DiPE (slower despite similar size) on a
battery — energy x robustness Pareto.

MEASURED INPUTS (update from EDGE_EXPERIMENT.md if re-measured):
  active current ~30 mA for all models (29-31 mA measured, model-independent);
  Vdd 3.3 V; per-inference latency below (ms).

Usage:
    python tools/paper/energy_envelope.py
    python tools/paper/energy_envelope.py --battery_mah 225 --duty_s 60 --sleep_ua 20
"""

import argparse

# latency ms @ F103RB -O2 soft-float (EDGE_EXPERIMENT.md 2026-07-27).
# MixLinear/CycleNet/DLinear = TBD (board port pending config lock).
LATENCY_MS = {
    "persistence": 0.08,
    "SparseTSF":   47.0,
    "FEATHer":     349.0,
    "FITS":        796.0,
    "DiPE_Linear": 1312.0,
    # "MixLinear": None,  # <- fill after board measurement
}
ACTIVE_MA = 30.0   # measured 29-31 mA, model-independent
VDD = 3.3          # V


def main():
    p = argparse.ArgumentParser(description="Energy / battery-life envelope")
    p.add_argument("--battery_mah", type=float, default=225.0,
                   help="battery capacity (mAh); CR2032 coin cell ~225")
    p.add_argument("--duty_s", type=float, default=60.0,
                   help="one inference every N seconds (duty period)")
    p.add_argument("--sleep_ua", type=float, default=20.0,
                   help="production bare-MCU STOP-mode sleep current (uA)")
    p.add_argument("--vdd", type=float, default=VDD)
    p.add_argument("--active_ma", type=float, default=ACTIVE_MA)
    args = p.parse_args()

    i_sleep_ma = args.sleep_ua / 1000.0
    print(f"Vdd={args.vdd} V | active={args.active_ma} mA | "
          f"sleep(STOP)={args.sleep_ua} uA | battery={args.battery_mah} mAh | "
          f"duty=1 inference / {args.duty_s:g} s\n")
    print(f"{'model':13s}{'lat(ms)':>9s}{'E/inf(mJ)':>11s}"
          f"{'avg(uA)':>10s}{'life(yr)':>10s}{'compute%':>10s}")
    for m, lat_ms in LATENCY_MS.items():
        lat_s = lat_ms / 1000.0
        e_inf_mj = args.active_ma * args.vdd * lat_s      # mA*V*s = mJ
        # average current over one duty period (active window + sleep rest)
        active_frac = min(lat_s / args.duty_s, 1.0)
        avg_ma = args.active_ma * active_frac + i_sleep_ma * (1 - active_frac)
        life_h = args.battery_mah / avg_ma                # mAh / mA = hours
        life_yr = life_h / 24.0 / 365.0
        # compute share of the standby energy budget over one period
        e_active_mj = e_inf_mj
        e_sleep_mj = i_sleep_ma * args.vdd * (args.duty_s - lat_s)
        compute_pct = 100.0 * e_active_mj / (e_active_mj + e_sleep_mj)
        print(f"{m:13s}{lat_ms:>9.2f}{e_inf_mj:>11.2f}"
              f"{avg_ma*1000:>10.1f}{life_yr:>10.2f}{compute_pct:>10.2f}")
    print("\n(life = projection: STOP-mode datasheet sleep, duty-cycled inference.)")


if __name__ == "__main__":
    main()
