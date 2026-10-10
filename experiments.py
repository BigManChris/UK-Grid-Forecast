"""Design experiments scored on the VALIDATION year only.

The test period is never looked at here. Each variant is trained on the
training years and scored on the validation year; the winning settings are
then fixed in gridforecast/config.py before the final test-period run.

    python experiments.py --offline        # uses data/processed/ (run `python run.py --download-only` first)
    python experiments.py --demo           # simulated data
"""

from __future__ import annotations

import argparse
import dataclasses
import sys

import numpy as np
import pandas as pd

from gridforecast import data, elexon, evaluate, train
from gridforecast.config import PROCESSED_DIR, RESULTS_DIR, Config

VARIANTS = {
    "baseline (absolute MW, trend feature)": dict(use_trend=True, relative_target=False),
    "no trend feature": dict(use_trend=False, relative_target=False),
    "relative to last 24h, trend feature": dict(use_trend=True, relative_target=True),
    "relative to last 24h, no trend": dict(use_trend=False, relative_target=True),
}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demo", action="store_true")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--models", default="linear,mlp,seq")
    p.add_argument("--seeds", default="42,43")
    args = p.parse_args(argv)

    base = Config()
    if args.demo:
        from gridforecast import synthetic
        tables = synthetic.simulate()
    elif args.offline:
        tables = elexon.load_raw_tables(PROCESSED_DIR)
    else:
        tables = elexon.download_all(base.data_start, base.data_end)
    demand, _ = data.clean_demand(tables["demand"], base)
    ds, _ = data.build_dataset(demand, data.clean_temperature(tables["temperature"]),
                               data.clean_ndf(tables["ndf"], tables.get("ndf_update"), cfg=base), base)
    series = demand.to_numpy()
    tr = (ds["split"] == "train").to_numpy()
    va = (ds["split"] == "val").to_numpy()
    y = ds["y"].to_numpy()
    taus = list(base.quantiles)

    rows = []
    for vname, settings in VARIANTS.items():
        for model in args.models.split(","):
            for seed in [int(s) for s in args.seeds.split(",")]:
                cfg = dataclasses.replace(base, seed=seed, **settings)
                m, prep, _ = train.fit(model, ds, series, cfg, tr, va, verbose=False)
                q = train.predict(m, prep, va)
                s = evaluate.summarise(y[va], q, taus)
                rows.append({"variant": vname, "model": model, "seed": seed,
                             "val_pinball_mw": s["mean_pinball_mw"], "val_cov90": s["cov90"]})
                print(f"{vname:40s} {model:7s} seed {seed}: pinball {s['mean_pinball_mw']:6.1f} MW, "
                      f"90% coverage {s['cov90']:.1%}", flush=True)
    df = pd.DataFrame(rows)
    summary = df.groupby(["variant", "model"], sort=False).agg(
        val_pinball_mw=("val_pinball_mw", "mean"), seed_spread_mw=("val_pinball_mw", np.ptp),
        val_cov90=("val_cov90", "mean")).reset_index()
    print("\n" + summary.round(3).to_string(index=False))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS_DIR / "validation_experiments.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
