"""Run the whole pipeline: download -> clean -> features -> train -> evaluate -> report.

Examples
    python run.py                 # real data from Elexon (downloads once, then cached)
    python run.py --demo          # simulated data, no network needed (for testing only)
    python run.py --rolling       # quarterly rolling-origin re-training over the test period
    python run.py --models linear,mlp --epochs 20
    python run.py --download-only # just fetch the data (needs only requests and pandas)
    python run.py --offline       # use previously downloaded data in data/processed/
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

from gridforecast import elexon
from gridforecast.config import PROCESSED_DIR, RESULTS_DIR, Config


def make_folds(ds: pd.DataFrame, cfg: Config, rolling: bool):
    """Return a list of (name, train_mask, val_mask, test_mask)."""
    td = ds["target_date"]
    if not rolling:
        return [("single split", (ds["split"] == "train").to_numpy(), (ds["split"] == "val").to_numpy(),
                 (ds["split"] == "test").to_numpy())]
    folds = []
    test_start = pd.Timestamp(cfg.test_start)
    last = td.max()
    q_start = test_start
    while q_start <= last:
        q_end = q_start + pd.offsets.QuarterBegin(startingMonth=1)
        val_start = q_start - pd.DateOffset(years=1)
        tr = (td < val_start).to_numpy()
        va = ((td >= val_start) & (td < q_start)).to_numpy()
        te = ((td >= q_start) & (td < q_end)).to_numpy()
        if te.any():
            folds.append((f"{q_start:%Y-%m}", tr, va, te))
        q_start = q_end
    return folds


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demo", action="store_true", help="use simulated data instead of downloading")
    p.add_argument("--start", default=None, help="first date to download (default 2018-01-01)")
    p.add_argument("--end", default=None, help="last date to download (default two days ago)")
    p.add_argument("--models", default=None, help="comma-separated subset of linear,mlp,seq")
    p.add_argument("--epochs", type=int, default=None, help="maximum training epochs")
    p.add_argument("--rolling", action="store_true", help="quarterly rolling-origin evaluation (slower)")
    p.add_argument("--no-oracle", action="store_true", help="skip the oracle-temperature ablation")
    p.add_argument("--download-only", action="store_true",
                   help="download the data to data/processed/ and stop (no PyTorch needed)")
    p.add_argument("--offline", action="store_true",
                   help="use data already saved in data/processed/ instead of downloading")
    args = p.parse_args(argv)

    cfg = Config()
    if args.start:
        cfg.data_start = args.start
    if args.end:
        cfg.data_end = args.end
    if args.models:
        cfg.models = tuple(m.strip() for m in args.models.split(","))
    if args.epochs:
        cfg.max_epochs = args.epochs
    if args.no_oracle:
        cfg.oracle_ablation = False
    t0 = time.time()

    # 1. data
    if args.demo:
        from gridforecast import synthetic
        print("DEMO MODE: generating simulated data (results are meaningless for the real grid)")
        tables = synthetic.simulate()
    elif args.offline:
        print(f"Loading previously downloaded data from {PROCESSED_DIR}")
        tables = elexon.load_raw_tables(PROCESSED_DIR)
    else:
        print(f"Downloading Elexon data from {cfg.data_start} (cached in data/raw/) ...")
        tables = elexon.download_all(cfg.data_start, cfg.data_end)
        elexon.save_raw_tables(tables, PROCESSED_DIR)
    if tables["demand"].empty:
        print("No demand data found. Check your internet connection (or run without --offline) and try again.")
        return 1
    if args.download_only:
        print(f"\nSaved to {PROCESSED_DIR}:")
        for name, df in tables.items():
            span = ""
            for col in ("settlementDate", "measurementDate"):
                if col in df and len(df):
                    span = f"  {df[col].min()} to {df[col].max()}"
            print(f"  {name + '.csv.gz':20s} {len(df):>9,} rows{span}")
        return 0

    # heavier imports only needed for modelling
    import torch

    from gridforecast import data, evaluate, report, train
    torch.set_num_threads(max(1, torch.get_num_threads()))

    # 2. cleaning and features
    demand, data_log = data.clean_demand(tables["demand"], cfg)
    temp = data.clean_temperature(tables["temperature"])
    ndf = data.clean_ndf(tables["ndf"], tables.get("ndf_update"), cfg=cfg)
    ds, ds_info = data.build_dataset(demand, temp, ndf, cfg)
    print(f"Dataset: {ds_info['rows_used']:,} rows; splits {ds['split'].value_counts().to_dict()}")
    for s in ("train", "val", "test"):
        if (ds["split"] == s).sum() < 48 * 30:
            print(f"Not enough data in the {s} split. Adjust the dates in gridforecast/config.py.")
            return 1
    series = demand.to_numpy()
    taus = list(cfg.quantiles)
    y_all = ds["y"].to_numpy()

    # 3. folds: train every model, predict validation and test
    folds = make_folds(ds, cfg, args.rolling)
    model_names = list(cfg.models) + (["mlp_oracle"] if cfg.oracle_ablation and ds_info["has_temperature"] else [])
    test_mask_all = np.zeros(len(ds), bool)
    preds = {n: np.full((len(ds), len(taus)), np.nan) for n in model_names + ["naive_week", "neso_dressed"]}
    recal = {n: np.full((len(ds), len(taus)), np.nan) for n in model_names}
    val_losses = {n: [] for n in model_names}
    histories = {}
    for fold_name, tr, va, te in folds:
        print(f"\n=== Fold {fold_name}: train {tr.sum():,} / val {va.sum():,} / test {te.sum():,} rows ===")
        test_mask_all |= te
        # benchmarks
        naive = ds["y_lag7d"].to_numpy()
        preds["naive_week"][te] = evaluate.dressed_point_forecast(naive, y_all, tr, taus)[te]
        if ds_info["has_ndf"]:
            try:
                preds["neso_dressed"][te] = evaluate.dressed_point_forecast(
                    ds["ndf"].to_numpy(), y_all, va, taus)[te]
            except ValueError:
                print("  NESO forecast has too little validation history for this fold; skipping it")
        # models
        for name in model_names:
            print(f"  Training {name} ...", flush=True)
            model, prep, hist = train.fit(name, ds, series, cfg, tr, va)
            val_q = train.predict(model, prep, va)
            val_losses[name].append(evaluate.pinball(y_all[va], val_q, taus).mean())
            preds[name][te] = train.predict(model, prep, te)
            # recalibrated copy: shift each quantile by the amount that would have made its
            # hit rate exact in this fold's validation data, then re-sort (applied to test only)
            shift = np.array([np.quantile(y_all[va] - val_q[:, j], t) for j, t in enumerate(taus)])
            recal[name][te] = np.sort(preds[name][te] + shift[None, :], axis=1)
            histories[name] = hist

    # 4. common evaluation sample
    test = ds[test_mask_all].copy()
    fc = {n: q[test_mask_all] for n, q in preds.items()}
    neso_ok = np.isfinite(fc["neso_dressed"]).all(axis=1)
    n_no_neso = int((~neso_ok).sum())
    if neso_ok.mean() >= 0.5:
        keep = neso_ok
        if (~keep).any():
            print(f"Scoring on the {keep.sum():,} test rows where the NESO benchmark exists "
                  f"(dropping {(~keep).sum():,})")
    else:
        print("NESO benchmark missing for most test rows; leaving it out of the comparison")
        fc.pop("neso_dressed")
        keep = np.ones(len(test), bool)
    for n in list(fc):
        keep &= np.isfinite(fc[n]).all(axis=1)
    test = test[keep]
    fc = {n: q[keep] for n, q in fc.items()}

    val_scores = {n: float(np.mean(v)) * 1.0 for n, v in val_losses.items()}
    selectable = {n: v for n, v in val_scores.items() if n != "mlp_oracle"}
    headline = min(selectable, key=selectable.get)
    # the recalibrated headline model is reported alongside, never used for selection
    fc[f"{headline}_recal"] = recal[headline][test_mask_all][keep]

    # 5. evaluation and report
    res = evaluate.evaluate_all(test, fc, taus)
    pairs = [(headline, "neso_dressed"), (headline, "naive_week")]
    if "seq" in fc and "mlp" in fc:
        pairs.append(("seq", "mlp"))
    if "mlp" in fc and "linear" in fc:
        pairs.append(("mlp", "linear"))
    if "mlp_oracle" in fc:
        pairs.append(("mlp_oracle", "mlp"))
    dm = evaluate.dm_table(res["daily"], [pr for pr in pairs if pr[0] != pr[1]])
    ndf_lead = our_lead = float("nan")
    if "neso_dressed" in fc:
        ndf_lead = float(((test.index - pd.to_datetime(test["ndf_publish"], utc=True)).dt.total_seconds() / 3600).median())
        our_lead = float(test["lead_h"].median())

    scheme = ("quarterly rolling-origin re-training (each quarter forecast by a model trained only on earlier data, "
              "with the preceding 12 months used for early stopping)") if args.rolling else \
        f"single chronological split: train to {cfg.train_end}, validate {cfg.val_start} to {cfg.val_end}, " \
        f"test from {cfg.test_start}"
    report.write_report({
        "results": res, "taus": taus, "test": test, "forecasts": fc, "headline": headline,
        "val_scores": val_scores, "dm": dm, "data_log": data_log, "ds_info": ds_info, "scheme": scheme,
        "histories": histories, "demo": args.demo, "argv": " ".join(sys.argv[1:] if argv is None else argv),
        "ndf_lead": ndf_lead, "our_lead": our_lead, "n_no_neso": n_no_neso if "neso_dressed" in fc else 0,
    })
    out = pd.DataFrame({"y": test["y"]}, index=test.index)
    for n, q in fc.items():
        for j, t in enumerate(taus):
            out[f"{n}_q{t:g}"] = np.round(q[:, j], 1)
    out.to_csv(RESULTS_DIR / "test_predictions.csv.gz")

    print("\nTest-period mean pinball loss (MW):")
    for n in fc:
        print(f"  {n:14s} {res['overall'][n]['mean_pinball_mw']:8.1f}   "
              f"90% coverage {res['overall'][n]['cov90']:.1%}")
    print(f"\nHeadline model (chosen on validation): {headline}")
    print(f"Report written to {RESULTS_DIR / 'RESULTS.md'}  ({time.time() - t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
