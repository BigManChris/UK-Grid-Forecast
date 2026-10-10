"""Figures and the auto-generated results report (results/RESULTS.md)."""

from __future__ import annotations

import datetime as dt
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import FIGURES_DIR, RESULTS_DIR  # noqa: E402

# validated categorical palette (fixed order) and a single-hue ramp for interval bands
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
# each forecaster keeps the same colour in every chart, whichever others are shown
MODEL_SLOT = {"naive_week": 0, "neso_dressed": 1, "linear": 2, "mlp": 3, "seq": 4, "recal": 5, "mlp_oracle": 6}
BANDS = {95: "#d6e6f8", 80: "#a9c9ef", 50: "#6fa3e3"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

LABELS = {
    "naive_week": "Naive (same time last week)",
    "neso_dressed": "NESO official forecast (error-dressed)",
    "linear": "Linear quantile regression",
    "mlp": "MLP",
    "seq": "MLP + 7-day CNN encoder",
    "mlp_oracle": "MLP + actual temperature (oracle, not valid)",
}


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)


def _colors(names):
    out = {}
    for n in names:
        key = "recal" if n.endswith("_recal") else n
        out[n] = SERIES[MODEL_SLOT.get(key, 7)]
    return out


def fan_chart(test, Q, taus, ndf_q, start, days, path, title):
    taus = list(taus)
    m = (test["target_date"] >= start) & (test["target_date"] < start + pd.Timedelta(days=days))
    t = test.index[m].tz_convert("Europe/London")
    q = Q[m.to_numpy()] / 1000
    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=130)
    for level, (lo, hi) in ((95, (0.025, 0.975)), (80, (0.1, 0.9)), (50, (0.25, 0.75))):
        ax.fill_between(t, q[:, taus.index(lo)], q[:, taus.index(hi)], color=BANDS[level], lw=0,
                        label=f"{level}% interval", step=None)
    ax.plot(t, q[:, taus.index(0.5)], color=SERIES[0], lw=1.6, label="Forecast median")
    if ndf_q is not None:
        ax.plot(t, ndf_q[m.to_numpy()] / 1000, color=SERIES[1], lw=1.4, ls="--", label="NESO forecast")
    ax.plot(t, test.loc[m, "y"] / 1000, color=INK, lw=1.4, label="Actual")
    _style(ax, title, "", "National demand (GW)")
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.DayLocator(tz="Europe/London"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b", tz="Europe/London"))
    ax.legend(frameon=False, ncol=6, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def reliability_chart(overall, taus, path):
    names = list(overall)
    col = _colors(names)
    fig, ax = plt.subplots(figsize=(6, 5.4), dpi=130)
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls=":", label="Perfect calibration")
    for n in names:
        hits = [overall[n][f"hit_{t}"] for t in taus]
        ax.plot(taus, hits, marker="o", ms=5, lw=2, color=col[n], label=LABELS.get(n, n))
    _style(ax, "Reliability: how often the actual falls below each quantile",
           "Quantile level", "Observed share below the forecast quantile")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def tod_chart(by_tod, path):
    names = list(by_tod)
    col = _colors(names)
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=130)
    for n in names:
        s = by_tod[n]
        ax.plot(s.index, s.values, lw=2, color=col[n], label=LABELS.get(n, n))
    _style(ax, "Mean pinball loss by time of day (UK local)", "Hour of day", "Mean pinball loss (MW)")
    ax.set_xticks(range(0, 25, 3))
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def bins_chart(bins, path, title):
    fig, ax = plt.subplots(figsize=(8, 4), dpi=130)
    x = np.arange(len(bins))
    ax.bar(x - 0.2, bins["expected"], width=0.38, color="#c3c2b7", label="Expected if calibrated")
    ax.bar(x + 0.2, bins["observed"], width=0.38, color=SERIES[0], label="Observed")
    ax.set_xticks(x, bins["bin"], rotation=35, fontsize=8)
    _style(ax, title, "Between forecast quantiles", "Share of outcomes")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def training_chart(histories, path):
    names = list(histories)
    col = _colors(names)
    fig, ax = plt.subplots(figsize=(8, 4), dpi=130)
    for n in names:
        h = histories[n]
        ax.plot(h["epoch"], h["val_pinball_mw"], lw=2, color=col[n], label=f"{LABELS.get(n, n)} (validation)")
        ax.plot(h["epoch"], h["train_pinball_mw"], lw=1, ls="--", color=col[n])
    _style(ax, "Training curves (dashed = training loss)", "Epoch", "Mean pinball loss (MW)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _fmt_table(df: pd.DataFrame, floatfmt: dict | None = None) -> str:
    floatfmt = floatfmt or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append(format(v, floatfmt.get(c, ".1f")) if np.isfinite(v) else "n/a")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(ctx: dict) -> None:
    """Write results/RESULTS.md from everything collected during a run."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    res, taus, test, fc = ctx["results"], ctx["taus"], ctx["test"], ctx["forecasts"]
    headline = ctx["headline"]
    order = [n for n in ("naive_week", "neso_dressed", "linear", "mlp", "seq", f"{headline}_recal", "mlp_oracle")
             if n in fc]
    LABELS[f"{headline}_recal"] = f"{LABELS.get(headline, headline)}, recalibrated on validation"

    # ---- figures ----
    ndf_q = fc.get("neso_dressed")
    ndf_med = ndf_q[:, list(taus).index(0.5)] if ndf_q is not None else None
    dates = pd.Series(test["target_date"].unique()).sort_values()
    jan = dates[(dates.dt.month == 1) & (dates.dt.dayofweek == 0)]
    week_start = jan.iloc[0] if len(jan) else dates.iloc[0]
    fan_chart(test, fc[headline], taus, ndf_med, week_start, 7, FIGURES_DIR / "fan_week.png",
              f"{LABELS.get(headline, headline)}: one January week of the test period")
    worst_day = res["daily"][headline].idxmax()
    fan_chart(test, fc[headline], taus, ndf_med, pd.Timestamp(worst_day) - pd.Timedelta(days=1), 3,
              FIGURES_DIR / "fan_worst_day.png",
              f"Around the hardest test day for {LABELS.get(headline, headline)} ({pd.Timestamp(worst_day):%d %b %Y})")
    reliability_chart({n: res["overall"][n] for n in order}, taus, FIGURES_DIR / "reliability.png")
    tod_chart({n: res["by_tod"][n] for n in order if n != "mlp_oracle"}, FIGURES_DIR / "pinball_by_time_of_day.png")
    bins_chart(res["bins"][headline], FIGURES_DIR / "quantile_bins.png",
               f"Calibration histogram: {LABELS.get(headline, headline)}")
    if ctx.get("histories"):
        training_chart(ctx["histories"], FIGURES_DIR / "training_curves.png")

    # ---- tables ----
    ov = pd.DataFrame(res["overall"]).T.loc[order]
    naive_pl = ov.loc["naive_week", "mean_pinball_mw"]
    main = pd.DataFrame({
        "Forecaster": [LABELS.get(n, n) for n in order],
        "Mean pinball (MW)": ov["mean_pinball_mw"].to_numpy(),
        "Skill vs naive": (1 - ov["mean_pinball_mw"] / naive_pl).to_numpy(),
        "MAE of median (MW)": ov["mae_mw"].to_numpy(),
        "MAPE (%)": ov["mape_pct"].to_numpy(),
        "50% cov.": ov["cov50"].to_numpy(),
        "90% cov.": ov["cov90"].to_numpy(),
        "95% cov.": ov["cov95"].to_numpy(),
        "90% width (MW)": ov["width90_mw"].to_numpy(),
    })
    fmt = {"Skill vs naive": ".1%", "MAPE (%)": ".2f", "50% cov.": ".1%", "90% cov.": ".1%", "95% cov.": ".1%"}

    val_tbl = pd.DataFrame({"Model": [LABELS.get(n, n) for n in ctx["val_scores"]],
                            "Validation mean pinball (MW)": list(ctx["val_scores"].values())})
    kup = pd.DataFrame(res["kupiec"])
    kup = kup[kup["model"].isin([headline, f"{headline}_recal", "neso_dressed"])].copy()
    kup["model"] = kup["model"].map(lambda n: LABELS.get(n, n))
    kup = kup.rename(columns={"expected_rate": "expected", "observed_rate": "observed"})
    reg = pd.DataFrame(res["regimes"])
    reg = reg[reg["model"].isin([headline, "neso_dressed", "naive_week"])].copy()
    reg["model"] = reg["model"].map(lambda n: LABELS.get(n, n))
    dm = pd.DataFrame(ctx["dm"])
    if not dm.empty:
        dm["A"] = dm["A"].map(lambda n: LABELS.get(n, n))
        dm["B"] = dm["B"].map(lambda n: LABELS.get(n, n))

    demo_banner = ("> **DEMO MODE: these numbers come from simulated data and mean nothing about the real GB "
                   "grid.** Run `python run.py` without `--demo` for real results.\n\n") if ctx["demo"] else ""
    q = ctx["data_log"]
    lines = [
        "# Results: day-ahead probabilistic forecasts of GB national demand",
        "",
        demo_banner + f"Generated {dt.datetime.now():%Y-%m-%d %H:%M} by `python run.py {ctx['argv']}`. "
        "See [docs/REPORT.md](../docs/REPORT.md) for the full methodology.",
        "",
        "## Data",
        "",
        f"- Demand history: {q['first'][:10]} to {q['last'][:10]} ({q['half_hours']:,} half-hours). "
        f"Duplicates removed: {q['duplicates_removed']}; implausible values removed: {q['implausible_values']}; "
        f"missing half-hours: {q['missing_before_fill']} before gap-filling, {q['missing_after_fill']} after.",
        f"- Rows used for modelling: {ctx['ds_info']['rows_used']:,} "
        f"(dropped {ctx['ds_info']['rows_dropped']:,} without complete inputs).",
        f"- Evaluation scheme: {ctx['scheme']}.",
        f"- Test rows scored: {len(test):,} half-hours over {test['target_date'].nunique()} days "
        f"({test['target_date'].min():%d %b %Y} to {test['target_date'].max():%d %b %Y}). "
        "Every forecaster is scored on exactly the same rows."
        + (f" {ctx['n_no_neso']:,} test half-hours with no NESO forecast published in the six hours before our "
           "issue time were excluded for every forecaster." if ctx.get("n_no_neso") else ""),
        f"- Temperature inputs: {'yes' if ctx['ds_info']['has_temperature'] else 'not available'}. "
        f"NESO benchmark: {'yes' if 'neso_dressed' in fc else 'not available'}"
        + (f" (median lead time of the official forecast: {ctx['ndf_lead']:.1f} hours before delivery; "
           f"ours: {ctx['our_lead']:.1f} hours)." if 'neso_dressed' in fc else "."),
        "",
        "## Model selection (validation data only)",
        "",
        "The headline model is chosen by validation loss, before looking at the test period.",
        "",
        _fmt_table(val_tbl),
        "",
        f"Headline model: **{LABELS.get(headline, headline)}**.",
        "",
        "## Test-period accuracy",
        "",
        _fmt_table(main, fmt),
        "",
        "Pinball loss is averaged over the quantiles " + ", ".join(f"{t:g}" for t in taus)
        + ". Lower is better. Skill = 1 - (loss / naive loss). Coverage should match the nominal level.",
        "",
        "![One week of forecasts](figures/fan_week.png)",
        "",
        "![Hardest day](figures/fan_worst_day.png)",
        "",
        "## Calibration",
        "",
        "![Reliability diagram](figures/reliability.png)",
        "",
        "![Calibration histogram](figures/quantile_bins.png)",
        "",
        "### Kupiec tests on the tails",
        "",
        "H0: tail exceedances happen at the nominal rate. Half-hourly exceedances are strongly clustered within a day, "
        "which breaks the test's independence assumption, so the 17:30-only version (one observation per day) is the "
        "more trustworthy of the two.",
        "",
        _fmt_table(kup[["model", "quantile", "sample", "n", "expected", "observed", "LR", "p_value"]],
                   {"expected": ".3f", "observed": ".3f", "LR": ".2f", "p_value": ".3f", "quantile": "g"}),
        "",
        "## Is the difference real? Diebold-Mariano tests",
        "",
        "On daily mean pinball loss, with a Newey-West variance (7 lags). A negative statistic means A is better than B.",
        "",
        _fmt_table(dm, {"mean_diff_mw": ".1f", "DM": ".2f", "p_value": ".4f"}) if not dm.empty else "_Not available._",
        "",
        "## Where the forecasts struggle",
        "",
        _fmt_table(reg, {"mean_pinball_mw": ".1f", "mae_mw": ".1f", "cov90": ".1%", "cov50": ".1%"}),
        "",
        "![Loss by time of day](figures/pinball_by_time_of_day.png)",
        "",
    ]
    if ctx.get("histories"):
        lines += ["## Training", "", "![Training curves](figures/training_curves.png)", ""]
    if "mlp_oracle" in fc:
        lines += ["## Oracle ablation", "",
                  "The oracle MLP is given the *actual* average temperature of the target day, which is not known when "
                  "the forecast is issued. Its gap to the ordinary MLP is an upper bound on what a perfect temperature "
                  "forecast would be worth. It is excluded from model selection.", ""]
    lines += ["---", "", "Contains BMRS data © Elexon Limited copyright and database right."]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    ov.to_csv(RESULTS_DIR / "metrics_overall.csv")
    reg.to_csv(RESULTS_DIR / "metrics_regimes.csv", index=False)
    with open(RESULTS_DIR / "run_info.json", "w") as f:
        json.dump({"headline": headline, "scheme": ctx["scheme"], "demo": ctx["demo"],
                   "val_scores": ctx["val_scores"]}, f, indent=2)
