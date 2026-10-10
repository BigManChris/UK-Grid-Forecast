"""Benchmarks and forecast evaluation.

Benchmarks
----------
* ``naive_week``  demand at the same wall-clock time one week earlier, with
                  quantiles from its empirical errors on the training years.
* ``neso_dressed`` the official earliest day-ahead national demand forecast
                  (published by NESO, distributed by Elexon), turned into a
                  probabilistic forecast by adding the empirical quantiles of
                  its errors in the validation year ("error dressing").

Metrics
-------
pinball loss per quantile and averaged, MAE/RMSE/MAPE of the median, central
interval coverage and width, quantile hit rates (reliability), Kupiec
proportion-of-failures tests on the tails, and a Diebold-Mariano test for
whether two forecasters' losses really differ.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import INTERVALS


# ----------------------------------------------------------------------------
# Benchmarks
# ----------------------------------------------------------------------------
def dressed_point_forecast(point: np.ndarray, actual: np.ndarray, calib_mask: np.ndarray,
                           taus) -> np.ndarray:
    """Point forecast + empirical error quantiles estimated on ``calib_mask`` rows."""
    ok = calib_mask & np.isfinite(point) & np.isfinite(actual)
    if ok.sum() < 100:
        raise ValueError("not enough calibration rows")
    err_q = np.quantile(actual[ok] - point[ok], taus)
    return point[:, None] + err_q[None, :]


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
def pinball(y: np.ndarray, Q: np.ndarray, taus) -> np.ndarray:
    """Pinball loss for every row and quantile, shape (n, K), in MW."""
    t = np.asarray(taus)[None, :]
    u = y[:, None] - Q
    return np.maximum(t * u, (t - 1) * u)


def kupiec(hits: np.ndarray, p: float) -> tuple[float, float]:
    """Kupiec (1995) proportion-of-failures likelihood-ratio test.

    H0: the probability of an exceedance equals p. Returns (LR statistic, p-value),
    with LR ~ chi-squared(1) under H0. Assumes exceedances are independent.
    """
    n, x = len(hits), int(hits.sum())
    if n == 0:
        return float("nan"), float("nan")
    phat = x / n

    def ll(q):
        q = min(max(q, 1e-12), 1 - 1e-12)
        return (n - x) * math.log(1 - q) + x * math.log(q)

    lr = -2 * (ll(p) - ll(phat))
    pval = math.erfc(math.sqrt(max(lr, 0) / 2))   # chi2(1) survival function
    return lr, pval


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, lag: int = 7) -> tuple[float, float]:
    """Diebold-Mariano test on two aligned loss series (here: daily mean pinball).

    d_t = loss_a - loss_b. H0: E[d_t] = 0. The variance of the mean uses a
    Newey-West estimator so autocorrelated daily losses don't overstate significance.
    Returns (DM statistic, two-sided p-value); negative DM means A is better.
    """
    d = np.asarray(loss_a) - np.asarray(loss_b)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return float("nan"), float("nan")
    dc = d - d.mean()
    var = dc @ dc / n
    for k in range(1, min(lag, n - 1) + 1):
        w = 1 - k / (lag + 1)
        var += 2 * w * (dc[k:] @ dc[:-k]) / n
    stat = d.mean() / math.sqrt(max(var, 1e-12) / n)
    pval = math.erfc(abs(stat) / math.sqrt(2))
    return stat, pval


def summarise(y: np.ndarray, Q: np.ndarray, taus) -> dict:
    taus = list(taus)
    pl = pinball(y, Q, taus)
    med = Q[:, taus.index(0.5)]
    err = y - med
    out = {
        "n": len(y),
        "mean_pinball_mw": float(pl.mean()),
        "mae_mw": float(np.abs(err).mean()),
        "rmse_mw": float(np.sqrt((err ** 2).mean())),
        "mape_pct": float(np.mean(np.abs(err) / y) * 100),
        "bias_mw": float(err.mean()),
    }
    for level, (lo, hi) in INTERVALS.items():
        a, b = Q[:, taus.index(lo)], Q[:, taus.index(hi)]
        out[f"cov{level}"] = float(np.mean((y >= a) & (y <= b)))
        out[f"width{level}_mw"] = float(np.mean(b - a))
    for t in taus:
        out[f"hit_{t}"] = float(np.mean(y < Q[:, taus.index(t)]))
    return out


def quantile_bin_histogram(y: np.ndarray, Q: np.ndarray, taus) -> pd.DataFrame:
    """Share of outcomes falling between consecutive forecast quantiles vs the share expected.

    This is the discrete version of a PIT histogram: if the forecasts are
    calibrated, observed and expected shares match in every bin.
    """
    edges = [0.0] + list(taus) + [1.0]
    pos = (y[:, None] >= Q).sum(axis=1)       # 0 = below the lowest quantile, K = above the highest
    counts = np.bincount(pos, minlength=len(taus) + 1) / len(y)
    return pd.DataFrame({"bin": [f"{edges[i]:g}-{edges[i + 1]:g}" for i in range(len(edges) - 1)],
                         "expected": np.diff(edges), "observed": counts})


def evaluate_all(test: pd.DataFrame, forecasts: dict[str, np.ndarray], taus) -> dict:
    """All metrics for every forecaster on the same test rows."""
    taus = list(taus)
    y = test["y"].to_numpy()
    res = {"overall": {}, "kupiec": [], "dm": [], "regimes": [], "by_tod": {}, "bins": {}, "daily": {}}
    local = test.index.tz_convert("Europe/London")
    tod = local.hour + local.minute / 60
    day = test["target_date"].to_numpy()

    for name, Q in forecasts.items():
        res["overall"][name] = summarise(y, Q, taus)
        res["bins"][name] = quantile_bin_histogram(y, Q, taus)
        pl = pinball(y, Q, taus).mean(axis=1)
        res["by_tod"][name] = pd.Series(pl).groupby(tod).mean()
        res["daily"][name] = pd.Series(pl).groupby(day).mean()
        # Kupiec on every half-hour, and on one half-hour per day (17:30, near the evening peak)
        peak = (local.hour == 17) & (local.minute == 30)
        for t in (0.025, 0.05, 0.95, 0.975):
            if t not in taus:
                continue
            q = Q[:, taus.index(t)]
            hits = (y > q) if t > 0.5 else (y < q)
            p = 1 - t if t > 0.5 else t
            for label, m in (("all half-hours", np.ones(len(y), bool)), ("17:30 only", peak)):
                lr, pv = kupiec(hits[m], p)
                res["kupiec"].append({"model": name, "quantile": t, "sample": label, "n": int(m.sum()),
                                      "expected_rate": p, "observed_rate": float(hits[m].mean()),
                                      "LR": lr, "p_value": pv})

    # regime breakdown
    dow = local.dayofweek
    regimes = {
        "weekday (non-holiday)": (dow < 5) & (test["holiday"] == 0) & (test["christmas"] == 0),
        "weekend (non-holiday)": (dow >= 5) & (test["holiday"] == 0) & (test["christmas"] == 0),
        "bank holiday or Christmas": (test["holiday"] == 1) | (test["christmas"] == 1),
        "clock-change day": test["periods_in_day"] != 48,
        "winter (Dec-Feb)": local.month.isin([12, 1, 2]),
        "summer (Jun-Aug)": local.month.isin([6, 7, 8]),
    }
    if "temp_target_oracle" in test:
        t_day = test.groupby("target_date")["temp_target_oracle"].first()
        cold_days = t_day[t_day <= t_day.quantile(0.10)].index
        regimes["coldest 10% of days"] = test["target_date"].isin(cold_days)
    for rname, m in regimes.items():
        m = np.asarray(m)
        if m.sum() == 0:
            continue
        for name, Q in forecasts.items():
            s = summarise(y[m], Q[m], taus)
            res["regimes"].append({"regime": rname, "model": name, "n": int(m.sum()),
                                   "mean_pinball_mw": s["mean_pinball_mw"], "mae_mw": s["mae_mw"],
                                   "cov90": s["cov90"], "cov50": s["cov50"]})
    return res


def dm_table(daily: dict[str, pd.Series], pairs: list[tuple[str, str]]) -> list[dict]:
    rows = []
    for a, b in pairs:
        if a in daily and b in daily:
            da, db = daily[a].align(daily[b], join="inner")
            stat, pv = diebold_mariano(da.to_numpy(), db.to_numpy())
            rows.append({"A": a, "B": b, "days": len(da), "mean_diff_mw": float((da - db).mean()),
                         "DM": stat, "p_value": pv})
    return rows
