"""Cleaning the raw tables and building a leakage-free modelling dataset.

The forecasting task
--------------------
At 09:00 UK time on day D-1 a forecast is issued for every settlement period
of day D (46, 48 or 50 half-hours). Every input must have been publicly
available at 09:00 on D-1. Demand outturn for a half-hour is published within
15 minutes of the half-hour ending, so the last usable observation is the
half-hour starting at 08:00 on D-1 (it ends 08:30 and is published by 08:45).

Each row of the dataset is one target half-hour. Its inputs are:

* lagged demand at the same wall-clock time 2, 7 and 14 days earlier,
* the last observed value and the mean of the last 24 hours before the cutoff,
* calendar information about the target half-hour (time of day, day of week,
  bank holidays, Christmas, season, clock-change status, lead time),
* the observed daily temperature two and three days before the target day,
* for the sequence model, the full last 7 days of half-hourly demand up to the cutoff.

``build_dataset`` asserts that no input comes from after the cutoff.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .calendar_utils import is_christmas_period, uk_holidays
from .config import LOCAL_TZ, Config

HALF_HOUR = pd.Timedelta(minutes=30)


# ----------------------------------------------------------------------------
# Cleaning
# ----------------------------------------------------------------------------
def _fill_short_gaps(s: pd.Series, max_len: int) -> pd.Series:
    """Linearly interpolate runs of missing values no longer than ``max_len``; leave longer runs missing."""
    isna = s.isna()
    run_id = (isna != isna.shift()).cumsum()
    run_len = isna.groupby(run_id).transform("sum")
    fillable = isna & (run_len <= max_len)
    filled = s.interpolate(limit_area="inside")
    return s.where(~fillable, filled)


def clean_demand(raw: pd.DataFrame, cfg: Config) -> tuple[pd.Series, dict]:
    """Return national demand (INDO, MW) on a regular UTC half-hour grid, plus a data-quality log."""
    df = raw.copy()
    df["startTime"] = pd.to_datetime(df["startTime"], utc=True)
    df["publishTime"] = pd.to_datetime(df["publishTime"], utc=True)
    n_raw = len(df)
    # if a half-hour was published more than once, keep the latest publication
    df = df.sort_values("publishTime").drop_duplicates("startTime", keep="last")
    n_dupes = n_raw - len(df)
    s = df.set_index("startTime")["initialDemandOutturn"].astype(float).sort_index()
    bad = (s <= 5000) | (s > 80000)
    s[bad] = np.nan
    grid = pd.date_range(s.index.min(), s.index.max(), freq="30min")
    s = s.reindex(grid)
    n_missing = int(s.isna().sum())
    s = _fill_short_gaps(s, cfg.max_gap_fill)
    log = {
        "first": str(grid[0]), "last": str(grid[-1]), "half_hours": len(grid),
        "duplicates_removed": int(n_dupes), "implausible_values": int(bad.sum()),
        "missing_before_fill": n_missing, "missing_after_fill": int(s.isna().sum()),
    }
    s.name = "demand"
    return s, log


def clean_ndf(*raws: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    """The official day-ahead national demand forecast *as it stood at our issue time*.

    NESO republishes its forecast through the day. For each target half-hour we
    keep the latest version published at or before 09:00 on D-1, the moment our
    own forecast is issued, so neither side sees information the other lacks.
    Versions published more than ``MAX_NDF_STALENESS`` before the issue time are
    discarded rather than used, because comparing against a stale forecast would
    flatter our models.
    """
    cfg = cfg or Config()
    frames = [r for r in raws if r is not None and not r.empty]
    if not frames:
        return pd.DataFrame(columns=["ndf", "ndf_publish"])
    df = pd.concat(frames, ignore_index=True)
    if "boundary" in df:
        df = df[df["boundary"].isna() | (df["boundary"].astype(str).str.upper() == "N")]
    df = df.dropna(subset=["nationalDemand", "startTime", "publishTime"]).copy()
    df["startTime"] = pd.to_datetime(df["startTime"], utc=True)
    df["publishTime"] = pd.to_datetime(df["publishTime"], utc=True)
    # issue time for each target half-hour: 09:00 UK time on the previous local day
    target_day = df["startTime"].dt.tz_convert(LOCAL_TZ).dt.tz_localize(None).dt.normalize()
    issue = (target_day - pd.Timedelta(days=1) + pd.Timedelta(hours=cfg.issue_hour_local)).dt.tz_localize(
        LOCAL_TZ, ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC")
    ok = (df["publishTime"] <= issue) & (df["publishTime"] >= issue - MAX_NDF_STALENESS)
    df = df[ok.to_numpy()]
    df = df.sort_values("publishTime").drop_duplicates("startTime", keep="last")
    out = df.set_index("startTime")[["nationalDemand", "publishTime"]].sort_index()
    out.columns = ["ndf", "ndf_publish"]
    out["ndf"] = out["ndf"].astype(float)
    return out


MAX_NDF_STALENESS = pd.Timedelta(hours=6)


def clean_temperature(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["temp", "temp_normal"])
    df = raw.copy()
    df["date"] = pd.to_datetime(df["measurementDate"]).dt.date
    df = df.sort_values("publishTime").drop_duplicates("date", keep="last").set_index("date")
    return pd.DataFrame({"temp": df["temperature"].astype(float),
                         "temp_normal": df["temperatureReferenceAverage"].astype(float)}).sort_index()


# ----------------------------------------------------------------------------
# Dataset construction
# ----------------------------------------------------------------------------
def _wallclock_shift(times: pd.DatetimeIndex, offsets: np.ndarray, days: int, t0: pd.Timestamp) -> np.ndarray:
    """Grid positions of the half-hour with the same UK wall-clock time ``days`` earlier.

    Lagging in UTC would compare 17:30 with 16:30 for a week after each clock
    change, so the lag is corrected by the change in UTC offset.
    """
    approx = times - pd.Timedelta(days=days)
    approx_off = approx.tz_convert(LOCAL_TZ).map(lambda x: x.utcoffset()).to_numpy()
    lag = approx + (offsets - approx_off)
    return ((lag - t0) // HALF_HOUR).to_numpy().astype(np.int64)


DOW_NAMES = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def build_dataset(demand: pd.Series, temp: pd.DataFrame, ndf: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict]:
    """One row per target half-hour, with features, target, cutoff position and split label."""
    times = demand.index
    t0 = times[0]
    values = demand.to_numpy()
    local = times.tz_convert(LOCAL_TZ)
    offsets = local.map(lambda x: x.utcoffset()).to_numpy()

    df = pd.DataFrame(index=times)
    df["pos"] = np.arange(len(times))
    df["y"] = values
    df["target_date"] = pd.to_datetime(local.date)

    # forecast issue time and information cutoff for each target day
    dates = df["target_date"].drop_duplicates()
    issue_local = (dates - pd.Timedelta(days=1) + pd.Timedelta(hours=cfg.issue_hour_local)).dt.tz_localize(LOCAL_TZ)
    issue_utc = issue_local.dt.tz_convert("UTC")
    last_start = (issue_utc - HALF_HOUR - pd.Timedelta(minutes=cfg.publication_lag_min)).dt.floor("30min")
    cutoff_map = pd.Series(((last_start - t0) // HALF_HOUR).to_numpy(), index=dates.to_numpy())
    issue_map = pd.Series(issue_utc.to_numpy(), index=dates.to_numpy())
    df["cutoff_pos"] = df["target_date"].map(cutoff_map).astype(np.int64)
    df["issue_time"] = pd.to_datetime(df["target_date"].map(issue_map), utc=True)

    # lagged demand at the same wall-clock time
    lag_pos = {}
    for d in (2, 7, 14):
        p = _wallclock_shift(times, offsets, d, t0)
        lag_pos[d] = p
        ok = (p >= 0) & (p < len(values))
        lagged = np.full(len(values), np.nan)
        lagged[ok] = values[p[ok]]
        df[f"y_lag{d}d"] = lagged

    cp = df["cutoff_pos"].to_numpy()
    valid_cp = (cp >= 47) & (cp < len(values))
    last_obs = np.full(len(values), np.nan)
    last_obs[valid_cp] = values[cp[valid_cp]]
    df["y_last"] = last_obs
    roll = pd.Series(values).rolling(48, min_periods=48).mean().to_numpy()
    mean24 = np.full(len(values), np.nan)
    mean24[valid_cp] = roll[cp[valid_cp]]
    df["y_mean24h"] = mean24

    # the leakage guard: every lagged input must sit at or before the information cutoff
    for d, p in lag_pos.items():
        assert np.all(p <= cp), f"lag {d}d uses data after the forecast cutoff"

    # calendar features of the target half-hour (UK local time)
    hour = local.hour + local.minute / 60
    for k in (1, 2, 3):
        df[f"tod_sin{k}"] = np.sin(2 * np.pi * k * hour / 24)
        df[f"tod_cos{k}"] = np.cos(2 * np.pi * k * hour / 24)
    dow = local.dayofweek
    for i, n in enumerate(DOW_NAMES):
        df[f"dow_{n}"] = (dow == i).astype(float)
    hol = uk_holidays(range(times[0].year - 1, times[-1].year + 2))
    td = df["target_date"]
    df["holiday"] = td.dt.date.isin(hol).astype(float)
    df["christmas"] = is_christmas_period(td).astype(float)
    for d in (7, 14):
        df[f"holiday_lag{d}d"] = (td - pd.Timedelta(days=d)).dt.date.isin(hol).astype(float)
    doy = local.dayofyear
    for k in (1, 2):
        df[f"doy_sin{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        df[f"doy_cos{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    df["dst"] = (offsets > pd.Timedelta(0)).astype(float)
    df["trend"] = (local.year - 2018) + doy / 365.25
    df["lead_h"] = (df.index - df["issue_time"]).dt.total_seconds().to_numpy() / 3600
    n_periods = df.groupby("target_date")["pos"].transform("size")
    df["periods_in_day"] = n_periods

    # temperature (observed on D-2 and D-3: both published before the cutoff).
    # Elexon's temperature history starts in April 2021. Rather than discard the earlier
    # demand history, missing temperatures are replaced by the seasonal average for that
    # day of year (estimated from training-period data only) and flagged with temp_missing,
    # so the model can learn to rely on temperature only when it is real.
    has_temp = temp is not None and not temp.empty
    if has_temp:
        tser = temp["temp"].copy()
        tser.index = pd.to_datetime(tser.index)
        tnorm = temp["temp_normal"].copy()
        tnorm.index = pd.to_datetime(tnorm.index)
        fit = tser[tser.index <= pd.Timestamp(cfg.train_end)]
        if len(fit) < 300:
            fit = tser
        doy_all = np.arange(1, 367)
        clim_raw = fit.groupby(fit.index.dayofyear).mean().reindex(doy_all)
        # smooth with a circular 31-day window so sparse days of the year don't make the climatology noisy
        wrapped = pd.concat([clim_raw.iloc[-15:], clim_raw, clim_raw.iloc[:15]])
        clim = wrapped.rolling(31, center=True, min_periods=5).mean().iloc[15:-15]
        clim = clim.interpolate(limit_direction="both")
        clim.index = doy_all

        def temp_on(dates: pd.Series):
            vals = dates.map(tser)
            filled = vals.fillna(pd.Series(clim.reindex(dates.dt.dayofyear).to_numpy(), index=dates.index))
            return vals.to_numpy(), filled.to_numpy()

        raw_d2, df["temp_d2"] = temp_on(td - pd.Timedelta(days=2))
        _, df["temp_d3"] = temp_on(td - pd.Timedelta(days=3))
        df["temp_missing"] = np.isnan(raw_d2).astype(float)
        norm_d2 = (td - pd.Timedelta(days=2)).map(tnorm).to_numpy()
        df["temp_anom_d2"] = np.where(np.isnan(raw_d2) | np.isnan(norm_d2), 0.0, raw_d2 - norm_d2)
        # NOT available at issue time: used only by the clearly labelled "oracle" ablation
        _, df["temp_target_oracle"] = temp_on(td)

    # official day-ahead forecast (benchmark only; never a model input)
    if ndf is not None and not ndf.empty:
        df = df.join(ndf, how="left")
    else:
        df["ndf"] = np.nan
        df["ndf_publish"] = pd.NaT

    # chronological split by target date
    split = np.full(len(df), "", dtype=object)
    split[(td <= cfg.train_end).to_numpy()] = "train"
    split[((td >= cfg.val_start) & (td <= cfg.val_end)).to_numpy()] = "val"
    split[(td >= cfg.test_start).to_numpy()] = "test"
    df["split"] = split

    # drop rows without a complete history window or with missing inputs
    feature_cols = feature_columns(df, oracle=False)
    hist_ok = np.zeros(len(df), dtype=bool)
    nan_csum = np.concatenate([[0], np.cumsum(np.isnan(values))])
    start = cp - cfg.history_len + 1
    good = (start >= 0) & (cp < len(values))
    hist_ok[good] = (nan_csum[cp[good] + 1] - nan_csum[start[good]]) == 0
    need = df[feature_cols + ["y"]].notna().all(axis=1).to_numpy() & hist_ok & (df["split"] != "").to_numpy()
    info = {"rows_total": len(df), "rows_used": int(need.sum()), "rows_dropped": int((~need).sum()),
            "has_temperature": has_temp, "has_ndf": bool(df["ndf"].notna().any())}
    df = df[need].copy()
    df.index.name = "time_utc"
    return df, info


BASE_FEATURES = (["y_lag2d", "y_lag7d", "y_lag14d", "y_last", "y_mean24h"]
                 + [f"tod_{f}{k}" for k in (1, 2, 3) for f in ("sin", "cos")]
                 + [f"dow_{n}" for n in DOW_NAMES]
                 + ["holiday", "christmas", "holiday_lag7d", "holiday_lag14d"]
                 + [f"doy_{f}{k}" for k in (1, 2) for f in ("sin", "cos")]
                 + ["dst", "trend", "lead_h"])
TEMP_FEATURES = ["temp_d2", "temp_d3", "temp_anom_d2", "temp_missing"]


DEMAND_FEATURES = ["y_lag2d", "y_lag7d", "y_lag14d", "y_last", "y_mean24h"]


def feature_columns(df: pd.DataFrame, oracle: bool, use_trend: bool = True) -> list[str]:
    cols = [c for c in BASE_FEATURES if use_trend or c != "trend"]
    if "temp_d2" in df:
        cols += TEMP_FEATURES
        if oracle:
            cols.append("temp_target_oracle")
    return cols
