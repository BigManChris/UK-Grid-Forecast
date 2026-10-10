"""Realistic simulated data in exactly the same format as the Elexon API.

Used by ``python run.py --demo`` (and the tests) so the whole pipeline can be
run and checked without a network connection. The numbers are made up: never
quote results from demo mode as findings.

The simulator includes the features that make GB demand forecasting
non-trivial: a daily profile with morning and evening peaks, lower weekend
and bank-holiday demand, heating demand on cold days, rooftop solar
suppressing midday demand in summer, a slow downward trend, persistent
(autocorrelated) noise, and the 46- and 50-period days at the clock changes.
"""

from __future__ import annotations


import numpy as np
import pandas as pd

from .calendar_utils import local_day_periods, uk_holidays


def simulate(start: str = "2019-01-01", end: str = "2026-09-30", seed: int = 0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, end, freq="D")
    hol = uk_holidays(range(days[0].year - 1, days[-1].year + 2))

    # daily temperature: seasonal cycle + AR(1) weather anomalies
    doy = days.dayofyear.to_numpy()
    anomaly = np.zeros(len(days))
    for i in range(1, len(days)):
        anomaly[i] = 0.8 * anomaly[i - 1] + rng.normal(0, 1.6)
    temp_normal = 10.5 - 6.0 * np.cos(2 * np.pi * (doy - 20) / 365.25)
    temp = temp_normal + anomaly
    temperature = pd.DataFrame({
        "measurementDate": days.strftime("%Y-%m-%d"),
        "publishTime": (days + pd.Timedelta(days=1)).strftime("%Y-%m-%dT06:00:00Z"),
        "temperature": temp.round(1),
        "temperatureReferenceAverage": temp_normal.round(1),
        "temperatureReferenceHigh": (temp_normal + 3).round(1),
        "temperatureReferenceLow": (temp_normal - 3).round(1),
    })

    demand_rows, ndf_rows, update_rows = [], [], []
    noise = 0.0
    fc_err = 0.0
    years0 = days[0].year
    for k, day in enumerate(days):
        d = day.date()
        starts_utc = local_day_periods(d)              # UTC start times of this settlement day
        local = starts_utc.tz_convert("Europe/London")
        hours = (local.hour + local.minute / 60).to_numpy()
        is_weekend = d.weekday() >= 5
        is_hol = d in hol or (d.month == 12 and d.day >= 24) or (d.month == 1 and d.day <= 1)
        season = 1 + 0.12 * np.cos(2 * np.pi * (doy[k] - 15) / 365.25)
        trend = 1 - 0.012 * (d.year - years0 + doy[k] / 365.25)

        profile = (0.72 + 0.20 * np.exp(-((hours - 8.5) / 2.5) ** 2) + 0.12 * np.exp(-((hours - 13) / 4) ** 2)
                   + 0.30 * np.exp(-((hours - 17.75) / 1.6) ** 2) - 0.15 * np.exp(-((hours - 3.5) / 2.2) ** 2))
        if is_weekend or is_hol:
            profile = profile - 0.09 * np.exp(-((hours - 9) / 3) ** 2) - 0.05
        heating = 0.012 * max(15.5 - temp[k], 0) * (1 + 0.5 * np.exp(-((hours - 18) / 3) ** 2))
        solar_strength = max(0.0, np.sin(np.pi * (doy[k] - 80) / 365.25 * 2) * 0.5 + 0.5)
        cloud = rng.uniform(0.3, 1.0)
        solar = 0.10 * solar_strength * cloud * np.clip(np.cos((hours - 13) / 4.5), 0, None) ** 2

        base = 31000 * trend * season * (profile + heating - solar)
        n = len(starts_utc)
        y = np.empty(n)
        for j in range(n):
            noise = 0.97 * noise + rng.normal(0, 140)
            y[j] = base[j] + noise
        # official-style forecast: actual + smooth error, larger when temperature anomalies are big
        err = np.empty(n)
        for j in range(n):
            fc_err = 0.9 * fc_err + rng.normal(0, 200 + 45 * abs(anomaly[k]))
            err[j] = fc_err

        # like the real data: the forecast is published at 08:45 on D-1, but for the first ten
        # half-hours the *earliest* version dates from 08:45 on D-2 and is revised at 08:45 on D-1
        pub_d1 = (starts_utc[0] - pd.Timedelta(hours=15, minutes=15)).strftime("%Y-%m-%dT%H:%M:%SZ")
        pub_d2 = (starts_utc[0] - pd.Timedelta(hours=39, minutes=15)).strftime("%Y-%m-%dT%H:%M:%SZ")
        for j in range(n):
            st = starts_utc[j].strftime("%Y-%m-%dT%H:%M:%SZ")
            pub = (starts_utc[j] + pd.Timedelta(minutes=45)).strftime("%Y-%m-%dT%H:%M:%SZ")
            demand_rows.append((pub, st, d.isoformat(), j + 1, round(y[j]), round(y[j] * 1.04 + 900)))
            row = (round(y[j] * 1.04 + 900 + err[j]), round(y[j] + err[j]), st, d.isoformat(), j + 1, "N")
            if j < 10:
                stale = 3 * err[j]   # the two-day-old version is less accurate
                ndf_rows.append((pub_d2, row[0] + round(stale), row[1] + round(stale)) + row[2:])
                update_rows.append((pub_d2, row[0] + round(stale), row[1] + round(stale)) + row[2:])
                update_rows.append((pub_d1,) + row)
            else:
                ndf_rows.append((pub_d1,) + row)

    demand = pd.DataFrame(demand_rows, columns=["publishTime", "startTime", "settlementDate", "settlementPeriod",
                                                "initialDemandOutturn", "initialTransmissionSystemDemandOutturn"])
    # simulate a few data-quality problems the cleaning step must handle
    drop = rng.choice(len(demand), size=40, replace=False)
    demand = demand.drop(index=drop)
    dup = demand.sample(25, random_state=seed).copy()
    dup["publishTime"] = "2000-01-01T00:00:00Z"   # an older, superseded publication
    dup["initialDemandOutturn"] = 0
    demand = pd.concat([demand, dup], ignore_index=True)

    cols = ["publishTime", "transmissionSystemDemand", "nationalDemand", "startTime",
            "settlementDate", "settlementPeriod", "boundary"]
    ndf = pd.DataFrame(ndf_rows, columns=cols)
    ndf = ndf[ndf["settlementDate"] >= "2021-04-20"].reset_index(drop=True)   # history starts in 2021, as in reality
    update = pd.DataFrame(update_rows, columns=cols)
    update = update[update["settlementDate"] >= "2021-04-20"].reset_index(drop=True)
    return {"demand": demand, "ndf": ndf, "ndf_update": update, "temperature": temperature}
