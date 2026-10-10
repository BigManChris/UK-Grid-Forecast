"""Downloading from Elexon's Insights Solution API (free, no key needed).

Data source: https://data.elexon.co.uk/bmrs/api/v1
Contains BMRS data (c) Elexon Limited copyright and database right.

Four datasets are used:

* ``/demand/outturn``                      INDO: half-hourly initial national demand outturn (MW)
* ``/forecast/demand/day-ahead/earliest``  NDF: the earliest day-ahead national demand
                                           forecast published for each half-hour (the benchmark)
* ``/forecast/demand/day-ahead/evolution`` every published version of the NDF for the first
                                           14 half-hours of each day. For 00:00-06:30 the
                                           "earliest" version is published two days ahead, so
                                           these later versions are needed to find the forecast
                                           NESO had published by our 09:00 issue time.
* ``/temperature``                         daily average GB temperature from six weather stations

Requests are split into chunks and every completed chunk is cached under
``data/raw/<dataset>/`` so re-running never downloads the same history twice.
If the API rejects a chunk as too large, the chunk length is halved (and remembered) and the request retried.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import pandas as pd
import requests

from .config import RAW_DIR

BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"
HEADERS = {"User-Agent": "Grid-Quantile-Forecast/0.1 (student research project)",
           "Accept": "application/json"}

# (endpoint, parameter style, default chunk length in days)
DATASETS = {
    "demand": ("/demand/outturn", "settlement_dates", 28),
    "ndf": ("/forecast/demand/day-ahead/earliest", "datetimes", 14),
    "ndf_update": ("/forecast/demand/day-ahead/evolution", "evolution", 14),
    "temperature": ("/temperature", "dates", 366),
}
NDF_FIRST_DATE = dt.date(2021, 4, 20)   # the day-ahead forecast history in the API starts in April 2021
EVOLUTION_PERIODS = list(range(1, 15))  # settlement periods 1-14 cover 00:00-06:30 even on clock-change days


class TooLarge(Exception):
    """The API refused the request because the date range was too long."""


def _params(style: str, start: dt.date, end: dt.date) -> dict:
    if style == "settlement_dates":
        return {"settlementDateFrom": start.isoformat(), "settlementDateTo": end.isoformat()}
    if style == "dates":
        return {"from": start.isoformat(), "to": end.isoformat()}
    # date-times: [start 00:00Z, end+1 00:00Z) widened by a day so local days are covered
    return {"from": f"{(start - dt.timedelta(days=1)).isoformat()}T00:00Z",
            "to": f"{(end + dt.timedelta(days=1)).isoformat()}T23:30Z"}


def _get(endpoint: str, params: dict, session: requests.Session) -> list[dict]:
    params = {**params, "format": "json"}
    for attempt in range(5):
        try:
            r = session.get(BASE_URL + endpoint, params=params, headers=HEADERS, timeout=60)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 400:
            # most often the date range is too long; the caller halves the range and retries,
            # and re-raises if even a single day is rejected
            raise TooLarge(r.text[:300])
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        body = r.json()
        if isinstance(body, dict):
            body = body.get("data", [])
        return body
    raise RuntimeError(f"Elexon API kept failing for {endpoint} {params}")


def _get_evolution(endpoint: str, a: dt.date, b: dt.date, session: requests.Session) -> list[dict]:
    """One request per settlement day (the evolution endpoint takes a single date)."""
    rows: list[dict] = []
    for i in range((b - a).days + 1):
        d = a + dt.timedelta(days=i)
        try:
            rows += _get(endpoint, {"settlementDate": d.isoformat(), "settlementPeriod": EVOLUTION_PERIODS},
                         session)
        except TooLarge:
            pass   # no forecast stored for that day
        time.sleep(0.1)
    return rows


def _chunks(start: dt.date, end: dt.date, days: int):
    cur = start
    while cur <= end:
        stop = min(cur + dt.timedelta(days=days - 1), end)
        yield cur, stop
        cur = stop + dt.timedelta(days=1)


def download(name: str, start: str, end: str | None = None, verbose: bool = True) -> pd.DataFrame:
    """Download (or load from cache) one dataset between two dates, inclusive."""
    endpoint, style, chunk_days = DATASETS[name]
    start_d = dt.date.fromisoformat(start)
    if style == "evolution":
        start_d = max(start_d, NDF_FIRST_DATE)
    end_d = dt.date.fromisoformat(end) if end else dt.date.today() - dt.timedelta(days=2)
    cache_dir = RAW_DIR / name
    cache_dir.mkdir(parents=True, exist_ok=True)
    recent = dt.date.today() - dt.timedelta(days=5)   # don't cache chunks that may still be revised

    # the largest chunk the API accepted last time is remembered, so cache keys line up between runs
    size_file = cache_dir / "_chunk_days.txt"
    if size_file.exists():
        chunk_days = int(size_file.read_text())

    session = requests.Session()
    frames: list[pd.DataFrame] = []
    todo = list(_chunks(start_d, end_d, chunk_days))
    n_done = 0
    while todo:
        a, b = todo.pop(0)
        path = cache_dir / f"{a}_{b}.json"
        if path.exists():
            rows = json.loads(path.read_text())
        else:
            try:
                if style == "evolution":
                    rows = _get_evolution(endpoint, a, b, session)
                else:
                    rows = _get(endpoint, _params(style, a, b), session)
            except TooLarge as e:
                if a == b:
                    raise RuntimeError(f"Elexon rejected the request for {name} on {a}: {e}")
                # halve this chunk and every remaining one, and remember the smaller size
                chunk_days = max(1, ((b - a).days + 1) // 2)
                size_file.write_text(str(chunk_days))
                rest_start = a
                todo = list(_chunks(rest_start, end_d, chunk_days))
                continue
            if b < recent:
                path.write_text(json.dumps(rows))
            time.sleep(0.2)   # be polite to a free public API
        if rows:
            frames.append(pd.DataFrame(rows))
        n_done += 1
        if verbose and (n_done % 10 == 0 or not todo):
            print(f"  {name}: {n_done} chunks done, up to {b}", flush=True)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def download_all(start: str, end: str | None = None) -> dict[str, pd.DataFrame]:
    out = {}
    for name in DATASETS:
        print(f"Downloading {name} ...", flush=True)
        try:
            out[name] = download(name, start, end)
        except Exception as e:  # the benchmark and temperature are optional; demand is not
            if name == "demand":
                raise
            print(f"  WARNING: could not download {name} ({e}); continuing without it")
            out[name] = pd.DataFrame()
        print(f"  {name}: {len(out[name]):,} rows")
    return out


def save_raw_tables(tables: dict[str, pd.DataFrame], folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(folder / f"{name}.csv.gz", index=False)


def load_raw_tables(folder: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for name in DATASETS:
        p = folder / f"{name}.csv.gz"
        out[name] = pd.read_csv(p) if p.exists() and p.stat().st_size > 30 else pd.DataFrame()
    return out
