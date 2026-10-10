"""Checks that the pipeline is correct where mistakes would be silent.

Run with:  python -m pytest -q
"""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from gridforecast import data, elexon, evaluate, synthetic
from gridforecast.calendar_utils import local_day_periods
from gridforecast.config import Config
from gridforecast.models import MLPQR, SeqQR, pinball_loss
from gridforecast.train import Prepared


@pytest.fixture(scope="module")
def built():
    cfg = Config()
    t = synthetic.simulate("2021-01-01", "2025-06-30", seed=1)
    demand, _ = data.clean_demand(t["demand"], cfg)
    ds, _ = data.build_dataset(demand, data.clean_temperature(t["temperature"]), data.clean_ndf(t["ndf"], t["ndf_update"]), cfg)
    return cfg, t, demand, ds


def test_clock_change_days_have_46_and_50_periods():
    assert len(local_day_periods(pd.Timestamp("2025-03-30").date())) == 46
    assert len(local_day_periods(pd.Timestamp("2025-10-26").date())) == 50
    assert len(local_day_periods(pd.Timestamp("2025-06-01").date())) == 48


def test_cutoff_is_0800_local_on_previous_day(built):
    cfg, _, demand, ds = built
    for day in ["2024-01-15", "2024-07-15", "2024-03-31", "2024-10-27"]:
        rows = ds[ds["target_date"] == day]
        cutoff_time = demand.index[rows["cutoff_pos"].iloc[0]].tz_convert("Europe/London")
        assert cutoff_time == pd.Timestamp(day, tz="Europe/London") - pd.Timedelta(hours=16)  # D-1 08:00


def test_no_future_information_in_features(built):
    """Changing every demand value after the cutoff must not change any input for that target day."""
    cfg, t, demand, ds = built
    day = pd.Timestamp("2024-02-14")
    cut = ds.loc[ds["target_date"] == day, "cutoff_pos"].iloc[0]
    tampered = demand.copy()
    tampered.iloc[cut + 1:] = tampered.iloc[cut + 1:] * 3 + 5000
    ds2, _ = data.build_dataset(tampered, data.clean_temperature(t["temperature"]), data.clean_ndf(t["ndf"], t["ndf_update"]), cfg)
    cols = data.feature_columns(ds, oracle=False)
    a = ds.loc[ds["target_date"] == day, cols]
    b = ds2.loc[ds2["target_date"] == day, cols]
    pd.testing.assert_frame_equal(a, b)
    # and the sequence model's history window is unchanged too
    mask = (ds["split"] == "train").to_numpy()
    p1 = Prepared(ds, demand.to_numpy(), cols, mask, cfg)
    p2 = Prepared(ds2, tampered.to_numpy(), cols, mask, cfg)
    idx = torch.as_tensor(np.flatnonzero((ds["target_date"] == day).to_numpy()))
    w1, _ = p1.history(idx)
    w2, _ = p2.history(idx)
    assert torch.allclose(w1 * p1.y_std + p1.y_mean, w2 * p2.y_std + p2.y_mean)


def test_wallclock_lag_across_clock_change(built):
    _, _, demand, ds = built
    row = ds.loc[pd.Timestamp("2024-04-02 16:30", tz="UTC")]           # 17:30 BST
    assert row["y_lag7d"] == demand.loc[pd.Timestamp("2024-03-26 17:30", tz="UTC")]   # 17:30 GMT


def test_latest_publication_wins_and_gaps_filled(built):
    cfg, t, demand, _ = built
    assert (demand > 5000).all()             # superseded zero values were discarded
    assert demand.isna().sum() == 0          # short gaps interpolated


def test_long_gaps_are_not_filled():
    s = pd.Series([1.0, np.nan, np.nan, 4.0, np.nan, np.nan, np.nan, np.nan, 9.0])
    out = data._fill_short_gaps(s, 2)
    assert out.iloc[1] == 2.0 and out.iloc[2] == 3.0
    assert out.iloc[4:8].isna().all()


def test_pinball_loss_values():
    taus = torch.tensor([0.1, 0.5, 0.9])
    pred = torch.zeros(1, 3)
    assert torch.isclose(pinball_loss(pred, torch.tensor([10.0]), taus), torch.tensor((1 + 5 + 9) / 3))
    assert torch.isclose(pinball_loss(pred, torch.tensor([-10.0]), taus), torch.tensor((9 + 5 + 1) / 3))


def test_quantiles_never_cross():
    torch.manual_seed(0)
    m = MLPQR(10, 9).eval()
    q = m(torch.randn(500, 10) * 10)
    assert (q.diff(dim=1) > 0).all()
    s = SeqQR(10, 9).eval()
    q = s(torch.randn(4, 10), (torch.randn(2, 336), torch.tensor([0, 1, 1, 0])))
    assert (q.diff(dim=1) > 0).all()


def test_kupiec_and_dm():
    rng = np.random.default_rng(0)
    _, p = evaluate.kupiec(rng.random(5000) < 0.05, 0.05)
    assert p > 0.01
    _, p = evaluate.kupiec(rng.random(5000) < 0.10, 0.05)
    assert p < 1e-6
    a = rng.normal(0, 1, 400)
    _, p = evaluate.diebold_mariano(a, a + rng.normal(0, 0.01, 400))
    assert p > 0.01
    _, p = evaluate.diebold_mariano(a, a + 0.5)
    assert p < 1e-6


def test_elexon_chunk_halving_and_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(elexon, "RAW_DIR", tmp_path)
    calls = []

    class Resp:
        def __init__(self, status, body):
            self.status_code, self._body, self.text = status, body, json.dumps(body)

        def json(self):
            return self._body

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(self.status_code)

    def fake_get(self, url, params=None, **kw):
        a, b = pd.Timestamp(params["settlementDateFrom"]), pd.Timestamp(params["settlementDateTo"])
        calls.append((a, b))
        if (b - a).days + 1 > 7:
            return Resp(400, {"error": "The date range must not exceed 7 days"})
        return Resp(200, {"data": [{"settlementDate": str(d.date())} for d in pd.date_range(a, b)]})

    monkeypatch.setattr(elexon.requests.Session, "get", fake_get)
    monkeypatch.setattr(elexon.time, "sleep", lambda s: None)
    df = elexon.download("demand", "2020-01-01", "2020-02-28", verbose=False)
    assert len(df) == 59 and df["settlementDate"].is_unique
    n = len(calls)
    df2 = elexon.download("demand", "2020-01-01", "2020-02-28", verbose=False)
    assert len(calls) == n and len(df2) == 59        # second run served entirely from cache


def test_benchmark_uses_forecast_published_by_issue_time(built):
    """NESO's forecast must be the latest version published at or before 09:00 on D-1."""
    _, t, _, ds = built
    day = ds[ds["target_date"] == "2024-05-14"]
    pub_local = pd.to_datetime(day["ndf_publish"], utc=True).dt.tz_convert("Europe/London")
    assert (pub_local == pd.Timestamp("2024-05-13 08:45", tz="Europe/London")).all()
    assert day["ndf"].notna().all()
    # without the revisions, the first ten half-hours would only have a stale version, which is discarded
    stale_only = data.clean_ndf(t["ndf"])
    first = pd.Timestamp("2024-05-14 00:00", tz="Europe/London").tz_convert("UTC")
    assert first not in stale_only.index
