"""Central settings for the whole pipeline.

Everything that defines the experiment (dates, quantiles, the forecast issue
time, model sizes) lives here, so a run is fully described by this file plus
the command-line flags recorded in the results report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

LOCAL_TZ = "Europe/London"

# Quantile levels the models predict. Symmetric pairs give central intervals:
# (0.025, 0.975) -> 95%, (0.05, 0.95) -> 90%, (0.1, 0.9) -> 80%, (0.25, 0.75) -> 50%.
QUANTILES = (0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975)
INTERVALS = {95: (0.025, 0.975), 90: (0.05, 0.95), 80: (0.10, 0.90), 50: (0.25, 0.75)}


@dataclass
class Config:
    # ----- data -----
    data_start: str = "2018-01-01"      # first settlement date to download
    data_end: str | None = None         # None = up to two days ago

    # ----- chronological splits (inclusive dates, by target settlement date) -----
    train_end: str = "2023-12-31"
    val_start: str = "2024-01-01"
    val_end: str = "2024-12-31"
    test_start: str = "2025-01-01"

    # ----- the forecasting task -----
    issue_hour_local: int = 9           # forecast issued at 09:00 UK time on day D-1 for all of day D
    publication_lag_min: int = 15       # Elexon publishes outturn within 15 min of period end
    history_len: int = 7 * 48           # half-hours of history fed to the sequence encoder
    max_gap_fill: int = 2               # interpolate gaps of at most this many half-hours

    # ----- models and training -----
    models: tuple = ("linear", "mlp", "seq")
    hidden: int = 128
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 1024
    max_epochs: int = 60
    patience: int = 6
    seed: int = 42
    oracle_ablation: bool = True        # also train an MLP that sees the *actual* target-day temperature
    use_trend: bool = False             # calendar-time trend feature (off: extrapolates badly; see experiments.py)
    relative_target: bool = False       # model demand relative to the last 24 hours' mean (scale-free)

    quantiles: tuple = field(default=QUANTILES)
