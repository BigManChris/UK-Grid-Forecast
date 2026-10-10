# Grid-Quantile-Forecast

Probabilistic day-ahead forecasts of Great Britain's national electricity demand, built with PyTorch and benchmarked against the system operator's official forecast.

Every morning at 09:00, the model forecasts the **full distribution** of national demand for each half-hour of the next day: nine quantiles from 2.5% to 97.5%, giving 50%, 80%, 90% and 95% prediction intervals. The models are trained with the pinball (quantile) loss and judged on accuracy *and* calibration: does a 90% interval really contain the outcome 90% of the time?

- **Methodology:** [docs/REPORT.md](docs/REPORT.md) explains every step and the reasoning behind it.
- **Results:** [results/RESULTS.md](results/RESULTS.md) is generated automatically by each run. The committed version comes from `python run.py --rolling`.

## Highlights

- **Leakage-free by construction.** Inputs are restricted to what was published by 09:00 the day before, and a test perturbs all later data to prove no input changes.
- **Three PyTorch models:** linear quantile regression, an MLP with a non-crossing quantile head, and an MLP with a dilated-CNN encoder over the last seven days of half-hourly demand.
- **Real benchmarks:** a seasonal naive forecast and the official NESO day-ahead forecast, both turned into distributions in the same way and scored on identical half-hours.
- **Evaluation like a risk desk:** pinball loss, coverage, reliability diagrams, Kupiec tail backtests, Diebold–Mariano significance tests and a breakdown by regime (holidays, cold days, clock changes).
- **An oracle ablation** that measures how much a perfect temperature forecast would be worth.

## Quick start

Requires Python 3.10 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

python run.py --demo              # 1-2 minutes on simulated data, no internet needed: checks everything works
python -m pytest -q               # correctness checks (leakage, clock changes, losses, tests)
python run.py                     # real data: downloads from Elexon, trains, writes results/RESULTS.md
```

The first real run downloads data from 2018 onwards, which takes a few minutes; it is cached in `data/raw/`, so later runs start immediately.

Options:

| Flag | Effect |
|---|---|
| `--rolling` | quarterly rolling-origin re-training over the test period (slower, more realistic) |
| `--models linear,mlp` | train a subset of `linear`, `mlp`, `seq` |
| `--epochs 20` | cap the number of training epochs |
| `--start 2020-01-01` | first date to download |
| `--no-oracle` | skip the oracle-temperature ablation |
| `--download-only` | fetch the data into `data/processed/` and stop (needs only `requests` and `pandas`) |
| `--offline` | train on data already in `data/processed/` without downloading |
| `--demo` | use simulated data (for testing only; results are meaningless) |

Dates for the train, validation and test splits are in [`gridforecast/config.py`](gridforecast/config.py).

## Project structure

```
run.py                       pipeline entry point
gridforecast/
    config.py                all settings: dates, quantiles, issue time, model sizes
    elexon.py                Elexon API client with chunking, retries and caching
    calendar_utils.py        settlement periods, clock changes, bank holidays
    data.py                  cleaning and leakage-free feature construction
    models.py                PyTorch models and the pinball loss
    train.py                 training loop with early stopping
    evaluate.py              benchmarks, metrics and statistical tests
    report.py                figures and results/RESULTS.md
    synthetic.py             realistic simulated data for --demo and tests
experiments.py               design experiments scored on the validation year only
tests/test_pipeline.py       correctness checks
docs/REPORT.md               full methodology report
```

## Data

All data comes from Elexon's free [Insights Solution API](https://developer.data.elexon.co.uk/): half-hourly national demand outturn, NESO's day-ahead national demand forecast, and daily GB temperature. No API key is needed.

Contains BMRS data © Elexon Limited copyright and database right.
