"""Training loop: standardise, fit with pinball loss, stop early on validation loss."""

from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
import torch

from .config import Config
from .data import DEMAND_FEATURES, feature_columns
from .models import MODEL_CLASSES, pinball_loss


class Prepared:
    """Standardised tensors for one model, using statistics from the training rows only."""

    def __init__(self, ds: pd.DataFrame, series: np.ndarray, cols: list[str], fit_mask: np.ndarray, cfg: Config):
        # In relative mode every demand quantity (target, lags, history) is divided by the
        # mean demand of the 24 hours before the cutoff, so the model learns shapes and
        # ratios rather than absolute levels, which drift as embedded solar grows.
        self.relative = cfg.relative_target
        level = ds["y_mean24h"].to_numpy(dtype=np.float64)
        self.level = level if self.relative else np.ones(len(ds))
        X = ds[cols].to_numpy(dtype=np.float64).copy()
        if self.relative:
            for j, c in enumerate(cols):
                if c in DEMAND_FEATURES:
                    X[:, j] = X[:, j] / level - 1
        self.x_mean = X[fit_mask].mean(axis=0)
        self.x_std = X[fit_mask].std(axis=0) + 1e-8
        y = ds["y"].to_numpy(dtype=np.float64) / self.level - (1 if self.relative else 0)
        self.y_mean = float(y[fit_mask].mean())
        self.y_std = float(y[fit_mask].std())
        self.X = torch.tensor((X - self.x_mean) / self.x_std, dtype=torch.float32)
        self.y = torch.tensor((y - self.y_mean) / self.y_std, dtype=torch.float32)
        # whole demand history, standardised the same way as the target
        self.raw_series = np.nan_to_num(series, nan=float(np.nanmean(series)))
        self.cutoff = torch.tensor(ds["cutoff_pos"].to_numpy(), dtype=torch.long)
        self.offsets = torch.arange(-cfg.history_len + 1, 1, dtype=torch.long)
        if self.relative:
            roll = pd.Series(self.raw_series).rolling(48, min_periods=1).mean().to_numpy()
            self.series = torch.tensor(self.raw_series, dtype=torch.float32)
            self.level_at = torch.tensor(roll, dtype=torch.float32)
        else:
            self.series = torch.tensor((self.raw_series - self.y_mean) / self.y_std, dtype=torch.float32)

    def history(self, idx: torch.Tensor):
        """Demand windows ending at each row's information cutoff (inclusive).

        All half-hours of the same target day share one window, so only the
        unique windows are returned, plus an index mapping each row to its window.
        This makes the sequence model dozens of times cheaper to train.
        """
        uniq, inverse = torch.unique(self.cutoff[idx], return_inverse=True)
        windows = self.series[uniq.unsqueeze(1) + self.offsets]
        if self.relative:
            lvl = self.level_at[uniq].unsqueeze(1)
            windows = ((windows / lvl - 1) - self.y_mean) / self.y_std
        return windows, inverse


def _predict_std(model, prep: Prepared, idx: np.ndarray, batch: int = 8192) -> torch.Tensor:
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(idx), batch):
            b = torch.as_tensor(idx[i:i + batch])
            hist = prep.history(b) if model.uses_history else None
            outs.append(model(prep.X[b], hist))
    return torch.cat(outs) if outs else torch.empty(0, 0)


def fit(name: str, ds: pd.DataFrame, series: np.ndarray, cfg: Config,
        train_mask: np.ndarray, val_mask: np.ndarray, verbose: bool = True):
    """Train one model. Returns (model, prepared tensors, training history)."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    cols = feature_columns(ds, oracle=(name == "mlp_oracle"), use_trend=cfg.use_trend)
    prep = Prepared(ds, series, cols, train_mask, cfg)
    taus = torch.tensor(cfg.quantiles, dtype=torch.float32)
    model = MODEL_CLASSES[name](n_features=len(cols), n_q=len(cfg.quantiles),
                                hidden=cfg.hidden, dropout=cfg.dropout)
    lr = cfg.lr * 10 if name == "linear" else cfg.lr   # a linear model tolerates (and needs) a larger step
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    train_idx = np.flatnonzero(train_mask)
    # group training rows by target day: batches are made of whole days
    days = ds["cutoff_pos"].to_numpy()[train_idx]
    order = np.argsort(days, kind="stable")
    bounds = np.flatnonzero(np.diff(days[order])) + 1
    day_groups = np.split(train_idx[order], bounds)
    days_per_batch = max(1, cfg.batch_size // 48)
    val_idx = np.flatnonzero(val_mask)
    y_val = prep.y[torch.as_tensor(val_idx)]

    best, best_loss, bad_epochs, hist_log = None, float("inf"), 0, []
    t_start = time.time()
    for epoch in range(cfg.max_epochs):
        model.train()
        day_perm = np.random.permutation(len(day_groups))
        total, n = 0.0, 0
        for i in range(0, len(day_perm), days_per_batch):
            b = torch.as_tensor(np.concatenate([day_groups[j] for j in day_perm[i:i + days_per_batch]]))
            hist = prep.history(b) if model.uses_history else None
            loss = pinball_loss(model(prep.X[b], hist), prep.y[b], taus)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item() * len(b)
            n += len(b)
        val_loss = pinball_loss(_predict_std(model, prep, val_idx), y_val, taus).item()
        sched.step(val_loss)
        # losses are in standardised units; convert to MW for readability
        hist_log.append({"epoch": epoch + 1, "train_pinball_mw": total / n * prep.y_std,
                         "val_pinball_mw": val_loss * prep.y_std})
        if verbose:
            print(f"    epoch {epoch + 1:3d}  train {total / n * prep.y_std:7.1f} MW   "
                  f"val {val_loss * prep.y_std:7.1f} MW", flush=True)
        if val_loss < best_loss - 1e-5:
            best_loss, best, bad_epochs = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                break
    model.load_state_dict(best)
    if verbose:
        print(f"    best val pinball {best_loss * prep.y_std:.1f} MW, {time.time() - t_start:.0f}s", flush=True)
    return model, prep, pd.DataFrame(hist_log)


def predict(model, prep: Prepared, mask: np.ndarray) -> np.ndarray:
    """Quantile forecasts in MW, shape (rows, K)."""
    idx = np.flatnonzero(mask)
    out = _predict_std(model, prep, idx).numpy().astype(np.float64)
    out = out * prep.y_std + prep.y_mean
    if prep.relative:   # back from ratio-to-level to MW (a monotone map, so quantiles stay quantiles)
        lvl = prep.level[idx][:, None]
        out = (out + 1) * lvl
    return out
