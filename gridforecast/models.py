"""PyTorch quantile-regression models.

All models map a feature vector (and optionally a window of recent demand)
to K quantiles of the target half-hour's demand, in standardised units.

* ``LinearQR``   linear quantile regression: one linear function per quantile.
                 Crossing quantiles are fixed after the fact by sorting
                 (monotone rearrangement, Chernozhukov, Fernandez-Val & Galichon 2010).
* ``MLPQR``      a multilayer perceptron with a non-crossing output head.
* ``SeqQR``      an MLP that additionally reads the last 7 days of half-hourly
                 demand through a dilated 1-D convolutional encoder.

Non-crossing head: the network outputs the lowest quantile and K-1 positive
increments (softplus), and cumulative sums give ordered quantiles by construction.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def pinball_loss(pred: torch.Tensor, y: torch.Tensor, taus: torch.Tensor) -> torch.Tensor:
    """Mean pinball (quantile) loss.

    For quantile level tau and error u = y - q:
        loss = tau * u        if u >= 0   (actual above the forecast quantile)
             = (tau - 1) * u  if u <  0   (actual below it)
    which equals max(tau * u, (tau - 1) * u).
    """
    u = y.unsqueeze(1) - pred
    return torch.maximum(taus * u, (taus - 1) * u).mean()


class NonCrossingHead(nn.Module):
    def __init__(self, d_in: int, n_q: int):
        super().__init__()
        self.lin = nn.Linear(d_in, n_q)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        raw = self.lin(h)
        base = raw[:, :1]
        steps = F.softplus(raw[:, 1:]) + 1e-4
        return torch.cat([base, base + torch.cumsum(steps, dim=1)], dim=1)


class LinearQR(nn.Module):
    uses_history = False

    def __init__(self, n_features: int, n_q: int, **_):
        super().__init__()
        self.lin = nn.Linear(n_features, n_q)

    def forward(self, x, hist=None):
        out = self.lin(x)
        if not self.training:
            out, _ = torch.sort(out, dim=1)   # monotone rearrangement at prediction time
        return out


def _mlp(d_in: int, hidden: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
    )


class MLPQR(nn.Module):
    uses_history = False

    def __init__(self, n_features: int, n_q: int, hidden: int = 128, dropout: float = 0.1, **_):
        super().__init__()
        self.body = _mlp(n_features, hidden, dropout)
        self.head = NonCrossingHead(hidden, n_q)

    def forward(self, x, hist=None):
        return self.head(self.body(x))


class HistoryEncoder(nn.Module):
    """Dilated 1-D CNN over the last 7 days (336 half-hours) of demand.

    The whole window lies before the information cutoff, so the convolutions
    need not be causal. Seven layers with kernel 3 and dilations 1, 2, 4, ..., 64
    give each output a receptive field of 255 half-hours (about 5.3 days),
    so the encoder can pick up the shape of recent daily profiles and any
    recent level shift (a cold spell, a holiday week) that the hand-made lag
    features summarise only crudely.
    """

    def __init__(self, channels: int = 32, out_dim: int = 32):
        super().__init__()
        layers, c_in = [], 1
        for d in (1, 2, 4, 8, 16, 32, 64):
            layers += [nn.Conv1d(c_in, channels, kernel_size=3, dilation=d, padding=d), nn.GELU()]
            c_in = channels
        self.conv = nn.Sequential(*layers)
        # summarise both the final day (recent level) and the whole week
        self.out = nn.Linear(2 * channels, out_dim)

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        h = self.conv(hist.unsqueeze(1))                  # (B, C, T)
        last_day = h[:, :, -48:].mean(dim=2)
        week = h.mean(dim=2)
        return F.gelu(self.out(torch.cat([last_day, week], dim=1)))


class SeqQR(nn.Module):
    uses_history = True

    def __init__(self, n_features: int, n_q: int, hidden: int = 128, dropout: float = 0.1, **_):
        super().__init__()
        self.enc = HistoryEncoder()
        self.body = _mlp(n_features + 32, hidden, dropout)
        self.head = NonCrossingHead(hidden, n_q)

    def forward(self, x, hist=None):
        windows, inverse = hist                      # unique windows + row -> window map
        return self.head(self.body(torch.cat([x, self.enc(windows)[inverse]], dim=1)))


MODEL_CLASSES = {"linear": LinearQR, "mlp": MLPQR, "seq": SeqQR, "mlp_oracle": MLPQR}
