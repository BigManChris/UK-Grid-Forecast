# Probabilistic Day-Ahead Forecasting of GB Electricity Demand

**Methodology report for the Grid-Quantile-Forecast project**

This report explains what the project does, how each step works, and why each design choice was made. The numbers from a run are written separately to [`results/RESULTS.md`](../results/RESULTS.md), which the pipeline generates automatically. That file appears after you run `python run.py`.

---

## Contents

1. [The question](#1-the-question)
2. [Why forecast quantiles instead of a single number](#2-why-forecast-quantiles-instead-of-a-single-number)
3. [The data](#3-the-data)
4. [Step 1: Downloading](#4-step-1-downloading)
5. [Step 2: Cleaning](#5-step-2-cleaning)
6. [Step 3: Defining the forecasting task precisely](#6-step-3-defining-the-forecasting-task-precisely)
7. [Step 4: Features](#7-step-4-features)
8. [Step 5: Splitting the data in time](#8-step-5-splitting-the-data-in-time)
9. [Step 6: The models](#9-step-6-the-models)
10. [Step 7: Training](#10-step-7-training)
11. [Step 8: Benchmarks](#11-step-8-benchmarks)
12. [Step 9: Evaluation](#12-step-9-evaluation)
13. [Step 10: The oracle ablation](#13-step-10-the-oracle-ablation)
14. [Safeguards against fooling ourselves](#14-safeguards-against-fooling-ourselves)
15. [Findings from the first real run](#15-findings-from-the-first-real-run)
16. [Limitations](#16-limitations)
17. [Extensions](#17-extensions)
18. [Reproducing the results](#18-reproducing-the-results)
19. [References](#19-references)

---

## 1. The question

**At 09:00 UK time each day, how well can we forecast the full probability distribution of Great Britain's national electricity demand for every half-hour of the following day, and is that forecast better than the system operator's?**

The output for each half-hour is not one number but nine quantiles (2.5%, 5%, 10%, 25%, 50%, 75%, 90%, 95% and 97.5%). Together they describe a forecast distribution: a central estimate plus ranges that should contain the actual demand 50%, 80%, 90% and 95% of the time.

The project is judged on two things:

- **Accuracy**, measured by the pinball loss (Section 12.1).
- **Calibration**: whether a "90% interval" really contains the outcome about 90% of the time (Section 12.2).

## 2. Why forecast quantiles instead of a single number

A single-number forecast says nothing about how wrong it might be. Most decisions in electricity, and in trading generally, depend on the tails rather than the average:

- The system operator must hold enough reserve generation to cover demand being higher than forecast. That depends on, say, the 97.5% quantile, not the median.
- A battery or a trader deciding whether to buy power in advance cares about the probability of scarcity, not the expected price.
- Risk limits (such as value-at-risk) are statements about quantiles.

A forecast distribution answers all of these at once. It also creates an honest test: a model that is overconfident (intervals too narrow) or timid (intervals too wide) is caught by calibration checks even when its central estimate is good.

## 3. The data

All data comes from **Elexon's Insights Solution API** (`https://data.elexon.co.uk/bmrs/api/v1`). Elexon runs the GB electricity settlement system and publishes market and system data for free, with no registration or API key. The data may be reused with the attribution "Contains BMRS data © Elexon Limited copyright and database right."

| Dataset | Endpoint | Resolution | Role |
|---|---|---|---|
| Initial National Demand Outturn (INDO) | `/demand/outturn` | half-hourly, MW | the quantity being forecast |
| Day-ahead National Demand Forecast (NDF), earliest publication | `/forecast/demand/day-ahead/earliest` | half-hourly, MW | the official benchmark |
| NDF revisions for 00:00–06:30 | `/forecast/demand/day-ahead/evolution` | half-hourly, MW | the benchmark as it stood at our issue time (Section 11) |
| GB average temperature | `/temperature` | daily, °C | input feature |

**History available.** Demand outturn goes back to the start of the download (1 January 2018). The API's history of NESO's forecast and of the temperature series starts in late April 2021. The demand record has a handful of whole missing days before 2021, which the cleaning step leaves missing (Section 5).

**Which demand measure.** Elexon publishes two demand measures. INDO is national demand including transmission losses, but excluding power stations' own consumption, pumped-storage pumping and interconnector exports. ITSDO (transmission system demand) includes those. This project forecasts INDO because it is the same definition as the official National Demand Forecast, so the benchmark comparison is like-for-like. INDO is also closer to underlying consumer demand, whereas pumped storage and exports respond to prices.

**What "national demand" does not include.** INDO is measured on the transmission network, so electricity from rooftop solar panels and small wind farms connected to local distribution networks never appears in it. Instead it shows up as *reduced* demand. On a sunny summer afternoon, measured demand dips by several gigawatts. This is why the forecasting problem has a weather component even in summer.

**Settlement periods.** The GB market divides each day into half-hour settlement periods numbered from 1, starting at 00:00 UK local time. Most days have 48. The day the clocks go forward has 46 and the day they go back has 50.

## 4. Step 1: Downloading

*Code: [`gridforecast/elexon.py`](../gridforecast/elexon.py)*

1. The date range (by default 1 January 2018 to two days ago) is split into chunks: 28 days for demand, 14 for the forecast, 366 for temperature. The forecast-revision endpoint accepts only one settlement date per request, so it is called once per day from April 2021.
2. Each chunk is requested as JSON. Failed requests are retried with exponential back-off (1, 2, 4, 8 and 16 seconds), and the client waits 0.2 seconds between requests to be polite to a free service.
3. If the API rejects a chunk with HTTP 400 (usually because the date range is too long), the chunk length is halved, remembered in `data/raw/<dataset>/_chunk_days.txt`, and the request retried.
4. Every completed chunk is saved under `data/raw/`. Re-running the pipeline reads from this cache, so the full history is only downloaded once. Chunks from the last five days are not cached, because recent data can still be revised.

Demand is required. The benchmark and temperature are optional: if either cannot be downloaded, the pipeline continues without it and says so in the results report.

## 5. Step 2: Cleaning

*Code: `clean_demand`, `clean_ndf` and `clean_temperature` in [`gridforecast/data.py`](../gridforecast/data.py)*

1. **Convert to UTC.** Every half-hour is identified by its UTC start time (`startTime`). Working in UTC makes the series a perfectly regular grid with no repeated or missing hours at the clock changes. Local time is used only to build calendar features.
2. **Remove duplicates.** If the same half-hour was published more than once, the latest publication is kept, since it reflects any corrections.
3. **Remove implausible values.** Values at or below 5,000 MW or above 80,000 MW are set to missing. GB national demand has stayed well inside this range in modern history, so anything outside it is a data error.
4. **Reindex onto a complete grid.** The series is aligned to every half-hour between the first and last observation, so any gaps become explicit missing values.
5. **Fill only short gaps.** Runs of at most two missing half-hours (one hour) are filled by linear interpolation between their neighbours. Longer gaps are left missing, and any forecast that would need those values is dropped rather than built on made-up data.
6. **Log what happened.** The number of duplicates, implausible values and missing half-hours before and after filling is written to the results report.

For the benchmark, the national figure is kept (zonal rows are discarded) and, for each half-hour, the version in force at our issue time is selected (Section 11). Temperature is reduced to one value per day.

## 6. Step 3: Defining the forecasting task precisely

*Code: `build_dataset` in [`gridforecast/data.py`](../gridforecast/data.py)*

This is the most important step for credibility. Day-ahead forecasting projects most often fail by accidentally using information that would not have existed when the forecast was made.

**The setup.** A forecast for day *D* is issued at **09:00 UK local time on day *D*−1**. It covers every half-hour of *D*, from 00:00 to 23:30 (or the 46 or 50 periods on clock-change days).

**What is known at 09:00.** Elexon publishes each half-hour's demand within 15 minutes of the half-hour ending. The half-hour starting at 08:00 ends at 08:30 and is published by 08:45, so it is available. The half-hour starting at 08:30 ends at 08:45 is not. The **information cutoff** is therefore the half-hour starting at 08:00 on *D*−1:

```
          D-2                          D-1                                   D
 ... ───────────────┬──────────────────────────────────────┬──────────────────────────────────┐
                    00:00          08:00 08:30 08:45  09:00 00:00                         23:30
                                     └─ last usable ─┘    │  └──────── forecast targets ───────┘
                                        half-hour   published  issue time
                                                      by here
```

Formally, for issue time *T*, the last usable half-hour is the latest start time *s* on the 30-minute grid with *s* + 30 min + 15 min ≤ *T*.

**Lead times.** The forecast for 00:00 on *D* is made 15 hours ahead, and the forecast for 23:30 is made 38.5 hours ahead. The median lead time is about 26.5 hours. Lead time is given to the model as a feature, since uncertainty grows with it.

**One row per target half-hour.** The dataset has one row for each half-hour to be forecast. This handles clock-change days naturally: a 46-period day simply contributes 46 rows. A model that output a fixed vector of 48 values would need special handling for those days.

## 7. Step 4: Features

*Code: `build_dataset` and `feature_columns` in [`gridforecast/data.py`](../gridforecast/data.py)*

Every feature is computable from information available at the cutoff.

| Group | Features | Why |
|---|---|---|
| Recent demand | demand at the same UK wall-clock time 2, 7 and 14 days earlier | demand is strongly weekly and daily periodic |
| Latest level | last observed half-hour; mean of the last 24 hours before the cutoff | captures the current level (a cold spell, a holiday week) |
| Time of day | sine and cosine of the local hour at 1, 2 and 3 cycles per day | smooth encoding of the daily profile, with 00:00 next to 23:30 |
| Day of week | seven 0/1 indicators | weekends differ sharply from weekdays |
| Holidays | target day is an England and Wales bank holiday; target day is in 24 Dec–2 Jan; the 7- and 14-day lag days were holidays | holidays look like Sundays, and a lag taken from a holiday is misleading |
| Season | sine and cosine of day of year at 1 and 2 cycles per year | heating, lighting and solar all follow the season |
| Clock | daylight-saving indicator | the profile shifts relative to sunlight |
| Lead time | hours between issue and target | uncertainty grows with lead time |
| Temperature | GB average temperature on *D*−2 and *D*−3; *D*−2 temperature minus its seasonal normal; a flag for missing temperature | heating demand; a cold spell tends to persist |
| History window (sequence model only) | the 336 half-hours (7 days) up to and including the cutoff | lets the network learn its own summaries of recent behaviour |

That is 28 base features plus 4 temperature features, 32 in total.

A calendar-time trend feature (years since 2018) was tested and dropped; Section 8.1 explains why.

**Missing temperature before April 2021.** Elexon's temperature series starts in April 2021, but demand goes back to 2018. Dropping three years of demand history would throw away much of what the models learn about daily and weekly patterns. Instead, missing temperatures are replaced by the average temperature for that day of the year (computed from training-period data only and smoothed over a 31-day window), the anomaly feature is set to zero, and a `temp_missing` flag is set to 1. The model can then learn to use temperature when it is real and to ignore it when it is not. Every test day, and all but two validation days, has real temperature data.

**Wall-clock lags.** "Same time a week ago" must mean the same *UK clock time*. Subtracting exactly 7 × 24 hours in UTC would compare 17:30 with 16:30 for a week after each clock change, which misaligns the evening peak. The lag is therefore corrected by the change in UTC offset. If *o*(*t*) is the UTC offset at time *t*, the lagged time is

$$t_{\text{lag}} = t - 7\,\text{days} + \big(o(t) - o(t - 7\,\text{days})\big).$$

A test checks that 17:30 BST on 2 April 2024 is matched with 17:30 GMT on 26 March 2024.

**What is deliberately left out.** The actual temperature on day *D* is *not* a feature, because it is unknown at 09:00 on *D*−1. A real forecaster would use a *weather forecast* for *D*, but no free archive of historical temperature forecasts is available through Elexon. The project therefore uses only observed past temperatures, and measures separately how much a perfect temperature forecast would be worth (Section 13).

**Standardisation.** Each feature and the target are standardised (mean subtracted, divided by standard deviation) using statistics from the **training rows only**. Using validation or test rows would leak information about the future into the scaling.

## 8. Step 5: Splitting the data in time

*Code: `make_folds` in [`run.py`](../run.py); dates in [`gridforecast/config.py`](../gridforecast/config.py)*

Rows are split by the target date, never at random:

| Split | Dates | Used for |
|---|---|---|
| Training | 2018 to 2023 | fitting model weights |
| Validation | 2024 | early stopping, choosing the headline model, calibrating the benchmark's error distribution |
| Test | 2025 onwards | the final, untouched evaluation |

A random split would put half-hours from the same day in both training and test. Neighbouring half-hours are almost identical, so the model would effectively be tested on data it had already seen, and results would look far better than they could be in practice.

**Rolling-origin evaluation (`--rolling`).** A single split tests one model that gets steadily staler through the test period. The rolling option instead mimics live use: for each calendar quarter in the test period, a fresh model is trained on all data up to one year before that quarter, early-stopped on the following year, and used to forecast the quarter. The quarterly forecasts are then joined and scored together.

### 8.1 Design choices tested on the validation year

A few design questions could not be settled in advance, so they were tested by training each variant on the training years and scoring it on **the 2024 validation year only**, with two random seeds each. The test period was not used. The script is [`experiments.py`](../experiments.py), and the results are saved in `results/validation_experiments.csv`.

| Variant | Linear | MLP | MLP + encoder |
|---|---|---|---|
| Baseline: absolute MW, with a trend feature | 362.2 | 304.1 | 307.1 |
| No trend feature | 362.3 | **300.8** | **300.6** |
| Demand relative to the last 24 hours' mean, with trend | 360.8 | 303.5 | 304.7 |
| Relative to the last 24 hours' mean, no trend | 360.9 | 304.5 | 303.3 |

*Mean validation pinball loss in MW, averaged over two seeds (lower is better). The spread between seeds was up to 8 MW.*

The differences are small, comparable to the noise between random seeds, so none of them is decisive on its own. The **trend feature was dropped** because removing it was slightly better for both neural networks *and* is safer in principle. In the test period the trend takes values beyond anything seen in training, and neural networks extrapolate unpredictably outside their training range. Expressing demand **relative to its recent level** (a scale-free target meant to cope with a drifting demand level) brought no gain, so the simpler absolute-MW target was kept.

## 9. Step 6: The models

*Code: [`gridforecast/models.py`](../gridforecast/models.py)*

### 9.1 The pinball loss

All models are trained by minimising the **pinball loss** (also called the quantile loss). For a quantile level *τ*, forecast quantile *q* and actual outcome *y*, let *u* = *y* − *q*:

$$\rho_\tau(u) = \begin{cases} \tau\, u & u \ge 0 \\ (\tau - 1)\, u & u < 0 \end{cases} \;=\; \max\big(\tau u,\ (\tau-1)u\big).$$

**Why this produces quantiles.** For the 90% quantile (*τ* = 0.9), a forecast that is too low costs 0.9 per MW, while a forecast that is too high costs only 0.1 per MW. The best response is to aim high, but how high? Differentiating the expected loss with respect to *q*:

$$\frac{\partial}{\partial q}\,\mathbb{E}\big[\rho_\tau(Y - q)\big] = F(q) - \tau,$$

where *F* is the cumulative distribution of *Y*. This is zero exactly when *F*(*q*) = *τ*, so the expected loss is minimised by the true *τ*-quantile. Training on the average pinball loss over all nine quantile levels therefore teaches the network all nine quantiles at once. With *τ* = 0.5, the loss is half the absolute error and gives the median.

### 9.2 Linear quantile regression (`linear`)

One linear function of the features for each quantile level: 32 inputs × 9 outputs, 297 parameters. It is the transparent baseline: if a neural network cannot beat it, the network's complexity is not earning its keep.

Separate linear functions can **cross**, for example a 90% quantile below the 50% quantile. At prediction time, each row's nine values are sorted. This "monotone rearrangement" can only move the estimated quantile curve closer to the true one, never further away (Chernozhukov, Fernández-Val and Galichon, 2010).

### 9.3 Multilayer perceptron (`mlp`)

Two hidden layers of 128 units with GELU activations and 10% dropout, followed by a non-crossing output head (about 22,000 parameters). The network can learn interactions a linear model cannot, such as the temperature effect being larger in the evening than at night, or the weekend effect differing by season.

**Non-crossing head.** Instead of nine independent outputs, the head produces the lowest quantile and eight increments, each passed through a softplus function (which is always positive):

$$q_1 = a_1, \qquad q_k = q_{k-1} + \operatorname{softplus}(a_k) + 10^{-4}, \quad k = 2,\dots,9.$$

The quantiles are therefore ordered by construction, with no post-processing needed.

### 9.4 MLP with a history encoder (`seq`)

The same MLP, but it also receives a 32-number summary of the last 7 days (336 half-hours) of demand up to the cutoff. The summary comes from a 1-D convolutional network: seven convolution layers with 32 channels, kernel size 3 and dilations 1, 2, 4, …, 64. Each output position therefore "sees" 255 half-hours, about 5.3 days. The encoder output is averaged over the final day and over the whole week, and the two averages are combined into the 32-number summary. This adds about 25,000 parameters (about 47,000 in total).

The idea is that the hand-made lag features summarise recent history crudely, while the encoder can learn which patterns matter, such as how the shape of yesterday's evening peak differs from normal. Whether that helps is an empirical question that the evaluation answers.

### 9.5 Model sizes

| Model | Parameters |
|---|---|
| Linear | 297 |
| MLP | ~22,000 |
| MLP + history encoder | ~47,000 |

These are small by modern standards. With roughly 100,000 training rows, larger networks would mostly add overfitting risk.

## 10. Step 7: Training

*Code: [`gridforecast/train.py`](../gridforecast/train.py)*

1. **Optimiser.** AdamW with learning rate 10⁻³ (10⁻² for the linear model, which needs larger steps) and weight decay 10⁻⁵.
2. **Batches made of whole days.** Each batch contains about 21 complete target days (around 1,000 rows). All half-hours of a day share the same history window, so the sequence model encodes each window once and reuses it for all of that day's rows. This makes training dozens of times faster than encoding a window per row, which is what makes the sequence model practical on a laptop CPU.
3. **Gradient clipping** at norm 5, to stop rare extreme errors from destabilising training.
4. **Learning-rate schedule.** The learning rate halves whenever the validation loss fails to improve for two epochs.
5. **Early stopping.** Training stops when the validation loss has not improved for six epochs (at most 60 epochs), and the weights from the best epoch are restored. This is the main defence against overfitting.
6. **Reproducibility.** Random seeds are fixed, so re-running gives the same results on the same machine.

Losses are computed on standardised values but reported in MW, so they are easy to interpret.

## 11. Step 8: Benchmarks

*Code: `dressed_point_forecast` in [`gridforecast/evaluate.py`](../gridforecast/evaluate.py)*

A model's score means little on its own. It needs reference points.

**Naive: same time last week (`naive_week`).** The point forecast is demand at the same wall-clock time seven days earlier. To make it probabilistic, the empirical quantiles of its errors over the training years are added to it. Any useful model must clearly beat this.

**The official forecast (`neso_dressed`).** NESO, the GB system operator, publishes its own day-ahead National Demand Forecast, produced by a professional team using weather forecasts and decades of experience.

NESO revises its forecast during the day, so *which version* is used matters. Inspecting the data shows the forecast for day *D* is published at about 08:45 UK time on *D*−1, just before our 09:00 issue time. However, for the first ten half-hours of *D* (00:00–04:30), the earliest version was published at 08:45 on *D*−2 and is then revised. Using the earliest version would hand NESO a forecast a day staler than ours for those hours and make our models look better than they are.

The benchmark therefore uses, for each half-hour, **the latest version published at or before 09:00 on *D*−1**: the forecast NESO had made public at the moment ours is issued. Versions published more than six hours before the issue time are discarded rather than used, and any half-hour without a qualifying version is left out of the comparison for every forecaster.

NESO publishes only a single number, so it is "dressed" in the same way: the empirical quantiles of its errors **in the validation year** are added to it. Using the validation year, rather than the test period, keeps the comparison fair, since we only use information we could have had in advance.

**Comparing like with like.** Every forecaster is scored on exactly the same test half-hours. If the official forecast is missing for some rows, those rows are dropped for everyone. The results report also shows the median lead time of the official forecast next to ours, so the reader can confirm the two are made at essentially the same time.

**What a fair outcome looks like.** NESO uses weather forecasts and this project does not, so NESO should win on days when the weather changes sharply. Getting close to the official forecast with a 47,000-parameter model and no weather forecast would be a strong result. Beating the naive benchmark by a wide margin is the minimum.

## 12. Step 9: Evaluation

*Code: [`gridforecast/evaluate.py`](../gridforecast/evaluate.py), [`gridforecast/report.py`](../gridforecast/report.py)*

### 12.1 Accuracy

- **Mean pinball loss (MW).** The pinball loss averaged over all nine quantiles and all test half-hours. It is a *proper scoring rule*: it is minimised in expectation only by reporting your true beliefs, so it cannot be gamed by making intervals artificially narrow or wide. It rewards calibration and sharpness together. Averaged over a fine grid of quantile levels it equals half the continuous ranked probability score (CRPS), so with nine levels it is a coarse approximation of CRPS/2.
- **Skill score versus naive:** 1 − (model loss ÷ naive loss). Zero means no better than last week; 100% would be perfect.
- **Median accuracy:** mean absolute error (MAE), root mean squared error (RMSE), mean absolute percentage error (MAPE) and bias of the 50% quantile.

### 12.2 Calibration

- **Interval coverage.** The share of test outcomes inside each central interval (50%, 80%, 90%, 95%), which should match the nominal level.
- **Interval width (sharpness).** The average width of each interval in MW. Narrow intervals are better *only if* coverage holds; trivially wide intervals are always calibrated and useless.
- **Reliability diagram.** For each quantile level *τ*, the share of outcomes below the forecast *τ*-quantile, plotted against *τ*. A perfectly calibrated forecaster lies on the diagonal.
- **Calibration histogram.** The share of outcomes falling between each pair of consecutive forecast quantiles, compared with the share expected (for example, 15% should fall between the 10% and 25% quantiles). This is the discrete version of a probability integral transform (PIT) histogram. A U-shape means intervals are too narrow; a hump means they are too wide; a slope means bias.

### 12.3 Tail tests (Kupiec)

For the 2.5%, 5%, 95% and 97.5% quantiles, an *exceedance* is an outcome beyond the quantile. Under correct calibration, exceedances occur at rate *p* (for example 5%). The Kupiec (1995) proportion-of-failures test compares the observed rate *x*/*n* with *p* using a likelihood ratio:

$$\mathrm{LR} = -2\ln\!\Big[(1-p)^{\,n-x}\,p^{\,x}\Big] + 2\ln\!\Big[\big(1-\tfrac{x}{n}\big)^{n-x}\big(\tfrac{x}{n}\big)^{x}\Big] \;\sim\; \chi^2_1 \text{ under } H_0.$$

This is the same test banks use to backtest value-at-risk models.

The test assumes exceedances are independent. Half-hourly exceedances are not: if 17:00 is above the 95% quantile, 17:30 probably is too. The test is therefore also run on **one half-hour per day (17:30, near the evening peak)**, which is much closer to independent. That version is the more trustworthy one.

### 12.4 Is the difference real? (Diebold–Mariano)

Two forecasters' losses on the same test period will always differ a little by chance. The Diebold–Mariano (1995) test asks whether the difference is larger than chance would explain. With daily mean pinball losses *L*ᴬ and *L*ᴮ, let *d*ₜ = *L*ᴬₜ − *L*ᴮₜ. Then

$$\mathrm{DM} = \frac{\bar d}{\sqrt{\hat\sigma^2_{\mathrm{NW}}/n}}, \qquad \hat\sigma^2_{\mathrm{NW}} = \hat\gamma_0 + 2\sum_{k=1}^{7}\Big(1 - \frac{k}{8}\Big)\hat\gamma_k,$$

where $\hat\gamma_k$ is the autocovariance of *d* at lag *k*. The Newey–West variance accounts for errors on nearby days being correlated (a model that struggles during a cold spell struggles for several days). Ignoring that would overstate significance. DM is compared with a standard normal; a negative value means A is better.

Tests reported: the headline model against NESO and against naive; the sequence model against the MLP (does the history encoder help?); the MLP against the linear model (does non-linearity help?); and the oracle against the MLP (what is temperature knowledge worth?).

### 12.5 Where the forecasts struggle

Accuracy and 50%/90% coverage are broken down by regime: weekdays, weekends, bank holidays and Christmas, clock-change days, winter, summer, and the coldest 10% of test days. The pinball loss is also plotted by time of day. Averages hide failures; a model can be well calibrated overall and badly overconfident on cold winter evenings, which is exactly when the grid is under most stress.

### 12.6 Recalibration

Neural networks trained with early stopping are often slightly overconfident: their intervals are a little too narrow. A simple remedy uses the validation year. For each quantile level *τ*, find the shift *δ*<sub>τ</sub> that would have made the forecast quantile's hit rate exactly *τ* on validation data,

$$\delta_\tau = \text{the } \tau\text{-quantile of } \big(y - \hat q_\tau\big) \text{ over the validation rows},$$

add it to every test-period forecast of that quantile, and re-sort. This is a simple version of conformalised quantile regression (Romano, Patterson and Candès, 2019). The recalibrated headline model is reported as an extra row so the reader can see its effect. It is never used to choose the headline model, and its effect is not used to make any design decision.

### 12.7 Model selection

The headline model is the one with the lowest **validation** loss, chosen before the test period is examined. Choosing the model with the best test score would turn the test set into a second validation set and make the reported test performance optimistically biased. This is the same multiple-testing problem that inflates backtested trading strategies.

## 13. Step 10: The oracle ablation

The `mlp_oracle` model is identical to the MLP but also receives the **actual** average temperature of the target day. That information is not available at 09:00 on *D*−1, so this model is **not a valid forecaster** and is excluded from model selection.

Its purpose is to answer a question: how much is temperature knowledge worth? The gap between the oracle and the ordinary MLP is an upper bound on the improvement a perfect temperature forecast could bring. Real temperature forecasts one to two days ahead are quite accurate (typically within 1–2 °C), so a large gap suggests that adding weather forecasts should be the next step. A small gap suggests effort is better spent elsewhere.

## 14. Safeguards against fooling ourselves

| Risk | Safeguard |
|---|---|
| Using future demand values | Information cutoff computed from the publication lag; `build_dataset` asserts every lag is at or before the cutoff; a test multiplies all demand after the cutoff by 3 and checks that no input for that day changes |
| Using future weather | Only temperatures from *D*−2 and *D*−3 are inputs; actual target-day temperature is used only by the labelled oracle |
| Clock-change errors | UTC grid; wall-clock lags; 46- and 50-period days handled as variable rows; tests for both |
| Random-split leakage | Splits by date only |
| Scaling leakage | Standardisation statistics from training rows only |
| Selecting on the test set | Headline model chosen on validation loss |
| Tuning the design on the test set | Design variants compared on the validation year only (`experiments.py`); recalibration shown alongside, never used for selection |
| Unfair benchmark | NESO's forecast taken as published by our issue time (a test checks this); its errors dressed using the validation year; identical test rows for every forecaster; lead times reported |
| Chance differences | Diebold–Mariano tests with autocorrelation-robust variance |
| Overconfident significance in tail tests | Kupiec test also on one observation per day |
| Silent data problems | Cleaning log in every results report; long gaps left missing rather than invented |

Run `python -m pytest -q` to execute the tests in [`tests/test_pipeline.py`](../tests/test_pipeline.py).

## 15. Findings from the first real run

*Run of 10 October 2026, quarterly rolling-origin evaluation (`python run.py --rolling`). Test period: 1 January 2025 to 8 October 2026. The full tables and charts are in [`results/RESULTS.md`](../results/RESULTS.md).*

> **Provisional:** at the time of this run, NESO's revised forecasts for 00:00–04:30 had not yet been downloaded, so those hours (about a fifth of the test period) were excluded for every forecaster. The numbers below cover 05:00–23:30 and will be updated once that data is in.

**1. The models clearly beat the naive benchmark but are well behind NESO.** The headline model (MLP with the history encoder, chosen on validation) has a mean pinball loss of 371.5 MW, 37% better than "same time last week" (593.5 MW). Its median forecast has a MAPE of 6.1%. NESO's official forecast, dressed in the same way, scores 207.9 MW with a MAPE of 3.2%. The gap is highly significant (Diebold–Mariano statistic 16.3, p < 0.0001).

**2. Each step up in model complexity helps, and the gains are statistically real.** The MLP beats linear quantile regression by 36 MW a day on average (p = 0.0001), and adding the 7-day history encoder improves on the MLP by a further 7 MW a day (p = 0.005).

**3. The gap to NESO is mostly a midday solar problem.** Our error peaks around 13:00 (about 560 MW mean pinball loss, against about 230 MW at 06:00). NESO's error peaks too, but far less (about 300 MW). Rooftop solar output depends on cloud cover, which a model without weather forecasts cannot anticipate, and NESO uses detailed weather forecasts. Summer 90% intervals cover only 79.5% of outcomes, the worst of any season.

**4. Temperature knowledge is worth something, but it isn't the main gap.** Giving the MLP the actual temperature of the target day (the oracle) cuts its loss by 31 MW (8%), still far short of NESO. Most of NESO's advantage must come from information a daily average temperature does not capture, above all solar and wind forecasts.

**5. The intervals are slightly too narrow, especially in the lower tail.** The headline model's 90% intervals contain 84.3% of outcomes. Demand fell below the 2.5% quantile 7.0% of the time: on sunny days, demand drops lower than the model expects. Recalibrating on the validation year lifts 90% coverage to 87.2% and slightly improves the pinball loss (368.5 MW). Neither model nor benchmark is perfectly calibrated in this period; NESO's dressed forecast also covers 87.7%.

**6. The test period is harder than the past.** Every forecaster, including NESO, does worse in 2025–26 than on the 2024 validation year, and validation loss rises in each successive rolling fold. Quarterly retraining helps a little (the headline model improves from 378.7 MW under a single split to 371.5 MW), but the main driver is that recent demand is more volatile.

**What to do next.** Elexon publishes NESO's day-ahead wind and solar generation forecasts. They are issued before our 09:00 cutoff, so they are legitimate inputs. Adding them targets the midday error directly and is the most promising next step.

## 16. Limitations

- **No weather forecasts.** This is the largest handicap relative to NESO. Section 13 measures its cost.
- **Embedded solar and wind are invisible.** They appear only as reduced demand, and their growth means the past is an imperfect guide to the future. The rolling-origin evaluation, which retrains every quarter, is the main defence against this drift.
- **Holidays.** Only England and Wales bank holidays are used. Scottish and Northern Irish holidays differ, but England and Wales account for most GB demand.
- **Initial versus settled data.** INDO is an *initial* operational measurement. Later settlement data can differ slightly. Both the model and the benchmark are scored against the same INDO values, so the comparison remains fair.
- **One test period.** 2025–2026 may not be representative. The rolling-origin option and the regime breakdown partly address this. Reporting results on another held-out year would strengthen the conclusions.
- **API assumptions.** The download code follows Elexon's published API specification but was written without live access to the API. If an endpoint's behaviour differs (for example, a maximum date range), the chunk-halving logic should adapt. Anything else would show up as an error or in the data-quality log.
- **Error dressing is static.** The benchmark's error distribution is the same in every half-hour, so it cannot widen on difficult days. This is the standard way to turn a point forecast into a distribution, but it slightly favours a model that adapts its uncertainty.

## 17. Extensions

1. **Weather forecasts.** Add historical temperature, wind and solar forecasts (for example from Open-Meteo's historical forecast archive) as genuine day-ahead inputs. The oracle ablation indicates the potential gain.
2. **Prices.** Apply the same pipeline to day-ahead or imbalance prices, using Elexon's market index data. Prices are heavy-tailed and can go negative, so tail calibration matters even more.
3. **Adaptive recalibration.** The recalibration in Section 12.6 applies one fixed shift per quantile. Making the shift depend on season or time of day, or updating it continuously as new errors arrive, would track changing conditions better.
4. **Battery trading simulation.** Use probabilistic price forecasts to schedule a simulated battery, and compare its profit using the full distribution, the median only, and perfect foresight.
5. **Ensembles.** Average the quantiles of several models or random seeds, which usually improves both accuracy and calibration.

## 18. Reproducing the results

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python -m pytest -q               # optional: run the checks
python run.py                     # download, train, evaluate, write results/RESULTS.md
python run.py --rolling           # optional: quarterly rolling-origin evaluation
python experiments.py --offline   # optional: the validation-year design experiments (Section 8.1)
```

The first run downloads several years of data, which takes a few minutes; later runs use the cache. Training all four models takes a few minutes on a laptop CPU.

`python run.py --demo` runs the whole pipeline on simulated data with no internet connection. It exists to test the code; its numbers mean nothing about the real grid.

## 19. References

- Chernozhukov, V., Fernández-Val, I. and Galichon, A. (2010). Quantile and probability curves without crossing. *Econometrica*, 78(3), 1093–1125.
- Diebold, F. X. and Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
- Gneiting, T. and Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association*, 102(477), 359–378.
- Hong, T. and Fan, S. (2016). Probabilistic electric load forecasting: a tutorial review. *International Journal of Forecasting*, 32(3), 914–938.
- Koenker, R. and Bassett, G. (1978). Regression quantiles. *Econometrica*, 46(1), 33–50.
- Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk measurement models. *Journal of Derivatives*, 3(2), 73–84.
- Newey, W. K. and West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708.
- Romano, Y., Patterson, E. and Candès, E. (2019). Conformalized quantile regression. *Advances in Neural Information Processing Systems*, 32.
- Elexon Insights Solution API documentation: <https://developer.data.elexon.co.uk/>

---

Contains BMRS data © Elexon Limited copyright and database right.
