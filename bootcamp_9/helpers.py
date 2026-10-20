"""Shared utilities for the Bootcamp 9 workbooks.

The upstream public package is an inference implementation. The fine-tuning
loop below is an educational adapter around its differentiable predict_block
primitive, not Mistral Forge's training pipeline.
"""
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

MODEL_ID = "mistralai/Timestral-5M"
# Set to the public model repository commit used for this exercise.
MODEL_REVISION = "bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f"
LSTM_LAGS = np.array([3, 2, 1, 24, 24 * 7, 24 * 365, 24 * 365 * 2])


@dataclass(frozen=True)
class ComparisonData:
    train_values: np.ndarray
    train_dates: np.ndarray
    origins: np.ndarray
    contexts: np.ndarray
    actuals: np.ndarray
    dates: np.ndarray
    seasonal_naive: np.ndarray
    horizon: int


def prepare_comparison(frame, train_size, context_length=672, horizon=24, days=14):
    """Use the existing split; all fine-tuning targets stay before train_size."""
    if horizon != 24:
        raise ValueError("Use 24: the existing LSTM outputs exactly 24 steps.")
    if context_length < 32 or context_length > 32768 or context_length % 32:
        raise ValueError("Context must be a multiple of 32 between 32 and 32768.")
    if days < 1:
        raise ValueError("Choose at least one evaluation day.")
    if train_size < max(context_length + horizon, int(LSTM_LAGS.max())):
        raise ValueError("Not enough training history for these models.")
    values = frame['mw'].to_numpy(dtype=np.float32, copy=True)
    dates = pd.to_datetime(frame['date']).to_numpy(dtype='datetime64[ns]')
    if not np.isfinite(values).all():
        raise ValueError("This exercise expects finite electricity measurements.")
    if train_size + days * horizon > len(values):
        raise ValueError("Not enough held-out data for the requested evaluation days.")
    origins = train_size + np.arange(days) * horizon
    for origin in origins:
        span = dates[origin - context_length:origin + horizon]
        if not np.all(np.diff(span) == np.timedelta64(1, 'h')):
            raise ValueError("An evaluation window contains missing or duplicated hours; choose a continuous period.")
    return ComparisonData(
        train_values=values[:train_size].copy(),
        train_dates=dates[:train_size].copy(),
        origins=origins,
        contexts=np.stack([values[o - context_length:o] for o in origins]),
        actuals=np.stack([values[o:o + horizon] for o in origins]),
        dates=np.stack([dates[o:o + horizon] for o in origins]),
        seasonal_naive=np.stack([values[o - horizon:o] for o in origins]),
        horizon=horizon,
    )


def training_windows(data):
    """Daily training origins whose entire context and target are hourly."""
    context_length = data.contexts.shape[1]
    candidates = range(context_length, len(data.train_values) - data.horizon + 1, 24)
    ends = np.array([
        end for end in candidates
        if np.all(np.diff(data.train_dates[end - context_length:end + data.horizon])
                  == np.timedelta64(1, 'h'))
    ], dtype=np.int64)
    if not len(ends):
        raise ValueError("No complete hourly windows in the training period.")
    inputs = np.stack([data.train_values[e - context_length:e] for e in ends])
    targets = np.stack([data.train_values[e:e + data.horizon] for e in ends])
    return torch.from_numpy(inputs), torch.from_numpy(targets), ends


def load_pretrained(cache_dir: Path, model_id=MODEL_ID):
    """Download only the pinned config and safetensors weights; run locally."""
    from timestral import TimestralForecaster

    if model_id != MODEL_ID:
        raise ValueError(f"Use {MODEL_ID} for this exercise.")
    return TimestralForecaster.from_pretrained(
        model_id, revision=MODEL_REVISION, device='cpu', cache_dir=cache_dir,
    )


def forecast_days(forecaster, data):
    """Each batch item is an independent day-ahead forecast, in original MW."""
    result = forecaster.forecast(data.contexts, prediction_length=data.horizon)
    predictions = result.median.detach().cpu().numpy()[:, 0, :]
    if predictions.shape != data.actuals.shape or not np.isfinite(predictions).all():
        raise ValueError("Forecasts must be finite and cover exactly the held-out targets.")
    return predictions


class FineTuneHistory(list):
    """Training losses plus the independent validation checkpoints."""
    def __init__(self):
        super().__init__()
        self.validation = []
        self.selected_step = None
        self.zero_shot_validation_rmse = None
        self.selected_validation_rmse = None
        self.fit_measurements = None
        self.fitting_windows = None
        self.initial_fitting_loss = None
        self.selected_epoch = None
        self.validation_windows = None
        self.validation_start = None
        self.validation_end = None


def training_quantiles(model, inputs, horizon):
    """Differentiable one-patch counterpart of the default public forecast.

    Match its symmetric averaging and non-negative clipping, not just the bare
    network. This task always predicts 24 hours, inside one 32-hour patch.
    """
    if horizon > model.config.patch_size:
        raise ValueError("This teaching adapter supports a single forecast patch.")
    mask = torch.isfinite(inputs)
    groups = torch.arange(len(inputs), device=inputs.device)
    positive = model.predict_block(inputs, mask, groups, 1)
    reflected = model.predict_block(-inputs, mask, groups, 1)
    quantiles = 0.5 * (positive + (-reflected).flip(-1))
    nonnegative = (inputs >= 0).all(dim=-1)
    quantiles = torch.where(nonnegative[:, None, None], quantiles.clamp_min(0), quantiles)
    return quantiles[:, :horizon, :]


def split_validation(data, validation_days=84):
    """Chronological fitting/validation split, omitting non-hourly windows.

    Contexts may contain earlier observations; every fitting target still ends
    before the first candidate validation origin. DST gaps/duplicates must not
    be interpreted as ordinary hourly steps.
    """
    fit_end = len(data.train_values) - validation_days * data.horizon
    context_length = data.contexts.shape[1]
    if fit_end < context_length + data.horizon:
        raise ValueError("Not enough history before the validation period.")
    values, dates = data.train_values, data.train_dates
    candidates = range(fit_end, len(values) - data.horizon + 1, data.horizon)
    origins = np.array([
        end for end in candidates
        if np.all(np.diff(dates[end - context_length:end + data.horizon])
                  == np.timedelta64(1, 'h'))
    ], dtype=np.int64)
    if not len(origins):
        raise ValueError("No complete hourly validation windows.")
    fitting = replace(data, train_values=values[:fit_end].copy(),
                      train_dates=dates[:fit_end].copy())
    validation = replace(
        fitting, origins=origins,
        contexts=np.stack([values[o - context_length:o] for o in origins]),
        actuals=np.stack([values[o:o + data.horizon] for o in origins]),
        dates=np.stack([dates[o:o + data.horizon] for o in origins]),
        seasonal_naive=np.stack([values[o - data.horizon:o] for o in origins]),
    )
    return fitting, validation


def point_loss(model, inputs, targets, training_scale, horizon=24):
    """Squared error of the same central forecast used in the results table.

    This adapts the median channel as a point forecast for RMSE. The other
    quantiles are not trained/calibrated prediction intervals in this exercise.
    """
    quantiles = training_quantiles(model, inputs, horizon)
    median = quantiles[:, :, model.config.quantile_knots.index(0.5)]
    return ((median - targets) / training_scale).square().mean()


def fine_tune(forecaster, data, epochs=5, batch_size=16, learning_rate=1e-4, seed=42,
              validation_days=84, recent_days=730):
    """Minimize point-forecast MSE on recent data; select on pre-test RMSE.

    Copy the pretrained model. Visit each fitting window once per epoch. The
    final 12 pre-test weeks are validation only; skip windows crossing bad hours.
    Use two recent years of fitting targets rather than equally weighting every
    older year. Neither validation nor test targets enter an optimizer update.
    """
    from timestral import TimestralForecaster

    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
        raise ValueError("Fine-tuning epochs must be a positive integer.")
    if batch_size < 1 or learning_rate <= 0:
        raise ValueError("Batch size and learning rate must be positive.")
    if validation_days < 1:
        raise ValueError("Reserve at least one day for validation.")
    if recent_days < 1:
        raise ValueError("Use at least one recent fitting day.")
    fitting, validation = split_validation(data, validation_days)
    fit_end = len(fitting.train_values)
    inputs, targets, ends = training_windows(fitting)
    recent = ends >= fit_end - recent_days * data.horizon
    inputs, targets = inputs[recent], targets[recent]
    if not len(inputs):
        raise ValueError("No complete windows in the recent fitting period.")
    model = deepcopy(forecaster.model)
    device = next(model.parameters()).device
    adapted = TimestralForecaster(
        model, max_context=forecaster.max_context,
        per_shot_horizon_patches=forecaster.per_shot_horizon_patches,
        quantile_real_cap_k=forecaster.quantile_real_cap_k,
        extreme_value_threshold=forecaster.extreme_value_threshold,
    )
    history = FineTuneHistory()
    history.fit_measurements = fit_end
    history.fitting_windows = len(inputs)
    zero_validation = forecast_days(forecaster, validation)
    history.zero_shot_validation_rmse = float(np.sqrt(np.mean((zero_validation - validation.actuals) ** 2)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    # A fixed training-only scale preserves the MW squared-error weighting.
    # Per-window normalization would give low-variance days extra influence.
    training_scale = max(float(np.std(fitting.train_values)), 1.0)
    history.validation_windows = len(validation.origins)
    history.validation_start = str(validation.dates[0, 0])
    history.validation_end = str(validation.dates[-1, -1])
    generator = torch.Generator().manual_seed(seed)
    probe = torch.linspace(0, len(inputs) - 1, min(64, len(inputs))).long()
    probe_x, probe_y = inputs[probe].to(device), targets[probe].to(device)

    def loss_for(x, y):
        return point_loss(model, x, y, training_scale, data.horizon)

    with torch.no_grad():
        history.initial_fitting_loss = float(loss_for(probe_x, probe_y))
    print(f"Fine-tuning {len(inputs):,} windows for {epochs} full epochs ({batch_size} windows/batch)")
    best_score, best_weights = float('inf'), None
    step = 0
    for epoch in range(epochs):
        order = torch.randperm(len(inputs), generator=generator)
        for start in range(0, len(inputs), batch_size):
            model.train()
            sample = order[start:start + batch_size]
            x, y = inputs[sample].to(device), targets[sample].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(x, y)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss; stop and check inputs or learning rate.")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not torch.isfinite(grad_norm) or grad_norm <= 0:
                raise ValueError("Gradients must be finite and nonzero for adaptation.")
            optimizer.step()
            history.append(float(loss.detach()))
            step += 1
        prediction = forecast_days(adapted, validation)
        score = float(np.sqrt(np.mean((prediction - validation.actuals) ** 2)))
        with torch.no_grad():
            fixed_loss = float(loss_for(probe_x, probe_y))
        history.validation.append({'epoch': epoch + 1, 'step': step,
                                   'rmse_mw': score, 'fitting_loss': fixed_loss})
        print(f"Epoch {epoch + 1}/{epochs}: fixed fitting loss {fixed_loss:.4f}; validation RMSE {score:.2f} MW")
        if score < best_score:
            best_score = score
            best_weights = deepcopy(model.state_dict())
            history.selected_step = step
            history.selected_epoch = epoch + 1
    model.load_state_dict(best_weights)
    model.eval()
    history.selected_validation_rmse = best_score
    print(f"Selected trained checkpoint: epoch {history.selected_epoch}; validation RMSE {best_score:.2f} MW")
    print(f"Zero-shot validation RMSE: {history.zero_shot_validation_rmse:.2f} MW")
    return adapted, history


def forecast_lstm(model, scaler, train_normalized, test_normalized, data):
    """Re-evaluate the existing LSTM at the SAME day-ahead forecast origins."""
    history = np.concatenate([train_normalized, test_normalized])
    # Reuse the original LSTM's exact seven lag positions and seven features.
    indices = data.origins[:, None] - LSTM_LAGS[None, :]
    if np.any(indices < 0) or np.any(indices >= data.origins[:, None]):
        raise ValueError("All LSTM input rows must precede their forecast origin.")
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            scaled = model(torch.as_tensor(history[indices], dtype=torch.float32, device=device))
        predictions = scaler.inverse_transform(scaled.cpu().numpy().reshape(-1, 1))
        return predictions.reshape(data.actuals.shape)
    finally:
        model.train(was_training)


def score_forecasts(actuals, predictions):
    """Same targets, units and weighting for every comparison row."""
    actuals = np.asarray(actuals)
    rows = []
    for name, prediction in predictions.items():
        prediction = np.asarray(prediction)
        if prediction.shape != actuals.shape or not np.isfinite(prediction).all():
            raise ValueError(f"{name}: forecasts must match the target shape and be finite.")
        error = prediction - actuals
        row = {'Model': name, 'RMSE (MW)': float(np.sqrt(np.mean(error ** 2))),
               'MAE (MW)': float(np.mean(np.abs(error)))}
        if np.any(actuals == 0):
            raise ValueError("MAPE is undefined when an actual value is zero.")
        row['MAPE (%)'] = float(100 * np.mean(np.abs(error / actuals)))
        rows.append(row)
    return pd.DataFrame(rows)


# Use the same model colors and line styles in every comparison figure.
FORECAST_STYLES = {
    'Same hour yesterday': {'color': '#8a8a8a', 'linestyle': ':', 'linewidth': 1.3},
    'Our LSTM': {'color': '#bc751b', 'linestyle': '-.', 'linewidth': 1.4},
    'Timestral zero-shot': {'color': '#2878b5', 'linestyle': '--', 'linewidth': 1.5},
    'Timestral fine-tuned': {'color': '#c43c39', 'linestyle': '-', 'linewidth': 1.6},
}


def plot_forecast_views(data, predictions):
    """Actual demand and identical forecasts at overview, daily and weekly scales.

    Choose the beginning/end views in advance, not the best-performing days.
    Display raw hourly values; the metrics in each subtitle use exactly those
    forecast hours. Context history is only drawn in the overview.
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter

    dates, actuals = data.dates.ravel(), data.actuals.ravel()
    views = [
        ('All six test weeks', slice(None), True),
        ('First 24 hours', slice(0, 24), False),
        ('First week', slice(0, 168), False),
        ('Last week', slice(-168, None), False),
    ]
    figures = []
    with plt.rc_context({'font.size': 11, 'axes.titlesize': 15, 'legend.fontsize': 10}):
        for title, span, show_history in views:
            fig, ax = plt.subplots(figsize=(15, 5.5), layout='constrained')
            if show_history:
                ax.plot(data.train_dates[-168:], data.train_values[-168:],
                        color='#2878b5', alpha=.45, linewidth=1.1,
                        label='Observed training history')
                ax.axvline(dates[0], color='#555555', linestyle=':', linewidth=1)
                ax.text(dates[0], .98, ' Test starts', transform=ax.get_xaxis_transform(),
                        color='#555555', fontsize=10, va='top')
            ax.plot(dates[span], actuals[span], color='#111111', linewidth=2,
                    label='Actual demand', zorder=5)
            metrics = []
            for name, prediction in predictions.items():
                values = np.asarray(prediction).ravel()[span]
                ax.plot(dates[span], values, label=name, **FORECAST_STYLES[name])
                if name != 'Same hour yesterday':
                    rmse = np.sqrt(np.mean((values - actuals[span]) ** 2))
                    short_name = name.removeprefix('Timestral ')
                    metrics.append(f'{short_name}: {rmse:,.0f} MW')
            ax.set_title(f'{title} · rolling 24-hour forecasts', loc='left', pad=42)
            ax.text(0, 1.045, 'RMSE over shown forecast hours  |  ' + '   ·   '.join(metrics),
                    transform=ax.transAxes, color='#555555', fontsize=10)
            ax.set_ylabel('Electricity consumption (MW)')
            ax.set_xlabel('Forecasted time')
            locator = mdates.AutoDateLocator(minticks=5, maxticks=9)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
            ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
            ax.grid(True, alpha=.2)
            ax.spines[['top', 'right']].set_visible(False)
            ax.margins(x=.01)
            fig.legend(*ax.get_legend_handles_labels(), loc='outside lower center',
                       ncol=3, frameon=False)
            figures.append(fig)
    return figures


def plot_daily_errors(data, predictions):
    """Show day-to-day error, retaining both reported test subperiods."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter

    fig, ax = plt.subplots(figsize=(15, 5), layout='constrained')
    for name, prediction in predictions.items():
        daily_rmse = np.sqrt(np.mean((np.asarray(prediction) - data.actuals) ** 2, axis=1))
        ax.plot(data.dates[:, 0], daily_rmse, label=name, **FORECAST_STYLES[name])
    if len(data.origins) > 14:
        boundary = data.dates[14, 0]
        ax.axvline(boundary, color='#555555', linestyle=':', linewidth=1)
        ax.text(boundary, .98, ' Weeks 3–6 start', transform=ax.get_xaxis_transform(),
                fontsize=10, color='#555555', va='top')
    ax.set_title('Forecast error by day · every test origin', loc='left', fontsize=15, pad=35)
    ax.set_ylabel('Daily RMSE (MW)')
    ax.set_xlabel('Start of the 24-hour forecast')
    ax.set_ylim(bottom=0)
    locator = mdates.AutoDateLocator(minticks=5, maxticks=9)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.grid(True, alpha=.2)
    ax.spines[['top', 'right']].set_visible(False)
    fig.legend(*ax.get_legend_handles_labels(), loc='outside lower center', ncol=4, frameon=False)
    return fig


def plot_learning_curves(history):
    """Fitting loss and independent validation error, with the selected epoch."""
    import matplotlib.pyplot as plt

    fig, (train_ax, validation_ax) = plt.subplots(2, 1, figsize=(15, 8), layout='constrained')
    steps = [record['step'] for record in history.validation]
    train_ax.plot(np.arange(1, len(history) + 1), history, color='#2878b5', alpha=.3,
                  linewidth=1, label='Minibatch MSE')
    train_ax.plot([0] + steps, [history.initial_fitting_loss] +
                  [record['fitting_loss'] for record in history.validation],
                  color='#111111', marker='o', linewidth=2, label='Same fixed fitting windows')
    train_ax.set_title('Fine-tuning · fitting loss', loc='left', fontsize=15)
    train_ax.set_ylabel('MSE / fixed training scale²')
    train_ax.set_xlabel('Optimizer update')
    train_ax.set_ylim(bottom=0)
    train_ax.legend(frameon=False)

    epochs = [record['epoch'] for record in history.validation]
    scores = [record['rmse_mw'] for record in history.validation]
    validation_ax.plot(epochs, scores, color='#c43c39', marker='o', linewidth=2,
                       label='Trained checkpoints')
    validation_ax.axhline(history.zero_shot_validation_rmse, color='#2878b5', linestyle='--',
                          label=f'Zero-shot: {history.zero_shot_validation_rmse:,.0f} MW')
    validation_ax.scatter(history.selected_epoch, history.selected_validation_rmse,
                          s=150, marker='*', color='#111111', zorder=5,
                          label=f'Selected epoch {history.selected_epoch}: {history.selected_validation_rmse:,.0f} MW')
    validation_ax.set_title('Fine-tuning · pre-test validation RMSE', loc='left', fontsize=15)
    validation_ax.set_ylabel('Validation RMSE (MW)')
    validation_ax.set_xlabel('Completed epoch')
    validation_ax.set_xticks(epochs)
    validation_ax.margins(y=.2)
    validation_ax.legend(frameon=False)
    for ax in [train_ax, validation_ax]:
        ax.grid(True, alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    return fig
