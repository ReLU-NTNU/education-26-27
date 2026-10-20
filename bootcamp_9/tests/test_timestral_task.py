"""Offline checks for the comparison protocol; no model download required."""
from dataclasses import replace
from contextlib import contextmanager
from types import SimpleNamespace
import sys
import unittest

import numpy as np
import pandas as pd
import torch

from bootcamp_9 import timestral_task as task


def example_frame():
    return pd.DataFrame({
        'date': pd.date_range('2000-01-01', periods=23000, freq='h'),
        'mw': 100 + np.sin(np.arange(23000) / 24) * 10,
    })


class SmallModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))
        self.config = SimpleNamespace(patch_size=32, quantile_knots=(0.1, 0.5, 0.9))
        self.groups = []

    def predict_block(self, inputs, observed, group_ids, horizon_patches):
        self.groups.append(group_ids.detach().clone())
        return inputs[:, -1, None, None].expand(-1, 32 * horizon_patches, 3) * (1 + self.bias)


class SmallForecaster:
    def __init__(self, model, max_context=64, per_shot_horizon_patches=32,
                 quantile_real_cap_k=1e4, extreme_value_threshold=1e30):
        self.model = model
        self.max_context = max_context
        self.per_shot_horizon_patches = per_shot_horizon_patches
        self.quantile_real_cap_k = quantile_real_cap_k
        self.extreme_value_threshold = extreme_value_threshold


    def forecast(self, context, prediction_length):
        with torch.no_grad():
            quantiles = task.training_quantiles(self.model, torch.as_tensor(context), prediction_length)
        return SimpleNamespace(median=quantiles[:, :, 1].unsqueeze(1))


@contextmanager
def fake_timestral():
    previous = sys.modules.get('timestral')
    sys.modules['timestral'] = SimpleNamespace(TimestralForecaster=SmallForecaster)
    try:
        yield
    finally:
        if previous is None:
            sys.modules.pop('timestral', None)
        else:
            sys.modules['timestral'] = previous


class ComparisonProtocolTests(unittest.TestCase):
    def setUp(self):
        self.frame = example_frame()
        self.split = 22000
        self.data = task.prepare_comparison(self.frame, self.split, context_length=64, days=2)

    def test_targets_context_and_yesterday_share_exact_forecast_origins(self):
        for day, origin in enumerate(self.data.origins):
            np.testing.assert_array_equal(self.data.actuals[day], self.frame.mw.iloc[origin:origin + 24].astype('float32'))
            np.testing.assert_array_equal(self.data.contexts[day], self.frame.mw.iloc[origin - 64:origin].astype('float32'))
            np.testing.assert_array_equal(self.data.seasonal_naive[day], self.frame.mw.iloc[origin - 24:origin].astype('float32'))
        # Previous test-day observations are allowed as history at the next origin.
        np.testing.assert_array_equal(self.data.contexts[1, -24:], self.data.actuals[0])

    def test_all_training_targets_end_before_test_split(self):
        x, y, ends = task.training_windows(self.data)
        self.assertTrue(np.all(ends + 24 <= self.split))
        np.testing.assert_array_equal(y[-1], self.data.train_values[ends[-1]:ends[-1] + 24])
        self.assertEqual(tuple(x.shape[1:]), (64,))

    def test_future_values_cannot_change_fine_tuning_windows(self):
        changed = self.frame.copy()
        changed.loc[self.split:, 'mw'] = 1e9
        altered = task.prepare_comparison(changed, self.split, context_length=64, days=2)
        for original, modified in zip(task.training_windows(self.data), task.training_windows(altered)):
            np.testing.assert_array_equal(original, modified)

    def test_training_windows_skip_missing_hours(self):
        changed = self.frame.copy()
        changed.loc[1000:, 'date'] += pd.Timedelta(hours=1)
        altered = task.prepare_comparison(changed, self.split, context_length=64, days=2)
        _, _, ends = task.training_windows(altered)
        self.assertFalse(any(end - 64 < 1000 < end + 24 for end in ends))
        self.assertGreater(len(ends), 0)

    def test_bad_evaluation_hours_and_wrong_horizon_are_rejected(self):
        changed = self.frame.copy()
        changed.loc[self.split + 1, 'date'] += pd.Timedelta(hours=1)
        with self.assertRaisesRegex(ValueError, 'missing or duplicated'):
            task.prepare_comparison(changed, self.split, context_length=64, days=2)
        with self.assertRaisesRegex(ValueError, 'existing LSTM'):
            task.prepare_comparison(self.frame, self.split, horizon=48)

    def test_lstm_reads_only_the_original_past_lags(self):
        class RecordingLSTM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.tensor(0.0))

            def forward(self, x):
                self.seen = x.detach().numpy()
                return x[:, :1, 0].expand(-1, 24) + self.bias

        rows = np.repeat(np.arange(23000, dtype='float32')[:, None], 7, axis=1)
        model = RecordingLSTM()
        scaler = SimpleNamespace(inverse_transform=lambda values: values)
        result = task.forecast_lstm(model, scaler, rows[:self.split], rows[self.split:], self.data)
        expected = self.data.origins[:, None] - task.LSTM_LAGS[None, :]
        np.testing.assert_array_equal(model.seen[:, :, 0], expected)
        self.assertTrue(np.all(expected < self.data.origins[:, None]))
        self.assertEqual(result.shape, self.data.actuals.shape)
        self.assertTrue(model.training)

    def test_scores_use_same_targets_and_original_units(self):
        actual = np.array([[1., 2.], [3., 4.]])
        scores = task.score_forecasts(actual, {'Model': actual + 1}).iloc[0]
        self.assertEqual(scores['RMSE (MW)'], 1.)
        self.assertEqual(scores['MAE (MW)'], 1.)
        self.assertAlmostEqual(scores['MAPE (%)'], 100 * np.mean(1 / actual))
        with self.assertRaisesRegex(ValueError, 'target shape'):
            task.score_forecasts(actual, {'Broken': np.zeros(4)})

    def test_fine_tuning_copies_weights_and_never_uses_test_values(self):
        forecaster = SmallForecaster(SmallModel())
        altered = replace(self.data, actuals=np.full_like(self.data.actuals, 1e12),
                          contexts=np.full_like(self.data.contexts, -1e12))
        with fake_timestral():
            adapted, losses = task.fine_tune(forecaster, self.data, steps=2, batch_size=4)
            second, second_history = task.fine_tune(forecaster, altered, steps=2, batch_size=4)
        self.assertEqual(float(forecaster.model.bias.detach()), 0.)
        self.assertNotEqual(float(adapted.model.bias.detach()), 0.)
        torch.testing.assert_close(adapted.model.bias, second.model.bias, rtol=0, atol=0)
        self.assertTrue(np.isfinite(losses).all())
        self.assertLess(losses.fit_measurements, self.split)
        self.assertEqual(losses.validation, second_history.validation)
        for group in adapted.model.groups:
            torch.testing.assert_close(group, torch.arange(len(group)))

    def test_validation_targets_do_not_enter_optimizer_updates(self):
        changed = self.frame.copy()
        changed.loc[self.split - 14 * 24:, 'mw'] += 10000
        altered = task.prepare_comparison(changed, self.split, context_length=64, days=2)
        forecaster = SmallForecaster(SmallModel())
        with fake_timestral():
            first, first_history = task.fine_tune(forecaster, self.data, steps=2, batch_size=4)
            second, second_history = task.fine_tune(forecaster, altered, steps=2, batch_size=4)
        np.testing.assert_array_equal(first_history, second_history)
        torch.testing.assert_close(first.model.bias, second.model.bias, rtol=0, atol=0)
        self.assertEqual(first_history.fit_measurements, self.split - 14 * 24)


if __name__ == '__main__':
    unittest.main()
