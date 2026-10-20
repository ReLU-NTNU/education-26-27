"""Reproduce the selected real-checkpoint experiment (optional Timestral extra).

Run from the repository root. This reports both the original test fortnight and
its following four weeks. Neither period selects settings or checkpoints.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from bootcamp_9 import timestral_task as task


def fingerprint(model):
    digest = hashlib.sha256()
    for parameter in model.parameters():
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optionally save results as JSON.')
    args = parser.parse_args()
    torch.set_num_threads(2)
    frame = pd.read_csv(ROOT / 'AEP_hourly.csv').rename(columns={'Datetime': 'date', 'AEP_MW': 'mw'})
    frame['date'] = pd.to_datetime(frame['date'])
    frame = frame.sort_values('date')
    frame = frame[(frame.date > '2007-01-01') & (frame.date < '2017-01-01')]
    data = task.prepare_comparison(frame, int(len(frame) * .8), days=42)
    pretrained = task.load_pretrained(ROOT / '.cache' / 'timestral')
    before = fingerprint(pretrained.model)
    started = time.monotonic()
    adapted, history = task.fine_tune(pretrained, data)
    elapsed = time.monotonic() - started
    assert fingerprint(pretrained.model) == before, 'Zero-shot weights changed.'
    assert fingerprint(adapted.model) != before, 'Fine-tuning did not update weights.'
    zero = task.forecast_days(pretrained, data)
    trained = task.forecast_days(adapted, data)
    report = {
        'model': task.MODEL_ID, 'model_revision': task.MODEL_REVISION,
        'torch_version': torch.__version__, 'seed': 42,
        'training': {'optimizer': 'AdamW', 'learning_rate': 1e-4,
                     'weight_decay': .01, 'batch_size': 16, 'epochs': 5,
                     'objective': 'central point forecast MSE / fixed training scale squared',
                     'recent_days': 730, 'validation_days_reserved': 84,
                     'fitting_windows': history.fitting_windows,
                     'validation_windows': history.validation_windows,
                     'validation_start': history.validation_start,
                     'validation_end': history.validation_end,
                     'selected_epoch': history.selected_epoch,
                     'initial_fitting_loss': history.initial_fitting_loss,
                     'zero_shot_validation_rmse': history.zero_shot_validation_rmse,
                     'checkpoints': history.validation},
        'periods': [],
    }
    for name, span in [('Weeks 1–2', slice(0, 14)), ('Weeks 3–6', slice(14, 42)),
                       ('All six weeks', slice(0, 42))]:
        scores = task.score_forecasts(data.actuals[span], {
            'Same hour yesterday': data.seasonal_naive[span],
            'Timestral-5M: zero-shot': zero[span],
            'Timestral-5M: fine-tuned': trained[span],
        })
        report['periods'].append({
            'period': name, 'start': str(data.dates[span][0, 0]),
            'end': str(data.dates[span][-1, -1]), 'scores': scores.to_dict(orient='records'),
        })
        print(f'\n{name}\n{scores.to_string(index=False)}')
    print(f'\nFine-tuning: {elapsed:.1f} seconds on CPU (2 threads).')
    # These are benchmark regressions for this pinned exercise, not a general
    # promise that fine-tuning helps every series or every evaluation period.
    for span in [slice(14, 42), slice(0, 42)]:
        assert np.mean((trained[span] - data.actuals[span]) ** 2) < np.mean((zero[span] - data.actuals[span]) ** 2)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
