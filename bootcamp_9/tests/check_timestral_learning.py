"""Optional real-checkpoint overfit diagnostic; never evaluates test targets.

Run from the repo root:
uv run --locked --project bootcamp_9 --extra timestral python bootcamp_9/tests/check_timestral_learning.py
"""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from bootcamp_9 import timestral_task as task


def main():
    torch.set_num_threads(2)
    frame = pd.read_csv(ROOT / 'AEP_hourly.csv').rename(columns={'Datetime': 'date', 'AEP_MW': 'mw'})
    frame['date'] = pd.to_datetime(frame['date'])
    frame = frame.sort_values('date')
    frame = frame[(frame.date > '2007-01-01') & (frame.date < '2017-01-01')]
    data = task.prepare_comparison(frame, int(len(frame) * 0.8))
    fitting, _ = task.split_validation(data)
    inputs, targets, _ = task.training_windows(fitting)
    indices = torch.linspace(0, len(inputs) - 1, 8).long()
    x, y = inputs[indices], targets[indices]
    pretrained = task.load_pretrained(ROOT / '.cache' / 'timestral')
    model = deepcopy(pretrained.model).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    scale = max(float(np.std(fitting.train_values)), 1.)
    initial_rmse = None
    for step in range(41):
        optimizer.zero_grad(set_to_none=True)
        quantiles = task.training_quantiles(model, x, 24)
        loss = task.point_loss(model, x, y, scale)
        rmse = float(((quantiles[:, :, model.config.median_index] - y) ** 2).mean().sqrt().detach())
        if step == 0:
            initial_rmse = rmse
        if step in [0, 10, 40]:
            print(f"Fixed-batch update {step}: loss {float(loss.detach()):.4f}; RMSE {rmse:.2f} MW")
        if step == 40:
            break
        loss.backward()
        if step == 0:
            nonzero = sum(p.grad is not None and bool(p.grad.abs().max() > 0) for p in model.parameters())
            total = sum(1 for p in model.parameters())
            print(f"Nonzero parameter gradients: {nonzero}/{total}")
            assert nonzero == total, 'Some model parameters receive no gradient.'
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        assert torch.isfinite(norm) and norm > 0
        optimizer.step()
    assert np.isfinite(rmse) and rmse < initial_rmse * 0.2, 'Cannot substantially overfit eight training windows.'
    print('Real-checkpoint learning diagnostic passed; validation and test targets were not used.')


if __name__ == '__main__':
    main()
