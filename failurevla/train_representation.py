"""Train failure/progress heads on a user-supplied labeled NPZ training split."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from .representation import FailureProgress, sampling_weights, supervised_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='NPZ with observation, state, action, target_action, failure, progress, stage arrays')
    p.add_argument('--config', default='configs/training.json')
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if a.epochs < 1:
        p.error('epochs must be positive')
    config = json.loads(Path(a.config).read_text(encoding='utf-8'))
    with np.load(a.data, allow_pickle=False) as data:
        observation, state, action, target = (torch.tensor(data[k], dtype=torch.float32) for k in ('observation', 'state', 'action', 'target_action'))
        failure, progress, stage = (torch.tensor(data[k], dtype=d) for k, d in [('failure', torch.long), ('progress', torch.float32), ('stage', torch.long)])
    count = len(state)
    if count == 0 or any(len(t) != count for t in (observation, action, target, failure, progress, stage)):
        raise ValueError('sample counts must match and be nonzero')
    if state.ndim != 3 or observation.ndim != 3 or action.ndim != 3 or target.ndim != 2:
        raise ValueError('expected batched history arrays and action targets')
    if state.shape[1] != config['history_length']:
        raise ValueError('history length does not match configuration')
    if any(not torch.isfinite(t).all() for t in (observation, state, action, target, progress)):
        raise ValueError('training arrays must be finite')
    if torch.any((progress < 0) | (progress > 1)) or torch.any(stage < 0):
        raise ValueError('invalid progress or stage')
    torch.manual_seed(a.seed)
    model = FailureProgress(observation.shape[-1], state.shape[-1], action.shape[-1], stages=int(stage.max())+1)
    sampler = WeightedRandomSampler(sampling_weights(failure, config['hard_case_weight']), count, replacement=True,
                                    generator=torch.Generator().manual_seed(a.seed))
    loader = DataLoader(TensorDataset(observation, state, action, target, failure, progress, stage), batch_size=64, sampler=sampler)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(a.epochs):
        total = 0.
        for obs, st, act, target_act, f, prog, s in loader:
            optimizer.zero_grad()
            loss = supervised_loss(model(obs, st, act), target_act, f, prog, s,
                                   config['lambda_failure'], config['lambda_progress'], config['lambda_stage'])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(st)
        print(f'epoch={epoch+1} train_loss={total/count:.6f}')
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'config': config, 'observation_dim': observation.shape[-1],
                'state_dim': state.shape[-1], 'action_dim': action.shape[-1], 'stages': int(stage.max())+1}, out)


if __name__ == '__main__':
    main()
