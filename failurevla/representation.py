"""Failure and progress heads for supervised training on labeled trajectories."""
import torch
from torch import nn
from torch.nn import functional as F

FAILURES = ('none', 'visual_localization', 'grasp', 'execution_deviation', 'target_change', 'premature_termination')


class FailureProgress(nn.Module):
    def __init__(self, observation_dim, state_dim, action_dim, hidden=128, stages=5):
        super().__init__()
        self.encoder = nn.GRU(observation_dim + state_dim + action_dim, hidden, batch_first=True)
        self.failure = nn.Linear(hidden, len(FAILURES))
        self.progress = nn.Linear(hidden, 1)
        self.stage = nn.Linear(hidden, stages)
        self.action = nn.Linear(hidden, action_dim)

    def forward(self, observation, state, action):
        if observation.shape[:2] != state.shape[:2] or state.shape[:2] != action.shape[:2]:
            raise ValueError('history batch and time dimensions must match')
        _, h = self.encoder(torch.cat((observation, state, action), dim=-1))
        h = h[-1]
        return {'failure_logits': self.failure(h), 'progress': self.progress(h).sigmoid().squeeze(-1),
                'stage_logits': self.stage(h), 'action': self.action(h)}


def supervised_loss(out, target_action, failure, progress, stage, lambda_failure=1., lambda_progress=1., lambda_stage=1.):
    if target_action.shape != out['action'].shape or progress.shape != out['progress'].shape:
        raise ValueError('target shapes do not match predictions')
    return (F.mse_loss(out['action'], target_action) + lambda_failure * F.cross_entropy(out['failure_logits'], failure)
            + lambda_progress * F.mse_loss(out['progress'], progress) + lambda_stage * F.cross_entropy(out['stage_logits'], stage))


def sampling_weights(labels, hard_weight=3.):
    if hard_weight <= 0:
        raise ValueError('hard_weight must be positive')
    labels = torch.as_tensor(labels)
    if labels.ndim != 1 or torch.any((labels < 0) | (labels >= len(FAILURES))):
        raise ValueError('invalid labels')
    return torch.where(labels == 0, 1., float(hard_weight)).double()
