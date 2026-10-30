"""History-conditioned action refinement and bounded recovery orchestration.

This module operates on policy-produced action chunks in the same normalized
action space used for training. It does not execute robot commands.
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from .contracts import ActionChunk
from .representation import FailureProgress, FAILURES


class FailureAwareRefiner(nn.Module):
    def __init__(self, observation_dim, state_dim, action_dim, hidden=128, stages=5, residual_scale=0.1):
        super().__init__()
        if min(observation_dim, state_dim, action_dim, hidden, stages) <= 0 or residual_scale <= 0:
            raise ValueError('dimensions and residual_scale must be positive')
        self.head = FailureProgress(observation_dim, state_dim, action_dim, hidden, stages)
        self.residual_scale = float(residual_scale)
        self.adapter = nn.Sequential(nn.Linear(action_dim + len(FAILURES) + 1, hidden), nn.ReLU(), nn.Linear(hidden, action_dim))
        # Zero initialization preserves base actions until the adapter is trained.
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, observation_history, state_history, action_history, base_actions):
        if base_actions.ndim != 3 or base_actions.shape[0] != state_history.shape[0] or base_actions.shape[-1] != action_history.shape[-1] or base_actions.shape[1] < 1:
            raise ValueError('base actions must be [batch, horizon, action_dim]')
        result = self.head(observation_history, state_history, action_history)
        probs = result['failure_logits'].softmax(-1)
        context = torch.cat((probs, result['progress'].unsqueeze(-1)), dim=-1)
        context = context.unsqueeze(1).expand(-1, base_actions.shape[1], -1)
        residual = self.residual_scale * torch.tanh(self.adapter(torch.cat((base_actions, context), dim=-1)))
        return {**result, 'actions': base_actions + residual, 'failure_probabilities': probs, 'residual': residual}


def refinement_loss(output, target_actions, failure, progress, stage, *, failure_weight=1., progress_weight=1., stage_weight=1.):
    if target_actions.shape != output['actions'].shape or progress.shape != output['progress'].shape:
        raise ValueError('target shapes do not match output')
    return (F.mse_loss(output['actions'], target_actions)
            + failure_weight * F.cross_entropy(output['failure_logits'], failure)
            + progress_weight * F.mse_loss(output['progress'], progress)
            + stage_weight * F.cross_entropy(output['stage_logits'], stage))


@dataclass(frozen=True)
class Prediction:
    chunk: ActionChunk
    failure_type: str
    failure_probability: float
    progress: float
    recovery_decision: str


class FailureAwareSmolVLA:
    """Wrap a SmolVLA-compatible policy and user-supplied data transforms.

    history_encoder returns three torch tensors of shape [1, time, dimension];
    action_decoder accepts refined tensor [1, horizon, action_dim] and returns
    an action sequence. Supply a fresh RecoveryController for each episode.
    """
    def __init__(self, policy, refiner, recovery, observation_encoder, history_encoder, action_decoder):
        self.policy, self.refiner, self.recovery = policy, refiner, recovery
        self.observation_encoder, self.history_encoder, self.action_decoder = observation_encoder, history_encoder, action_decoder

    def predict_with_diagnostics(self, observation, instruction, chunk_id):
        with torch.inference_mode():
            batch = self.observation_encoder(observation, instruction)
            base = self.policy.predict_action_chunk(batch)
            histories = self.history_encoder(observation, instruction)
            output = self.refiner(*histories, base)
            if output['actions'].shape[0] != 1 or not torch.isfinite(output['actions']).all() or not torch.isfinite(output['failure_probabilities']).all() or not torch.isfinite(output['progress']).all():
                raise ValueError('invalid single-observation policy output')
            probability, index = output['failure_probabilities'][0].max(dim=-1)
            failure = FAILURES[int(index)]
            decision = self.recovery.decide(failure, float(probability))
            actions = self.action_decoder(output['actions'])
            chunk = ActionChunk(actions, observation.timestamp, str(chunk_id), 'FailureAwareSmolVLA')
            return Prediction(chunk, failure, float(probability), float(output['progress'][0]), decision)

    def predict(self, observation, instruction, chunk_id):
        """Executor-compatible path: callers must handle non-continue decisions."""
        result = self.predict_with_diagnostics(observation, instruction, chunk_id)
        if result.recovery_decision != 'continue':
            raise RuntimeError(f'recovery decision requires handling: {result.recovery_decision}')
        return result.chunk
