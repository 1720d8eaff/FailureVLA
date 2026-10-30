"""Policy adapter with explicit user-supplied observation and action transforms."""
from .contracts import ActionChunk


class SmolVLAAdapter:
    def __init__(self, policy, observation_encoder, action_decoder):
        self.policy = policy
        self.observation_encoder = observation_encoder
        self.action_decoder = action_decoder

    def predict(self, observation, instruction, chunk_id):
        batch = self.observation_encoder(observation, instruction)
        actions = self.policy.predict_action_chunk(batch)
        return ActionChunk(self.action_decoder(actions), observation.timestamp, str(chunk_id), 'SmolVLA')


class MockPolicy:
    """Deterministic interface test double, not model inference."""
    def __init__(self, actions):
        self.actions = actions

    def predict_action_chunk(self, batch):
        return self.actions
