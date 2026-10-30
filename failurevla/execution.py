"""Policy execution scheduler: synchronous, naive async, or aligned async."""
from time import monotonic
from .action_buffer import ActionBuffer
from .worker import InferenceWorker
import numpy as np


class Executor:
    def __init__(self, policy, mode, max_age_s=.5, max_state_error=.1, replan_threshold=1):
        if mode not in {'synchronous', 'naive_async', 'aligned_async'}:
            raise ValueError('unknown mode')
        self.policy, self.mode, self.replan_threshold = policy, mode, replan_threshold
        self.buffer = ActionBuffer(max_age_s, max_state_error)
        self.worker = InferenceWorker(policy) if mode != 'synchronous' else None
        self.pending, self.request_id, self.naive = None, 0, []
        self.latencies, self.blocking_s, self.received, self.rejected = [], 0., 0, 0

    def tick(self, observation, instruction):
        now = monotonic()
        if self.pending is not None and self.pending[0].done():
            future, observed_state, request_id = self.pending
            chunk, timing = future.result()
            self.latencies.append(timing['inference_s'])
            self.received += 1
            if self.mode == 'aligned_async':
                accepted = self.buffer.replace(request_id, chunk.observation_timestamp, now,
                                               observed_state, observation.joint_positions, chunk.actions)
                self.rejected += int(not accepted)
            else:
                self.naive = list(chunk.actions)
            self.pending = None
        needs = (self.buffer.needs_replan(self.replan_threshold) if self.mode == 'aligned_async'
                 else len(self.naive) <= (0 if self.mode == 'synchronous' else self.replan_threshold))
        if needs and self.pending is None:
            request_id = self.request_id
            self.request_id += 1
            if self.mode == 'synchronous':
                start = monotonic()
                chunk = self.policy.predict(observation, instruction, str(request_id))
                elapsed = monotonic() - start
                self.blocking_s += elapsed
                self.latencies.append(elapsed)
                self.received += 1
                self.naive = list(chunk.actions)
            else:
                self.pending = (self.worker.submit(observation, instruction, str(request_id)),
                                np.asarray(observation.joint_positions, dtype=float).copy(), request_id)
        if self.mode == 'aligned_async':
            return self.buffer.pop(now=monotonic())
        return self.naive.pop(0) if self.naive else None

    def close(self):
        if self.worker:
            self.worker.close()
