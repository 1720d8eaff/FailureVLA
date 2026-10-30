"""Single-worker asynchronous inference with observable timing and exceptions."""
from concurrent.futures import ThreadPoolExecutor
from time import monotonic


class InferenceWorker:
    def __init__(self, policy):
        self.policy = policy
        self.executor = ThreadPoolExecutor(max_workers=1)

    def submit(self, observation, instruction, chunk_id):
        submitted = monotonic()
        def run():
            started = monotonic()
            chunk = self.policy.predict(observation, instruction, chunk_id)
            finished = monotonic()
            return chunk, {'submitted_at': submitted, 'started_at': started, 'finished_at': finished,
                           'inference_s': finished - started, 'queue_s': started - submitted}
        return self.executor.submit(run)

    def close(self):
        self.executor.shutdown(wait=True)
