"""Thread-safe state-aligned chunk replacement for an external inference service."""
from collections import deque
from threading import Lock
import numpy as np


class ActionBuffer:
    def __init__(self, max_age_s, max_state_error):
        if not np.isfinite(max_age_s) or not np.isfinite(max_state_error) or max_age_s < 0 or max_state_error < 0:
            raise ValueError("thresholds must be finite and nonnegative")
        self.max_age_s, self.max_state_error = max_age_s, max_state_error
        self._actions = deque()
        self._last_request = -1
        self._lock = Lock()

    def replace(self, request_id, observed_at, now, observed_state, current_state, actions):
        """Accept only newer, timely, state-aligned chunks; timestamps share a monotonic clock."""
        a = np.asarray(actions, dtype=float)
        old, cur = np.asarray(observed_state, dtype=float), np.asarray(current_state, dtype=float)
        if old.ndim != 1 or old.size == 0 or old.shape != cur.shape or a.ndim != 2 or not len(a):
            raise ValueError("invalid state or action shape")
        if not np.isfinite(old).all() or not np.isfinite(cur).all() or not np.isfinite(a).all() or not np.isfinite([observed_at, now]).all():
            raise ValueError("non-finite input")
        with self._lock:
            if request_id <= self._last_request or not 0 <= now - observed_at <= self.max_age_s or np.linalg.norm(cur - old) > self.max_state_error:
                return False
            self._actions = deque(row.copy() for row in a)
            self._last_request = request_id
            return True

    def pop(self):
        with self._lock:
            return self._actions.popleft() if self._actions else None
