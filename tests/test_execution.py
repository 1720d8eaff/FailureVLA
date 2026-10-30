import unittest
from time import monotonic
from failurevla.contracts import RobotObservation
from failurevla.execution import Executor
from failurevla.policy import MockPolicy, SmolVLAAdapter


class ExecutionTest(unittest.TestCase):
    def test_all_modes(self):
        for mode in ('synchronous', 'naive_async', 'aligned_async'):
            obs = RobotObservation(monotonic(), [0.], None, None, 0.)
            policy = SmolVLAAdapter(MockPolicy([[1.], [2.]]), lambda o, i: {}, lambda a: a)
            e = Executor(policy, mode, max_age_s=5., replan_threshold=0)
            result = e.tick(obs, 'grasp')
            if mode != 'synchronous':
                e.pending[0].result(timeout=2)
                result = e.tick(obs, 'grasp')
            self.assertEqual(float(result[0]), 1.)
            self.assertEqual(e.received, 1)
            e.close()
