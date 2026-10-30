import unittest
import torch
from types import SimpleNamespace
from failurevla.contracts import RobotObservation
from failurevla.policy import MockPolicy, SmolVLAAdapter
from failurevla.worker import InferenceWorker
from failurevla.recovery import RecoveryController
from failurevla.representation import FailureProgress, supervised_loss, sampling_weights
from failurevla.evaluation import summarize
from robot_backends.unitree_grasp_backend import UnitreeGraspBackend


class ComponentsTest(unittest.TestCase):
    def test_representation_gradient(self):
        model = FailureProgress(2, 3, 2, hidden=8, stages=3)
        out = model(torch.zeros(4, 5, 2), torch.zeros(4, 5, 3), torch.zeros(4, 5, 2))
        loss = supervised_loss(out, torch.zeros(4, 2), torch.zeros(4, dtype=torch.long), torch.zeros(4), torch.zeros(4, dtype=torch.long))
        loss.backward()
        self.assertIsNotNone(model.encoder.weight_ih_l0.grad)
        self.assertEqual(sampling_weights([0, 2]).tolist(), [1., 3.])

    def test_worker_and_recovery(self):
        obs = RobotObservation(2., [0.], None, None, 0.)
        adapter = SmolVLAAdapter(MockPolicy([[1.]]), lambda o, i: (o, i), lambda x: x)
        worker = InferenceWorker(adapter)
        chunk, timing = worker.submit(obs, 'grasp', 4).result(timeout=2)
        worker.close()
        self.assertEqual(chunk.observation_timestamp, 2.)
        self.assertGreaterEqual(timing['inference_s'], 0)
        recovery = RecoveryController('rule', 1)
        self.assertEqual(recovery.decide('grasp', .9), 'regrasp')
        self.assertEqual(recovery.decide('grasp', .9), 'terminate')

    def test_metrics_and_backend(self):
        result = summarize([{'task_success': 1, 'failure_tp': 1, 'failure_fp': 1, 'recovery_attempts': 2, 'recovery_successes': 1}])
        self.assertEqual(result['failure_precision'], .5)
        self.assertEqual(result['recovery_success_rate'], .5)
        self.assertIsNone(result['grasp_success_rate'])
        bridge = SimpleNamespace(observe=lambda: 'observation', plan_recovery=lambda req: (req.failure_type, req.observation))
        backend = UnitreeGraspBackend(bridge)
        self.assertEqual(backend.plan_grasp_recovery('grasp', 1), ('grasp', 'observation'))
        self.assertIsNone(backend.plan_grasp_recovery('grasp', 0))
