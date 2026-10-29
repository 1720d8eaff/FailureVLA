import unittest
import torch
from failurevla.contracts import RobotObservation
from failurevla.failure_aware import FailureAwareRefiner, FailureAwareSmolVLA, refinement_loss
from failurevla.recovery import RecoveryController


class Policy:
    def predict_action_chunk(self, batch): return torch.zeros(1, 4, 2)


class FailureAwareTests(unittest.TestCase):
    def test_trainable_fusion_and_bounded_residual(self):
        torch.manual_seed(7)
        model=FailureAwareRefiner(2,3,2,hidden=8,residual_scale=.2)
        x=(torch.randn(2,4,2),torch.randn(2,4,3),torch.randn(2,4,2),torch.zeros(2,5,2))
        out=model(*x)
        self.assertTrue(torch.equal(out['actions'],x[-1]))
        loss=refinement_loss(out,torch.ones_like(out['actions']),torch.tensor([0,1]),torch.tensor([.1,.8]),torch.tensor([0,1]))
        loss.backward()
        self.assertGreater(model.adapter[-1].weight.grad.abs().sum().item(),0)
        self.assertGreater(model.head.failure.weight.grad.abs().sum().item(),0)
        with torch.no_grad(): model.adapter[-1].weight.fill_(10)
        self.assertLessEqual(model(*x)['residual'].abs().max().item(),.200001)

    def test_policy_recovery_and_budget(self):
        model=FailureAwareRefiner(2,3,2,hidden=8)
        with torch.no_grad():
            model.head.failure.weight.zero_();model.head.failure.bias.fill_(-10);model.head.failure.bias[2]=10
        history=lambda obs,ins:(torch.zeros(1,3,2),torch.zeros(1,3,3),torch.zeros(1,3,2))
        wrapper=FailureAwareSmolVLA(Policy(),model,RecoveryController('rule',1),lambda o,i:{},history,lambda x:x[0].tolist())
        observation=RobotObservation(12.,[0.],None,None,0.)
        result=wrapper.predict_with_diagnostics(observation,'grasp',1)
        self.assertEqual(result.failure_type,'grasp')
        self.assertEqual(result.recovery_decision,'regrasp')
        self.assertEqual(result.chunk.observation_timestamp,12.)
        self.assertEqual(wrapper.predict_with_diagnostics(observation,'grasp',2).recovery_decision,'terminate')
        with self.assertRaises(RuntimeError): wrapper.predict(observation,'grasp',3)
