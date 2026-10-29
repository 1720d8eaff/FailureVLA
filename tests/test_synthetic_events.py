from pathlib import Path
import pytest
import torch
from failurevla.synthetic_events import load_synthetic_events, model_inputs, EVENT_TO_MODEL
from failurevla.representation import FailureProgress, FAILURES
from failurevla.recovery import RecoveryController

FIXTURE=Path(__file__).parent/'fixtures/synthetic_failure_events.csv'


def test_real_failure_head_accepts_synthetic_fixture():
    rows=load_synthetic_events(FIXTURE)
    obs,state,action=model_inputs(rows)
    output=FailureProgress(2,2,1,hidden=8)(torch.from_numpy(obs),torch.from_numpy(state),torch.from_numpy(action))
    assert output['failure_logits'].shape==(len(rows),len(FAILURES))
    assert output['progress'].shape==(len(rows),)
    assert torch.isfinite(output['failure_logits']).all()
    assert all(EVENT_TO_MODEL[r['failure_type']] in FAILURES for r in rows)


@pytest.mark.parametrize('kind,decision', [('normal','continue'),('grasp_slip','regrasp'),('missed_grasp','regrasp'),('pose_deviation','replan'),('collision','replan'),('timeout','replan'),('incomplete_execution','replan')])
def test_real_recovery_policy_on_each_event_type(kind,decision):
    row=next(r for r in load_synthetic_events(FIXTURE) if r['failure_type']==kind)
    controller=RecoveryController('rule',budget=1)
    assert controller.decide(EVENT_TO_MODEL[row['failure_type']],.9)==decision
    assert controller.budget==(1 if kind=='normal' else 0)
