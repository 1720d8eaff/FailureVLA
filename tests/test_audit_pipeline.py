import json
import time
import numpy as np
import pytest
from failurevla.strict_metrics import completed_summary
from failurevla.runtime import run
from failurevla.checkpoint import assemble
from failurevla.data import load_episode


def test_completed_metrics_reject_missing_and_mixed():
    with pytest.raises(ValueError):completed_summary([{'method':'B6'}])
    r=dict(method='B6',episode_id='1',source='synthetic_fixture',task_success=False,recovery_attempts=1,recovery_successes=0,inference_blocking_s=.1,task_time_s=1.)
    assert completed_summary([r])['task_success_rate']==0
    with pytest.raises(ValueError):completed_summary([r,{**r,'source':'real_robot','episode_id':'2'}])


def test_rgb_modalities_are_assembled_from_array_content():
    image=np.full((10,12,3),255,dtype=np.uint8)
    frame=assemble({'front':image},[.1,.2],'grasp')
    assert frame['observation.images.front'].shape==(3,10,12)
    assert frame['observation.images.front'].mean()==1.
    with pytest.raises(ValueError):assemble({'front':image.astype(float)},[0],'x')


def test_periodic_loop_records_real_deadlines(tmp_path):
    class Executor:
        def tick(self,obs,ins):return obs
        def close(self):self.closed=True
    e=Executor();received=[]
    rows=run(e,lambda:time.monotonic(),received.append,'x',4,tmp_path/'timing.csv',hz=100)
    assert len(received)==4 and e.closed
    assert rows[-1][2]-rows[0][2]>=.02
    assert (tmp_path/'timing.csv').is_file()
