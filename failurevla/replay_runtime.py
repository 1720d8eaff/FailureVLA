"""Run the actual Executor at 30 Hz with recorded trajectory chunks.

Injected delay is artificial, not a model inference measurement.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from .data import load_episode,state
from .contracts import RobotObservation,ActionChunk
from .execution import Executor
from .runtime import run


def main():
    p=argparse.ArgumentParser();p.add_argument('--episode',required=True);p.add_argument('--output',default='artifacts/replay');p.add_argument('--steps',type=int,default=60);p.add_argument('--delay-s',type=float,default=.04);a=p.parse_args()
    if a.delay_s<0 or not np.isfinite(a.delay_s):raise ValueError('invalid delay')
    ep=load_episode(a.episode);states=state(ep);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);summary={}
    for mode in ('synchronous','naive_async','aligned_async'):
        clock=[0]
        def observe():
            i=clock[0]%len(states);clock[0]+=1
            return RobotObservation(time.monotonic(),states[i,:7].tolist(),None,None,float(states[i,7:].mean()))
        class Replay:
            def predict(self,observation,instruction,chunk_id):
                time.sleep(a.delay_s)
                index=int(np.argmin(np.linalg.norm(states[:,:7]-observation.joint_positions,axis=1)))
                chunk=states[index:min(index+8,len(states)),:7]
                return ActionChunk(chunk,observation.timestamp,str(chunk_id),'recorded_trajectory_replay')
        executor=Executor(Replay(),mode,max_age_s=.2)
        rows=run(executor,observe,lambda action:None,'trajectory replay',a.steps,out/f'{mode}.csv')
        summary[mode]={'ticks':len(rows),'lateness_p50_s':float(np.quantile([r[4] for r in rows],.5)),'lateness_p95_s':float(np.quantile([r[4] for r in rows],.95)),'work_p95_s':float(np.quantile([r[5] for r in rows],.95)),'inference_count':executor.received,'chunks_rejected':executor.rejected}
    (out/'summary.json').write_text(json.dumps({'source':'recorded_trajectory_replay','injected_delay_s':a.delay_s,'target_hz':30,'episode':Path(a.episode).name,'results':summary},indent=2));print(json.dumps(summary))

if __name__=='__main__':main()
