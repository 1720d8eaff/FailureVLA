"""Deadline-scheduled executor loop with per-cycle telemetry and no hardware imports."""
import csv
import math
from pathlib import Path
import time


def run(executor,observe,consume,instruction,steps,output,hz=30.):
    if not math.isfinite(hz) or hz<=0 or steps<1: raise ValueError('invalid loop configuration')
    deadline=time.perf_counter();period=1/hz;rows=[]
    try:
        for step in range(steps):
            remaining=deadline-time.perf_counter()
            if remaining>0:time.sleep(remaining)
            started=time.perf_counter();observation=observe();action=executor.tick(observation,instruction)
            if action is not None:consume(action)
            finished=time.perf_counter()
            rows.append([step,deadline,started,finished,started-deadline,finished-started,action is not None])
            deadline+=period
    finally: executor.close()
    p=Path(output);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['step','scheduled_s','started_s','finished_s','lateness_s','work_s','action_available']);w.writerows(rows)
    return rows
