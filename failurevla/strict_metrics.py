"""Strict completed-episode validation, separate from partial diagnostic metrics."""
import math
from .evaluation import summarize


def completed_summary(rows):
    if not rows:raise ValueError('no completed episodes')
    seen=set()
    for r in rows:
        required=('method','episode_id','source','task_success','recovery_attempts','recovery_successes','inference_blocking_s','task_time_s')
        if any(k not in r or r[k] is None for k in required):raise ValueError('incomplete episode record')
        key=(r['method'],r['episode_id'])
        if key in seen:raise ValueError('duplicate episode')
        seen.add(key)
        if r['source'] not in ('real_robot','mujoco_simulation','synthetic_fixture'):raise ValueError('unknown source')
        if r['task_success'] not in (0,1,False,True):raise ValueError('invalid task_success')
        for k in ('recovery_attempts','recovery_successes','inference_blocking_s','task_time_s'):
            if not math.isfinite(r[k]) or r[k]<0:raise ValueError('invalid measurement')
        if any(r[k]!=int(r[k]) for k in ('recovery_attempts','recovery_successes')) or r['recovery_successes']>r['recovery_attempts']:raise ValueError('invalid recovery counts')
    # Prevent combining synthetic and physical outcomes or different methods.
    if len({(r['source'],r['method']) for r in rows})!=1:raise ValueError('group by source and method first')
    return summarize(rows)
