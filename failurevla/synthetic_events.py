"""Loader for independent synthetic failure-event fixtures.

The mapping to the representation's coarser classes is an explicit test
convention; it is not a claim that the taxonomies are semantically identical.
"""
import csv
import math
import numpy as np

EVENT_TO_MODEL = {'normal':'none','grasp_slip':'grasp','missed_grasp':'grasp',
                  'pose_deviation':'execution_deviation','collision':'execution_deviation',
                  'timeout':'premature_termination','incomplete_execution':'premature_termination'}
FEATURES = ('ee_pose_error','object_pose_error','gripper_width','estimated_slip','action_residual')
PHASES = ('approach','grasp','lift','tilt','release')


def load_synthetic_events(path):
    with open(path, encoding='utf-8', newline='') as f:
        rows=list(csv.DictReader(f))
    if not rows: raise ValueError('no events')
    seen=set()
    for r in rows:
        if r.get('source')!='synthetic_sdv' or r.get('validation_status')!='passed_semantic_rules': raise ValueError('expected validated synthetic events')
        if r.get('failure_type') not in EVENT_TO_MODEL or r.get('phase') not in PHASES: raise ValueError('unknown type or phase')
        key=(r['episode_id'],int(r['step_id']))
        if key in seen: raise ValueError('duplicate event')
        seen.add(key)
        values=[float(r[k]) for k in (*FEATURES,'progress')]
        if not all(math.isfinite(v) for v in values) or not 0<=values[-1]<=1: raise ValueError('invalid numeric data')
    return rows


def model_inputs(rows):
    """Map scalar synthetic features to [batch, time=1, dim] tensors as arrays."""
    features=np.asarray([[float(r[k]) for k in FEATURES] for r in rows],dtype=np.float32)
    observation=features[:,:2,None].transpose(0,2,1)
    state=features[:,2:4,None].transpose(0,2,1)
    action=features[:,4:,None].transpose(0,2,1)
    return observation,state,action
