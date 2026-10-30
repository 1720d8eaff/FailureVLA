"""Evaluate a trained refiner on held-out windows, never task success."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .failure_aware import FailureAwareRefiner
from .representation import FAILURES


def evaluate(checkpoint,data):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=FailureAwareRefiner(**{k:saved[k] for k in ('observation_dim','state_dim','action_dim','hidden','stages','residual_scale')});model.load_state_dict(saved['model']);model.eval()
    with np.load(data,allow_pickle=False) as f:
        x=[torch.from_numpy(f[k].astype(np.float32)) for k in ('observation','state','action','base_actions')];target=f['target_actions'];failure=f['failure'];progress=f['progress']
    if len(failure)==0: raise ValueError('empty held-out set')
    predictions=[];progs=[];actions=[]
    with torch.inference_mode():
        for start in range(0,len(failure),64):
            o=model(*(v[start:start+64] for v in x));predictions.extend(o['failure_logits'].argmax(-1).tolist());progs.extend(o['progress'].tolist());actions.extend(o['actions'].numpy())
    matrix=np.zeros((len(FAILURES),len(FAILURES)),dtype=int)
    for a,b in zip(failure,predictions):matrix[int(a),b]+=1
    return {'scope':'held-out labeled windows, not robot task success','samples':len(failure),'failure_accuracy':float(np.mean(failure==predictions)),'progress_mae':float(np.abs(progress-progs).mean()),'action_mse':float(np.mean((np.asarray(actions)-target)**2)),'base_action_mse':float(np.mean((x[-1].numpy()-target)**2)),'failure_classes':FAILURES,'confusion_matrix':matrix.tolist()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--data',required=True);p.add_argument('--output',required=True);a=p.parse_args();r=evaluate(a.checkpoint,a.data);out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(r,indent=2));print(json.dumps(r))

if __name__=='__main__':main()
