"""Prepare train/validation/test windows from explicit per-frame annotations.

No failure class is inferred from HDF5 shapes or a zero right-arm channel.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .data import load_episode,state
from .representation import FAILURES


def prepare(data_root,annotation_path,output,history=4,horizon=4,seed=42):
    annotations=[json.loads(x) for x in Path(annotation_path).read_text(encoding='utf-8').splitlines() if x.strip()]
    if len(annotations)<3 or history<1 or horizon<1: raise ValueError('need >=3 annotated episodes and positive windows')
    names=[a['episode'] for a in annotations]
    if len(set(names))!=len(names) or any(Path(n).name!=n for n in names): raise ValueError('duplicate or invalid episode names')
    order=np.random.default_rng(seed).permutation(len(names));n=max(1,len(names)//5)
    groups={'test':order[:n],'validation':order[n:2*n],'train':order[2*n:]}
    out=Path(output);out.mkdir(parents=True,exist_ok=True);manifest={}
    for split,indices in groups.items():
        rows={k:[] for k in ('observation','state','action','base_actions','target_actions','failure','progress','stage')}
        manifest[split]=[]
        for index in indices:
            a=annotations[index];ep=load_episode(Path(data_root)/a['episode']);s=state(ep);T=len(s)
            labels=a['frames']
            if len(labels)!=T or a.get('source') not in ('human_annotation','synthetic_fixture'): raise ValueError('annotation coverage or source invalid')
            f=np.asarray([FAILURES.index(x['failure']) for x in labels]);p=np.asarray([x['progress'] for x in labels],float);g=np.asarray([x['stage'] for x in labels])
            if not np.isfinite(p).all() or np.any((p<0)|(p>1)) or np.any((g<0)|(g>4)|(g!=g.astype(int))): raise ValueError('invalid labels')
            delta=np.diff(s,axis=0,prepend=s[:1])
            for i in range(history-1,T-horizon):
                rows['observation'].append(ep['eef_left'][i-history+1:i+1]);rows['state'].append(s[i-history+1:i+1]);rows['action'].append(delta[i-history+1:i+1])
                # Explicit last-delta baseline, not actions from a VLA checkpoint.
                rows['base_actions'].append(np.repeat(delta[i:i+1],horizon,axis=0));rows['target_actions'].append(delta[i+1:i+1+horizon])
                for key,value in [('failure',f[i]),('progress',p[i]),('stage',g[i])]: rows[key].append(value)
            manifest[split].append({'episode':a['episode'],'source':a['source'],'sha256':hashlib.sha256((Path(data_root)/a['episode']).read_bytes()).hexdigest()})
        if not rows['failure']: raise ValueError(f'no windows in {split}')
        np.savez(out/f'{split}.npz',**{k:np.asarray(v,dtype=np.int64 if k in ('failure','stage') else np.float32) for k,v in rows.items()})
    (out/'split_manifest.json').write_text(json.dumps({'seed':seed,'history':history,'horizon':horizon,'base_policy':'last_delta','annotation_sha256':hashlib.sha256(Path(annotation_path).read_bytes()).hexdigest(),'splits':manifest},indent=2))
    return manifest


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--annotations',required=True);p.add_argument('--output',required=True);p.add_argument('--history',type=int,default=4);p.add_argument('--horizon',type=int,default=4);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    prepare(a.data,a.annotations,a.output,a.history,a.horizon,a.seed)

if __name__=='__main__':main()
