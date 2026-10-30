"""Reproducible procedural fixture -> annotation -> training -> held-out smoke run."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import h5py
import numpy as np
from .labeled_data import prepare
from .representation import FAILURES


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='artifacts/synthetic_smoke');a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);data=out/'trajectories';data.mkdir(exist_ok=True)
    rng=np.random.default_rng(42);annotations=[]
    for i in range(12):
        name=f'fixture_{i:02d}.h5';T=28;t=np.arange(T)/30
        q=.1*np.sin(t[:,None]*np.arange(1,8)[None,:]+i);fingers=np.repeat(np.linspace(0,500,T)[:,None],6,axis=1)
        with h5py.File(data/name,'w') as f:
            for key,value in {'t':t,'joints':q,'fingers':fingers,'eef_left':rng.normal(size=(T,10))*.01,'eef_right':np.zeros((T,10)),'instruction':np.bytes_('procedural fixture')}.items():f[key]=value
            f.attrs['source']='synthetic_fixture'
        annotations.append({'episode':name,'source':'synthetic_fixture','frames':[{'failure':FAILURES[i%6],'progress':j/(T-1),'stage':min(4,j*5//T)} for j in range(T)]})
    labels=out/'annotations.jsonl';labels.write_text(''.join(json.dumps(x)+'\n' for x in annotations))
    prepare(data,labels,out/'splits',history=4,horizon=4,seed=42)
    subprocess.run([sys.executable,'-m','failurevla.train_failure_aware','--data',str(out/'splits/train.npz'),'--output',str(out/'synthetic_fixture.pt'),'--epochs','2'],check=True)
    subprocess.run([sys.executable,'-m','failurevla.heldout','--checkpoint',str(out/'synthetic_fixture.pt'),'--data',str(out/'splits/test.npz'),'--output',str(out/'heldout_metrics.json')],check=True)
    (out/'provenance.json').write_text(json.dumps({'source':'synthetic_fixture','seed':42,'scope':'procedural smoke test; no VLA checkpoint and no robot task success','epochs':2},indent=2))

if __name__=='__main__':main()
