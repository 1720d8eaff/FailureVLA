"""Supervised training on an already episode-separated NPZ split."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from .failure_aware import FailureAwareRefiner, refinement_loss


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    if args.epochs<1: p.error('epochs must be positive')
    keys=('observation','state','action','base_actions','target_actions','failure','progress','stage')
    with np.load(args.data,allow_pickle=False) as data:
        arrays=[np.asarray(data[k]) for k in keys]
    if not len(arrays[0]) or any(len(a)!=len(arrays[0]) for a in arrays) or any(not np.isfinite(a).all() for a in arrays): raise ValueError('invalid or nonfinite arrays')
    obs, state, action, base, target, failure, progress, stage=arrays
    if any(x.ndim!=3 for x in (obs,state,action,base,target)) or base.shape!=target.shape or base.shape[2]!=action.shape[2] or obs.shape[:2]!=state.shape[:2] or obs.shape[:2]!=action.shape[:2]: raise ValueError('invalid history or action shape')
    if any(x.ndim!=1 for x in (failure,progress,stage)) or np.any((failure<0)|(failure>=6)|(failure!=np.floor(failure))) or np.any((progress<0)|(progress>1)) or np.any((stage<0)|(stage>=5)|(stage!=np.floor(stage))): raise ValueError('invalid labels')
    torch.manual_seed(args.seed)
    tensors=[torch.tensor(a,dtype=torch.long if k in ('failure','stage') else torch.float32) for k,a in zip(keys,arrays)]
    loader=DataLoader(TensorDataset(*tensors),batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(args.seed))
    model=FailureAwareRefiner(obs.shape[-1],state.shape[-1],action.shape[-1])
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    for epoch in range(args.epochs):
        total=0.
        for ob,st,ac,ba,ta,fa,pr,sg in loader:
            optimizer.zero_grad(); loss=refinement_loss(model(ob,st,ac,ba),ta,fa,pr,sg);loss.backward();optimizer.step();total+=loss.item()*len(ob)
        print(f'epoch={epoch+1} train_loss={total/len(obs):.6f}')
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'model':model.state_dict(),'observation_dim':obs.shape[-1],'state_dim':state.shape[-1],'action_dim':action.shape[-1],'hidden':128,'stages':5,'residual_scale':model.residual_scale},out)


if __name__=='__main__': main()
