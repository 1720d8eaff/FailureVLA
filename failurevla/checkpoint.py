"""Load SmolVLA with the checkpoint's own pre/postprocessing pipeline."""
import argparse
from pathlib import Path
import json
import numpy as np
import torch


class LoadedSmolVLA:
    def __init__(self,checkpoint,device='cpu'):
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors
        self.policy=SmolVLAPolicy.from_pretrained(checkpoint).to(device).eval()
        self.pre,self.post=make_pre_post_processors(self.policy.config,checkpoint,preprocessor_overrides={'device_processor':{'device':device}})

    def predict_action_chunk(self,frame):
        # predict_action_chunk is in normalized space; use checkpoint postprocessor
        # once per timestep to recover physical action units.
        with torch.inference_mode():
            chunk=self.policy.predict_action_chunk(self.pre(frame))
            if chunk.ndim!=3 or chunk.shape[0]!=1: raise ValueError('expected [1,horizon,action]')
            result=torch.stack([self.post(chunk[:,i]) for i in range(chunk.shape[1])],dim=1)
            if not torch.isfinite(result).all(): raise ValueError('nonfinite action')
            return result


def assemble(images,state,instruction):
    s=np.asarray(state,dtype=np.float32)
    if s.ndim!=1 or not np.isfinite(s).all() or not images or not instruction: raise ValueError('invalid observation')
    frame={'observation.state':torch.from_numpy(s),'task':instruction}
    for key,image in images.items():
        a=np.asarray(image)
        if a.ndim!=3 or a.shape[-1]!=3 or a.dtype!=np.uint8: raise ValueError('expected RGB uint8 HWC')
        frame['observation.images.'+key]=torch.from_numpy(a.copy()).permute(2,0,1).float()/255
    return frame


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--input',required=True);p.add_argument('--instruction',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');a=p.parse_args()
    with np.load(a.input,allow_pickle=False) as f: frame=assemble({k[6:]:f[k] for k in f.files if k.startswith('image.')},f['state'],a.instruction)
    model=LoadedSmolVLA(a.checkpoint,a.device);actions=model.predict_action_chunk(frame)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);np.save(out,actions.cpu().numpy())
    out.with_suffix('.json').write_text(json.dumps({'source':'checkpoint_inference','checkpoint':a.checkpoint,'shape':list(actions.shape),'postprocessor':'checkpoint pipeline'},indent=2))

if __name__=='__main__':main()
