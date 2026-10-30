import json
import h5py
import numpy as np
import pytest
from failurevla.labeled_data import prepare
from failurevla.data import load_episode


def fixture(root):
    records=[]
    for i in range(5):
        path=root/f'e{i}.h5';T=12
        with h5py.File(path,'w') as f:
            f['t']=np.arange(T)/30
            for k,d in [('joints',7),('fingers',6),('eef_left',10),('eef_right',10)]:f[k]=np.zeros((T,d))
            f['instruction']=np.bytes_('synthetic fixture')
        records.append({'episode':path.name,'source':'synthetic_fixture','frames':[{'failure':'none','progress':j/(T-1),'stage':0} for j in range(T)]})
    annotations=root/'labels.jsonl';annotations.write_text(''.join(json.dumps(x)+'\n' for x in records))
    return annotations,records


def test_real_loader_and_episode_split(tmp_path):
    labels,_=fixture(tmp_path);m=prepare(tmp_path,labels,tmp_path/'prepared',2,2)
    groups=[{r['episode'] for r in v} for v in m.values()]
    assert not groups[0]&groups[1] and not groups[1]&groups[2] and not groups[0]&groups[2]
    ep=load_episode(tmp_path/'e0.h5');assert ep['eef_right_available'] is False
    with np.load(tmp_path/'prepared/train.npz') as f:assert f['state'].shape[1:]==(2,13)


def test_unknown_annotations_rejected(tmp_path):
    labels,records=fixture(tmp_path)
    for r in records:r['frames'][0]['failure']='unreviewed'
    labels.write_text(''.join(json.dumps(x)+'\n' for x in records))
    with pytest.raises(ValueError):prepare(tmp_path,labels,tmp_path/'prepared',2,2)
