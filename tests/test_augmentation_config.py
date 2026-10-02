import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_aug_scope():
 d=json.loads((ROOT/'reproducibility/deep_training_specification.json').read_text()); assert len(d['models']['efficientnetv2s']['augmentation'])==5; assert d['models']['densenet121_xrv']['augmentation']==[]; assert d['models']['eva_x_s']['augmentation']==[]
