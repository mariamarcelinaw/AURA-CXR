import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_branch_input_shapes_and_normalization():
 d=json.loads((ROOT/'reproducibility/deep_training_specification.json').read_text()); assert 'TorchXRayVision' in d['models']['densenet121_xrv']['input']['framework']; assert '[0.5,0.5,0.5]' in d['models']['eva_x_s']['input']['mean']; assert 'BGR' in d['models']['resnet50']['input']['scaling']
