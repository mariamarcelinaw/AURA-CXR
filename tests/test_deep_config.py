import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_deep_spec_models():
 d=json.loads((ROOT/'reproducibility/deep_training_specification.json').read_text()); assert set(d['models'])=={'efficientnetv2s','resnet50','densenet121_xrv','eva_x_s'}
def test_selected_refit_epochs():
 d=json.loads((ROOT/'reproducibility/deep_training_specification.json').read_text()); assert d['models']['densenet121_xrv']['final_refit']['derived_epoch_phase1']==2; assert d['models']['eva_x_s']['final_refit']['derived_epoch_phase1']==2
